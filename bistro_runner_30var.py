"""
BISTRO 2-Stage Feature Selection Pipeline
==========================================
Stage 1: 전체 29개 변수로 추론 → Attention 기반 변수 랭킹 → 상위 K개 선택
Stage 2: 선택된 K개 변수만으로 재추론 → Attention Map + CF 분석

결과: data/real_inference_results.npz (Stage 2 기준, 대시보드용)
      data/stage1_screening.npz     (Stage 1 전체 랭킹, 참고용)

실행:
    .venv-bistro/bin/python3 bistro_runner_30var.py [--top-k 10]
"""

import os
import sys
import math
import argparse
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Repo 경로 설정 ────────────────────────────────────────────
BISTRO_REPO   = "/tmp/bistro-repo"
MODEL_PATH    = f"{BISTRO_REPO}/bistro-finetuned"
DATA_DIR      = f"{BISTRO_REPO}/data"
PANEL_CSV       = os.path.join(DATA_DIR, "macro_panel.csv")
PANEL_DAILY_CSV = os.path.join(DATA_DIR, "macro_panel_daily.csv")
FREQ_CSV        = os.path.join(DATA_DIR, "variable_freq.csv")
OUTPUT_DIR      = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE     = os.path.join(OUTPUT_DIR, "real_inference_results.npz")
STAGE1_FILE     = os.path.join(OUTPUT_DIR, "stage1_screening.npz")

sys.path.insert(0, f"{BISTRO_REPO}/src")

# ── 추론 설정 ─────────────────────────────────────────────────
FREQ      = "M"
CTX       = 120
PDT       = 12
PSZ       = 32
BSZ       = 32
WINDOW_DISTANCE = 12
FORECAST_START_DATE = "2023-01-01"
TARGET_COL = "CPI_KR_YoY"


# ============================================================
# Data
# ============================================================

def load_panel(covariate_subset=None):
    """macro_panel.csv 로딩. covariate_subset 지정 시 해당 변수만 사용."""
    panel = pd.read_csv(PANEL_CSV, index_col=0)
    panel.index = pd.PeriodIndex(panel.index, freq=FREQ)

    all_covariates = [c for c in panel.columns if c != TARGET_COL]

    if covariate_subset is not None:
        missing = [c for c in covariate_subset if c not in all_covariates]
        if missing:
            raise ValueError(f"Missing covariates: {missing}")
        covariates = covariate_subset
    else:
        covariates = all_covariates

    panel = panel[[TARGET_COL] + covariates]
    panel = panel.ffill().bfill()
    return panel, covariates


def load_daily_panel(covariate_subset=None):
    """macro_panel_daily.csv 로딩 (일별). covariate_subset 지정 시 해당 변수만."""
    panel = pd.read_csv(PANEL_DAILY_CSV, index_col=0, parse_dates=True)

    all_covariates = [c for c in panel.columns if c != TARGET_COL]

    if covariate_subset is not None:
        missing = [c for c in covariate_subset if c not in all_covariates]
        if missing:
            raise ValueError(f"Missing covariates: {missing}")
        covariates = covariate_subset
    else:
        covariates = all_covariates

    panel = panel[[TARGET_COL] + covariates]
    panel = panel.ffill().bfill()
    return panel, covariates


def load_variable_freq():
    """변수별 원본 주기 메타데이터 로딩."""
    if os.path.exists(FREQ_CSV):
        df = pd.read_csv(FREQ_CSV)
        return dict(zip(df["variable"], df["original_freq"]))
    return {}


# ============================================================
# Daily Panel Inference
# ============================================================

def run_bistro_inference_daily(daily_panel, covariates, module, n_layers, capture_attention=True):
    """
    일별 패널을 직접 받아 BISTRO 추론.
    monthly 전처리의 expand_monthly_to_daily 단계를 건너뛰고
    이미 일별인 데이터를 직접 투입.
    """
    from preprocessing_util import aggregate_daily_forecast_to_monthly
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast

    n_variates = 1 + len(covariates)

    # ── 일별 DataFrame 준비 ────────────────────────────────
    df = daily_panel[[TARGET_COL] + covariates].copy()
    df.columns = ["target"] + covariates
    df = df.sort_index()

    # forecast 시작일 기준 cutoff
    forecast_start_dt = pd.Timestamp(FORECAST_START_DATE)
    cutoff_dt = forecast_start_dt - pd.Timedelta(days=1)

    # context / prediction steps (일 단위)
    pdt_steps = PSZ * PDT   # 32 × 12 = 384 일
    ctx_steps = PSZ * CTX   # 32 × 120 = 3840 일
    dist_steps = PSZ * WINDOW_DISTANCE

    # cutoff 이후 데이터 + 패딩 확보
    df_after_cutoff = df.loc[forecast_start_dt:]
    n_needed = pdt_steps
    if len(df_after_cutoff) < n_needed:
        # 미래 패딩 (marker values)
        last_date = df.index[-1]
        pad_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                  periods=n_needed - len(df_after_cutoff), freq="D")
        pad_vals = [-1 if i % 2 == 0 else -2 for i in range(len(pad_dates))]
        pad_df = pd.DataFrame({"target": pad_vals}, index=pad_dates)
        df = pd.concat([df, pad_df])
        df = df[~df.index.duplicated(keep="first")].sort_index()

    # item_id 추가
    df["item_id"] = "cpi_kr_yoy"

    cutoff_period_daily = pd.Period(cutoff_dt.strftime("%Y-%m-%d"), freq="D")

    ds = PandasDataset.from_long_dataframe(
        df, item_id="item_id",
        past_feat_dynamic_real=covariates, feat_dynamic_real=[],
    )
    _, test_template = split(ds, date=cutoff_period_daily)
    test_data = test_template.generate_instances(
        prediction_length=pdt_steps, windows=1,
        distance=dist_steps, max_history=ctx_steps,
    )

    # Attention Hook
    captured = {}
    restore_fn = lambda: None
    if capture_attention:
        captured, restore_fn = install_attention_hooks(module, n_layers)

    model = MoiraiForecast(
        module=module,
        prediction_length=pdt_steps,
        context_length=ctx_steps,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    predictor = model.create_predictor(batch_size=BSZ)

    inputs    = list(test_data.input)
    labels    = list(test_data.label)
    forecasts = list(predictor.predict(test_data.input))
    restore_fn()

    # 집계 (daily → monthly)
    samples      = np.asarray(forecasts[0].samples, dtype=float)
    label_target = np.asarray(labels[0]["target"], dtype=float)
    inp_target   = np.asarray(inputs[0]["target"], dtype=float)
    last_input   = float(inp_target[-1]) if inp_target.size > 0 else None

    preds_baseline, _, _ = aggregate_daily_forecast_to_monthly(
        samples, np.zeros(PDT * PSZ), last_input,
        steps_per_period=PSZ, expected_periods=PDT,
    )
    preds, gts, ci = aggregate_daily_forecast_to_monthly(
        samples, label_target, last_input,
        steps_per_period=PSZ, expected_periods=PDT,
    )

    forecast_start = pd.Period(FORECAST_START_DATE, freq=FREQ)
    pred_index = pd.period_range(start=forecast_start, periods=PDT, freq=FREQ)

    forecast_df = pd.DataFrame({
        "date":       [str(p) for p in pred_index],
        "bistro_med": preds,
        "ci_lo":      ci[:, 0],
        "ci_hi":      ci[:, 1],
    })

    # AR(1)
    # 월별 CPI를 일별 패널에서 추출 (월말값)
    cpi_daily = daily_panel[[TARGET_COL]]
    cpi_monthly = cpi_daily.resample("M").last()
    cpi_monthly.index = cpi_monthly.index.to_period("M")
    cpi_monthly.columns = ["cpi_yoy"]
    try:
        from inference_util import ar1_forecast
        train_y = cpi_monthly["cpi_yoy"].loc[:pred_index[0] - 1].tail(CTX).astype(float)
        ar1 = ar1_forecast(train_y, pred_index, method="ols", trend="c", validate_index=True)
        forecast_df["ar1"] = ar1.values
    except Exception as e:
        print(f"    AR(1) skipped: {e}")
        forecast_df["ar1"] = np.nan

    actual_reindexed = cpi_monthly["cpi_yoy"].reindex(pred_index)
    forecast_df["actual"] = actual_reindexed.values

    layer_names = sorted(captured.keys(), key=lambda x: int(x.split(".")[2])) if captured else []

    # prep 호환 객체 생성 (기존 코드와 인터페이스 유지)
    class DailyPrepCompat:
        def __init__(self):
            self.forecast_start = forecast_start
            self.train_end = forecast_start - 1
            self.pdt_steps = pdt_steps
            self.ctx_steps = ctx_steps
            self.dist_steps = dist_steps
            self.cutoff_period_daily = cutoff_period_daily
            self.daily_long_df = df

    prep = DailyPrepCompat()

    return forecast_df, preds_baseline, prep, predictor, captured, layer_names


# ============================================================
# Attention Hooks
# ============================================================

_orig_sdpa = None  # 원본 SDPA 참조 보관

def install_attention_hooks(module, n_layers=12):
    global _orig_sdpa
    import torch
    import torch.nn.functional as F

    captured      = {}
    current_layer = [-1]
    hooks         = []

    for i in range(n_layers):
        def make_pre(idx):
            def pre_hook(mod, args):
                current_layer[0] = idx
            return pre_hook
        h = module.encoder.layers[i].self_attn.register_forward_pre_hook(make_pre(i))
        hooks.append(h)

    if _orig_sdpa is None:
        _orig_sdpa = F.scaled_dot_product_attention

    def _hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, scale=None,
                     is_causal=False, **kw):
        if current_layer[0] >= 0:
            sc = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))
            with torch.no_grad():
                raw = torch.matmul(query.float(), key.float().transpose(-2, -1)) * sc
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        raw = raw.masked_fill(~attn_mask, float("-inf"))
                    else:
                        raw = raw + attn_mask.float()
                w = torch.softmax(raw, dim=-1)
                avg = w.mean(dim=tuple(range(w.dim() - 2)))
                layer_name = f"encoder.layers.{current_layer[0]}.self_attn"
                captured[layer_name] = avg.cpu().numpy()

        return _orig_sdpa(query, key, value,
                          attn_mask=attn_mask, dropout_p=dropout_p,
                          scale=scale, is_causal=is_causal, **kw)

    import torch.nn.functional as _F_mod
    _F_mod.scaled_dot_product_attention = _hooked_sdpa
    torch.nn.functional.scaled_dot_product_attention = _hooked_sdpa

    def restore():
        for h in hooks:
            h.remove()
        torch.nn.functional.scaled_dot_product_attention = _orig_sdpa

    return captured, restore


# ============================================================
# Core Inference (재사용)
# ============================================================

def run_bistro_inference(panel, covariates, module, n_layers, capture_attention=True):
    """
    BISTRO 추론 실행.

    Returns: (forecast_df, preds_baseline, prep, predictor, captured, layer_names)
    """
    from preprocessing_util import (
        prepare_long_df_monthly_for_daily_inference,
        aggregate_daily_forecast_to_monthly,
    )
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast

    n_variates = 1 + len(covariates)

    # DataFrame 준비
    df = panel[[TARGET_COL] + covariates].copy()
    df.columns = ["target"] + covariates
    df["item_id"] = "cpi_kr_yoy"

    prep = prepare_long_df_monthly_for_daily_inference(
        df,
        item_id_col="item_id",
        target_col="target",
        past_dynamic_real_cols=covariates,
        freq=FREQ,
        forecast_start_date=FORECAST_START_DATE,
        pdt_patches=PDT,
        ctx_patches=CTX,
        steps_per_period=PSZ,
        rolling_windows=1,
        window_distance_patches=WINDOW_DISTANCE,
    )

    ds = PandasDataset.from_long_dataframe(
        prep.daily_long_df,
        item_id="item_id",
        past_feat_dynamic_real=covariates,
        feat_dynamic_real=[],
    )
    train, test_template = split(ds, date=prep.cutoff_period_daily)
    test_data = test_template.generate_instances(
        prediction_length=prep.pdt_steps,
        windows=1,
        distance=prep.dist_steps,
        max_history=prep.ctx_steps,
    )

    # Attention Hook
    captured = {}
    restore_fn = lambda: None
    if capture_attention:
        captured, restore_fn = install_attention_hooks(module, n_layers)

    model = MoiraiForecast(
        module=module,
        prediction_length=prep.pdt_steps,
        context_length=prep.ctx_steps,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    predictor = model.create_predictor(batch_size=BSZ)

    inputs    = list(test_data.input)
    labels    = list(test_data.label)
    forecasts = list(predictor.predict(test_data.input))
    restore_fn()

    # 집계
    samples      = np.asarray(forecasts[0].samples, dtype=float)
    label_target = np.asarray(labels[0]["target"], dtype=float)
    inp_target   = np.asarray(inputs[0]["target"], dtype=float)
    last_input   = float(inp_target[-1]) if inp_target.size > 0 else None

    preds_baseline, _, _ = aggregate_daily_forecast_to_monthly(
        samples, np.zeros(PDT * PSZ), last_input,
        steps_per_period=PSZ, expected_periods=PDT,
    )
    preds, gts, ci = aggregate_daily_forecast_to_monthly(
        samples, label_target, last_input,
        steps_per_period=PSZ, expected_periods=PDT,
    )

    pred_index = pd.period_range(start=prep.forecast_start, periods=PDT, freq=FREQ)

    forecast_df = pd.DataFrame({
        "date":       [str(p) for p in pred_index],
        "bistro_med": preds,
        "ci_lo":      ci[:, 0],
        "ci_hi":      ci[:, 1],
    })

    # AR(1)
    cpi_monthly = panel[[TARGET_COL]]
    cpi_monthly.columns = ["cpi_yoy"]
    try:
        from inference_util import ar1_forecast
        train_y = cpi_monthly["cpi_yoy"].loc[:pred_index[0] - 1].tail(CTX).astype(float)
        ar1 = ar1_forecast(train_y, pred_index, method="ols", trend="c", validate_index=True)
        forecast_df["ar1"] = ar1.values
    except Exception as e:
        print(f"    AR(1) skipped: {e}")
        forecast_df["ar1"] = np.nan

    # Actual
    actual_reindexed = cpi_monthly["cpi_yoy"].reindex(pred_index)
    forecast_df["actual"] = actual_reindexed.values

    layer_names = sorted(captured.keys(), key=lambda x: int(x.split(".")[2])) if captured else []

    return forecast_df, preds_baseline, prep, predictor, captured, layer_names


# ============================================================
# Attention Ranking
# ============================================================

def compute_attention_ranking(captured, layer_names, variates, n_variates):
    """Attention 기반 공변량 랭킹 반환."""
    from bistro_core import BISTROConfig, AttentionHookManager, AttentionAnalyzer

    attn_arrays = np.stack([captured[k] for k in layer_names], axis=0)
    ctx_patches = attn_arrays.shape[-1] // n_variates

    cfg = BISTROConfig(
        variates=variates, target_idx=0,
        ctx_patches=ctx_patches, pdt_patches=PDT, patch_size=PSZ,
    )
    hook_mgr = AttentionHookManager()
    for i, name in enumerate(layer_names):
        hook_mgr.attention_maps[name] = attn_arrays[i][np.newaxis, np.newaxis, ...]
    ana = AttentionAnalyzer(cfg, hook_mgr)

    imp = ana.target_importance()
    cov_imp = imp.drop(TARGET_COL).sort_values(ascending=False)
    self_attn = float(imp[TARGET_COL])

    return cov_imp, self_attn, attn_arrays


# ============================================================
# Counterfactual Analysis
# ============================================================

def run_counterfactuals(panel, covariates, predictor, prep, preds_baseline, cf_vars,
                        perturb_months=12):
    """
    구간 교란 기반 민감도 분석.

    전체 120개월이 아닌, 예측 직전 최근 perturb_months 개월만 ±1σ 교란.
    Instance normalization은 전체 문맥의 mean/std를 사용하므로,
    부분 교란은 정규화 후에도 패턴 변화로 살아남는다.
    """
    from preprocessing_util import (
        prepare_long_df_monthly_for_daily_inference,
        aggregate_daily_forecast_to_monthly,
    )
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split

    cf_variates = []
    cf_impacts = []
    cf_sigmas = []
    cf_preds_plus = []
    cf_preds_minus = []

    for cov_name in cf_vars:
        ctx_start = prep.train_end - (CTX - 1)
        ctx_series = panel.loc[ctx_start:prep.train_end, cov_name]
        sigma = float(ctx_series.std())

        # 교란 구간: 예측 직전 최근 perturb_months 개월만
        perturb_start = prep.train_end - (perturb_months - 1)
        print(f"\n  [{cov_name}] σ={sigma:.4f}, perturb: {perturb_start}~{prep.train_end} ({perturb_months}m)")

        preds_dir = []
        for sign in [+1, -1]:
            df_cf = panel[[TARGET_COL] + covariates].copy()
            df_cf.columns = ["target"] + covariates
            df_cf["item_id"] = "cpi_kr_yoy"
            # 최근 perturb_months 개월만 교란
            df_cf.loc[perturb_start:prep.train_end, cov_name] += sign * sigma

            prep_cf = prepare_long_df_monthly_for_daily_inference(
                df_cf,
                item_id_col="item_id",
                target_col="target",
                past_dynamic_real_cols=covariates,
                freq=FREQ,
                forecast_start_date=FORECAST_START_DATE,
                pdt_patches=PDT,
                ctx_patches=CTX,
                steps_per_period=PSZ,
                rolling_windows=1,
                window_distance_patches=WINDOW_DISTANCE,
            )

            ds_cf = PandasDataset.from_long_dataframe(
                prep_cf.daily_long_df,
                item_id="item_id",
                past_feat_dynamic_real=covariates,
                feat_dynamic_real=[],
            )
            _, test_cf = split(ds_cf, date=prep_cf.cutoff_period_daily)
            test_data_cf = test_cf.generate_instances(
                prediction_length=prep_cf.pdt_steps,
                windows=1,
                distance=prep_cf.dist_steps,
                max_history=prep_cf.ctx_steps,
            )

            forecasts_cf = list(predictor.predict(test_data_cf.input))
            samples_cf = np.asarray(forecasts_cf[0].samples, dtype=float)
            inp_cf = np.asarray(list(test_data_cf.input)[0]["target"], dtype=float)
            last_cf = float(inp_cf[-1]) if inp_cf.size > 0 else None

            preds_cf, _, _ = aggregate_daily_forecast_to_monthly(
                samples_cf, np.zeros(PDT * PSZ), last_cf,
                steps_per_period=PSZ, expected_periods=PDT,
            )
            preds_dir.append(preds_cf)
            label = "+1σ" if sign > 0 else "-1σ"
            print(f"    {label}  median[0]={preds_cf[0]:.4f}  (baseline={preds_baseline[0]:.4f})")

        cf_variates.append(cov_name)
        cf_sigmas.append(sigma)
        cf_preds_plus.append(preds_dir[0])
        cf_preds_minus.append(preds_dir[1])
        impact = (np.mean(np.abs(preds_dir[0] - preds_baseline)) +
                  np.mean(np.abs(preds_dir[1] - preds_baseline))) / 2.0
        cf_impacts.append(impact)
        print(f"    CF Impact = {impact:.4f} pp")

    return {
        "cf_variates":    cf_variates,
        "cf_impacts":     np.array(cf_impacts),
        "cf_sigmas":      np.array(cf_sigmas),
        "cf_preds_plus":  np.array(cf_preds_plus),
        "cf_preds_minus": np.array(cf_preds_minus),
    }


# ============================================================
# Save
# ============================================================

def save_results(output_path, variates, attn_arrays, layer_names, forecast_df,
                 panel, cf_results, stage_info=None):
    """npz 저장."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    cpi_monthly = panel[[TARGET_COL]]
    cpi_monthly.columns = ["cpi_yoy"]

    n_variates = len(variates)
    ctx_patches = attn_arrays.shape[-1] // n_variates

    save_dict = {
        "attn_arrays":    attn_arrays,
        "layer_names":    np.array(layer_names),
        "variates":       np.array(variates),
        "n_variates":     np.array(n_variates),
        "ctx_patches":    np.array(ctx_patches),
        "forecast_date":  forecast_df["date"].values,
        "forecast_med":   forecast_df["bistro_med"].values,
        "forecast_ci_lo": forecast_df["ci_lo"].values,
        "forecast_ci_hi": forecast_df["ci_hi"].values,
        "forecast_ar1":   forecast_df["ar1"].values,
        "forecast_actual":forecast_df["actual"].values,
        "history_date":   np.array([str(p) for p in cpi_monthly.index[-60:]]),
        "history_cpi":    cpi_monthly["cpi_yoy"].values[-60:],
    }

    if cf_results:
        save_dict.update({
            "cf_variates":    np.array(cf_results["cf_variates"]),
            "cf_impacts":     cf_results["cf_impacts"],
            "cf_sigmas":      cf_results["cf_sigmas"],
            "cf_preds_plus":  cf_results["cf_preds_plus"],
            "cf_preds_minus": cf_results["cf_preds_minus"],
        })

    if stage_info:
        for k, v in stage_info.items():
            save_dict[k] = v

    np.savez(output_path, **save_dict)
    print(f"\n✅ Saved: {output_path}")


# ============================================================
# Main Pipeline
# ============================================================

def main(top_k=10, use_daily=False):
    from uni2ts.model.moirai import MoiraiModule

    # ── 모델 로딩 (한 번만) ────────────────────────────────────
    print("Loading BISTRO model...")
    module = MoiraiModule.from_pretrained(MODEL_PATH)
    module.eval()
    n_layers = len(list(module.encoder.layers))
    mode_label = "DAILY" if use_daily else "MONTHLY"
    print(f"  {n_layers} transformer layers. Mode: {mode_label}\n")

    # ===========================================================
    # STAGE 1: 전체 변수 스크리닝
    # ===========================================================
    print("=" * 60)
    print(f"STAGE 1: Full screening (all covariates) [{mode_label}]")
    print("=" * 60)

    if use_daily:
        panel_full, covariates_full = load_daily_panel()
    else:
        panel_full, covariates_full = load_panel()
    n_full = 1 + len(covariates_full)
    print(f"  {n_full} variates (1 target + {len(covariates_full)} covariates)")

    inference_fn = run_bistro_inference_daily if use_daily else run_bistro_inference

    forecast_s1, preds_bl_s1, prep_s1, pred_s1, captured_s1, layers_s1 = \
        inference_fn(panel_full, covariates_full, module, n_layers)

    print(f"\n  Captured {len(layers_s1)} layers, shape: {captured_s1[layers_s1[0]].shape}")

    # Attention 랭킹
    variates_full = [TARGET_COL] + covariates_full
    cov_imp, self_attn, attn_s1 = compute_attention_ranking(
        captured_s1, layers_s1, variates_full, n_full
    )

    uniform_share = (1.0 - self_attn) / len(covariates_full)

    print(f"\n  Self-attention (CPI→CPI): {self_attn:.1%}")
    print(f"  Cross-attention budget: {1-self_attn:.1%}")
    print(f"  Uniform share (baseline): {uniform_share:.2%}")
    print(f"\n  Full attention ranking:")
    for i, (v, a) in enumerate(cov_imp.items()):
        marker = "✓" if a >= uniform_share else " "
        print(f"    {i+1:>2}. [{marker}] {v:<20s} {a:.4f} ({a:.1%})")

    # Stage 1 저장 (참고용)
    save_results(
        STAGE1_FILE, variates_full, attn_s1, layers_s1, forecast_s1,
        panel_full, cf_results=None,
        stage_info={
            "s1_ranking_vars": np.array(cov_imp.index.tolist()),
            "s1_ranking_attn": cov_imp.values,
            "s1_self_attn":    np.array(self_attn),
            "s1_uniform_share": np.array(uniform_share),
            "data_mode":       np.array(mode_label),
        },
    )

    # ── 변수 선택 ─────────────────────────────────────────────
    # 기준: uniform share 이상 OR top_k (둘 중 더 큰 집합)
    above_uniform = cov_imp[cov_imp >= uniform_share].index.tolist()
    top_k_list = cov_imp.head(top_k).index.tolist()
    selected = list(dict.fromkeys(top_k_list + above_uniform))  # 순서 유지, 중복 제거

    print(f"\n  Selection criteria:")
    print(f"    - Above uniform share ({uniform_share:.2%}): {len(above_uniform)} vars")
    print(f"    - Top-{top_k}: {len(top_k_list)} vars")
    print(f"    → Selected (union): {len(selected)} vars")
    print(f"    → {selected}")

    coverage = float(cov_imp[cov_imp.index.isin(selected)].sum())
    print(f"    → Cross-attention coverage: {coverage:.1%}")

    # ===========================================================
    # STAGE 2: 선택된 변수로 재추론
    # ===========================================================
    print("\n" + "=" * 60)
    print(f"STAGE 2: Re-inference with {len(selected)} selected covariates [{mode_label}]")
    print("=" * 60)

    if use_daily:
        panel_s2, covariates_s2 = load_daily_panel(covariate_subset=selected)
    else:
        panel_s2, covariates_s2 = load_panel(covariate_subset=selected)
    n_s2 = 1 + len(covariates_s2)
    print(f"  {n_s2} variates (1 target + {len(covariates_s2)} covariates)")

    forecast_s2, preds_bl_s2, prep_s2, pred_s2, captured_s2, layers_s2 = \
        inference_fn(panel_s2, covariates_s2, module, n_layers)

    print(f"\n  Captured {len(layers_s2)} layers, shape: {captured_s2[layers_s2[0]].shape}")
    print(f"\n  Stage 2 forecast:")
    print(forecast_s2.to_string(index=False))

    # Stage 2 Attention 분석
    variates_s2 = [TARGET_COL] + covariates_s2
    cov_imp_s2, self_attn_s2, attn_s2 = compute_attention_ranking(
        captured_s2, layers_s2, variates_s2, n_s2
    )

    print(f"\n  Stage 2 self-attention: {self_attn_s2:.1%}")
    print(f"  Stage 2 attention ranking:")
    for i, (v, a) in enumerate(cov_imp_s2.items()):
        print(f"    {i+1:>2}. {v:<20s} {a:.4f} ({a:.1%})")

    # ── Stage 2 CF (월별 모드만 — daily에서는 Ablation으로 대체) ──
    cf_results = None
    if not use_daily:
        print("\n" + "-" * 40)
        print(f"Stage 2 CF Analysis ({len(covariates_s2)} covariates)")
        print("-" * 40)
        cf_results = run_counterfactuals(
            panel_s2, covariates_s2, pred_s2, prep_s2, preds_bl_s2,
            cf_vars=covariates_s2,
        )
    else:
        print("\n  CF skipped in daily mode (deprecated — use Ablation instead)")

    # Stage 2 결과 저장 (대시보드용)
    save_results(
        OUTPUT_FILE, variates_s2, attn_s2, layers_s2, forecast_s2,
        panel_s2, cf_results,
        stage_info={
            # Stage 1 스크리닝 정보도 포함 (대시보드에서 참조)
            "s1_all_vars":       np.array(cov_imp.index.tolist()),
            "s1_all_attn":       cov_imp.values,
            "s1_self_attn":      np.array(self_attn),
            "s1_n_total":        np.array(len(covariates_full)),
            "s2_selected_vars":  np.array(selected),
            "data_mode":         np.array(mode_label),
        },
    )

    # ── 요약 ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE [{mode_label}]")
    print("=" * 60)
    print(f"  Stage 1: {n_full} vars → attention ranking → {len(selected)} selected")
    print(f"  Stage 2: {n_s2} vars → new attention + CF for all {len(covariates_s2)} covariates")
    print(f"  Output:  {OUTPUT_FILE}")
    print(f"  Stage1:  {STAGE1_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=10,
                        help="Stage 1에서 최소 선택할 상위 변수 수 (default: 10)")
    parser.add_argument("--daily", action="store_true",
                        help="일별 패널 데이터 사용 (macro_panel_daily.csv)")
    args = parser.parse_args()
    main(top_k=args.top_k, use_daily=args.daily)
