"""
BISTRO Real Inference Runner
=============================
실제 BISTRO 모델로 추론을 실행하고, Attention 가중치를 캡처해
data/real_inference_results.npz 에 저장한다.

실행:
    .venv-bistro/bin/python3 bistro_runner.py

의존성: .venv-bistro (Python 3.11, uni2ts, torch, gluonts 설치됨)
"""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Repo 경로 설정 ────────────────────────────────────────────
BISTRO_REPO   = "/tmp/bistro-repo"
MODEL_PATH    = f"{BISTRO_REPO}/bistro-finetuned"
DATA_DIR      = f"{BISTRO_REPO}/data"
OUTPUT_FILE   = os.path.join(os.path.dirname(__file__), "data", "real_inference_results.npz")

sys.path.insert(0, f"{BISTRO_REPO}/src")

# ── 추론 설정 ─────────────────────────────────────────────────
FREQ      = "M"
CTX       = 120   # context 개월 수 (각 변수당 120 patches)
PDT       = 12    # 예측 개월 수
PSZ       = 32    # steps per period (daily)
BSZ       = 32
WINDOW_DISTANCE = 12
FORECAST_START_DATE = "2023-01-01"


# ============================================================
# 1. 데이터 로딩 & 전처리
# ============================================================

def load_data():
    print("Loading data...")

    # Korean CPI YoY monthly
    cpi = pd.read_csv(f"{DATA_DIR}/bis_cpi_kr_yoy_m.csv", index_col=0)
    cpi.index = pd.to_datetime(cpi.index).to_period(FREQ)
    cpi.columns = ["cpi_yoy"]

    # Korean policy rate monthly
    rate = pd.read_csv(f"{DATA_DIR}/bis_cbpol_kr_m.csv", index_col=0)
    rate.index = pd.to_datetime(rate.index).to_period(FREQ)
    rate.columns = ["rate"]

    # Oil price (WTI daily → monthly mean)
    oil_daily = pd.read_csv(f"{DATA_DIR}/fred_oil_price_d.csv", index_col=0)
    oil_daily.index = pd.to_datetime(oil_daily.index)
    oil_daily.columns = ["oil"]
    oil = oil_daily["oil"].resample("M").mean()  # pandas 2.1 uses "M" not "ME"
    oil.index = oil.index.to_period(FREQ)
    oil = oil.to_frame("oil")

    # 공통 기간으로 정렬
    common = cpi.index.intersection(rate.index).intersection(oil.index)
    cpi   = cpi.loc[common]
    rate  = rate.loc[common]
    oil   = oil.loc[common]

    print(f"  CPI:  {cpi.index[0]} ~ {cpi.index[-1]}  ({len(cpi)} months)")
    print(f"  Rate: {rate.index[0]} ~ {rate.index[-1]}  ({len(rate)} months)")
    print(f"  Oil:  {oil.index[0]} ~ {oil.index[-1]}  ({len(oil)} months)")

    # 결합 DataFrame (long format)
    df = cpi.copy()
    df["item_id"] = "cpi_kr_yoy"
    df = df.merge(rate, left_index=True, right_index=True, how="inner")
    df = df.merge(oil,  left_index=True, right_index=True, how="inner")
    df.columns = ["target", "item_id", "rate", "oil"]

    return df, cpi


# ============================================================
# 2. Attention 캡처용 SDPA Monkey-patch
# ============================================================

def install_attention_hooks(module, n_layers=12):
    """
    F.scaled_dot_product_attention 를 monkey-patch 하고
    각 레이어의 self_attn 에 pre-hook 을 달아 layer_idx 를 추적한다.
    """
    import torch
    import torch.nn.functional as F

    captured      = {}      # "encoder.layers.{i}.self_attn" → np.ndarray (q, k)
    current_layer = [-1]    # pre-hook 이 업데이트
    hooks         = []

    # 각 레이어 pre-hook: SDPA 호출 전 layer_idx 설정
    for i in range(n_layers):
        def make_pre(idx):
            def pre_hook(mod, args):
                current_layer[0] = idx
            return pre_hook
        h = module.encoder.layers[i].self_attn.register_forward_pre_hook(make_pre(i))
        hooks.append(h)

    _orig_sdpa = F.scaled_dot_product_attention

    def _hooked_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, scale=None,
                     is_causal=False, **kw):
        if current_layer[0] >= 0:
            sc = scale if scale is not None else 1.0 / math.sqrt(query.size(-1))
            with torch.no_grad():
                # query/key: [batch, groups, hpg, q_len, head_dim]
                raw = torch.matmul(query.float(), key.float().transpose(-2, -1)) * sc
                if attn_mask is not None:
                    if attn_mask.dtype == torch.bool:
                        raw = raw.masked_fill(~attn_mask, float("-inf"))
                    else:
                        raw = raw + attn_mask.float()
                w = torch.softmax(raw, dim=-1)
                # head-average: [q_len, kv_len]
                avg = w.mean(dim=tuple(range(w.dim() - 2)))
                layer_name = f"encoder.layers.{current_layer[0]}.self_attn"
                captured[layer_name] = avg.cpu().numpy()

        return _orig_sdpa(query, key, value,
                          attn_mask=attn_mask, dropout_p=dropout_p,
                          scale=scale, is_causal=is_causal, **kw)

    # monkey-patch: uni2ts.module.attention 내부의 F 를 교체
    import uni2ts.module.attention as _attn_mod
    import torch.nn.functional as _F_mod
    _F_mod.scaled_dot_product_attention = _hooked_sdpa
    # torch.nn.functional 자체도 교체 (두 모듈이 같은 객체인 경우가 많지만 안전하게)
    torch.nn.functional.scaled_dot_product_attention = _hooked_sdpa

    def restore():
        for h in hooks:
            h.remove()
        torch.nn.functional.scaled_dot_product_attention = _orig_sdpa

    return captured, restore


# ============================================================
# 3. 추론 실행
# ============================================================

def run_inference():
    print("=" * 60)
    print("BISTRO Real Inference + Attention Hook")
    print("=" * 60)

    # 데이터 준비
    df, cpi_monthly = load_data()

    # GluonTS 전처리
    sys.path.insert(0, f"{BISTRO_REPO}/src")
    from preprocessing_util import prepare_long_df_monthly_for_daily_inference
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    prep = prepare_long_df_monthly_for_daily_inference(
        df,
        item_id_col="item_id",
        target_col="target",
        past_dynamic_real_cols=["rate", "oil"],
        freq=FREQ,
        forecast_start_date=FORECAST_START_DATE,
        pdt_patches=PDT,
        ctx_patches=CTX,
        steps_per_period=PSZ,
        rolling_windows=1,
        window_distance_patches=WINDOW_DISTANCE,
    )
    print(f"\nForecast start: {prep.forecast_start}, train end: {prep.train_end}")
    print(f"ctx_steps={prep.ctx_steps}, pdt_steps={prep.pdt_steps}")

    # GluonTS Dataset
    ds = PandasDataset.from_long_dataframe(
        prep.daily_long_df,
        item_id="item_id",
        past_feat_dynamic_real=["rate", "oil"],
        feat_dynamic_real=[],
    )
    train, test_template = split(ds, date=prep.cutoff_period_daily)
    test_data = test_template.generate_instances(
        prediction_length=prep.pdt_steps,
        windows=1,
        distance=prep.dist_steps,
        max_history=prep.ctx_steps,
    )

    # 모델 로딩
    print("\nLoading BISTRO model...")
    module = MoiraiModule.from_pretrained(MODEL_PATH)
    module.eval()
    n_layers = len(list(module.encoder.layers))
    print(f"Model loaded. {n_layers} transformer layers.")

    # Attention Hook 설치
    print("Installing attention hooks...")
    captured, restore_hooks = install_attention_hooks(module, n_layers)

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

    print("\nRunning inference (hooks active)...")
    inputs    = list(test_data.input)
    labels    = list(test_data.label)
    forecasts = list(predictor.predict(test_data.input))
    restore_hooks()
    print(f"  Captured {len(captured)} attention layers.")

    # ── 예측 결과 집계 ─────────────────────────────────────────
    from preprocessing_util import aggregate_daily_forecast_to_monthly

    # baseline 예측 집계 (CF 비교용으로 미리 보관)
    _samples_bl  = np.asarray(forecasts[0].samples, dtype=float)
    _inp_bl      = np.asarray(inputs[0]["target"], dtype=float)
    _last_bl     = float(_inp_bl[-1]) if _inp_bl.size > 0 else None
    preds_baseline, _, _ = aggregate_daily_forecast_to_monthly(
        _samples_bl, np.zeros(PDT * PSZ), _last_bl,
        steps_per_period=PSZ, expected_periods=PDT,
    )

    samples = np.asarray(forecasts[0].samples, dtype=float)
    label_target = np.asarray(labels[0]["target"], dtype=float)
    inp_target   = np.asarray(inputs[0]["target"], dtype=float)
    last_input   = float(inp_target[-1]) if inp_target.size > 0 else None

    preds, gts, ci = aggregate_daily_forecast_to_monthly(
        samples, label_target, last_input,
        steps_per_period=PSZ, expected_periods=PDT,
    )

    pred_index = pd.period_range(
        start=prep.forecast_start, periods=PDT, freq=FREQ
    )

    forecast_df = pd.DataFrame({
        "date":       [str(p) for p in pred_index],
        "bistro_med": preds,
        "ci_lo":      ci[:, 0],
        "ci_hi":      ci[:, 1],
    })

    # AR(1) 기준선
    try:
        from inference_util import ar1_forecast
        train_y = cpi_monthly["cpi_yoy"].loc[:pred_index[0] - 1].tail(CTX).astype(float)
        ar1 = ar1_forecast(train_y, pred_index, method="ols", trend="c", validate_index=True)
        forecast_df["ar1"] = ar1.values
    except Exception as e:
        print(f"  AR(1) skipped: {e}")
        forecast_df["ar1"] = np.nan

    # 실제값
    actual_reindexed = cpi_monthly["cpi_yoy"].reindex(pred_index)
    forecast_df["actual"] = actual_reindexed.values

    print("\nForecast result (monthly median):")
    print(forecast_df.to_string(index=False))

    # ── Attention 처리 ─────────────────────────────────────────
    # captured: 레이어별 (q_len, kv_len) 어텐션 행렬
    # n_variates = 3 (CPI + rate + oil), ctx_patches = CTX = 120
    n_variates  = 3
    ctx_patches = CTX
    variates    = ["CPI_KR_YoY", "Rate_KR", "Oil_WTI"]

    layer_names = sorted(captured.keys(), key=lambda x: int(x.split(".")[2]))
    attn_shapes = {k: v.shape for k, v in captured.items()}
    print("\nAttention shapes per layer:", set(attn_shapes.values()))

    # 저장할 attention arrays
    attn_arrays = np.stack([captured[k] for k in layer_names], axis=0)  # (n_layers, q, k)

    # ── Counterfactual Analysis ────────────────────────────────
    print("\n" + "=" * 60)
    print("Counterfactual Analysis (±1σ per covariate)")
    print("=" * 60)
    cf_results = run_counterfactuals(
        df, predictor, prep, preds_baseline,
        n_variates=n_variates, ctx_patches=CTX,
    )

    # ── 저장 ───────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    np.savez(
        OUTPUT_FILE,
        # attention
        attn_arrays   = attn_arrays,         # (n_layers, q_len, kv_len)
        layer_names   = np.array(layer_names),
        variates      = np.array(variates),
        n_variates    = np.array(n_variates),
        ctx_patches   = np.array(ctx_patches),
        # forecast
        forecast_date    = forecast_df["date"].values,
        forecast_med     = forecast_df["bistro_med"].values,
        forecast_ci_lo   = forecast_df["ci_lo"].values,
        forecast_ci_hi   = forecast_df["ci_hi"].values,
        forecast_ar1     = forecast_df["ar1"].values,
        forecast_actual  = forecast_df["actual"].values,
        # history (recent 60 months)
        history_date  = np.array([str(p) for p in cpi_monthly.index[-60:]]),
        history_cpi   = cpi_monthly["cpi_yoy"].values[-60:],
        # counterfactual
        cf_variates      = np.array(cf_results["cf_variates"]),
        cf_impacts       = cf_results["cf_impacts"],
        cf_sigmas        = cf_results["cf_sigmas"],
        cf_preds_plus    = cf_results["cf_preds_plus"],
        cf_preds_minus   = cf_results["cf_preds_minus"],
    )
    print(f"\n✅ Saved: {OUTPUT_FILE}")
    return OUTPUT_FILE


# ============================================================
# 4. Counterfactual Analysis
# ============================================================

COVARIATE_MAP = {
    "rate": "Rate_KR",
    "oil":  "Oil_WTI",
}


def run_counterfactuals(df, predictor, prep_baseline, preds_baseline, n_variates=3, ctx_patches=120):
    """
    Context sensitivity analysis: 과거 문맥(past_feat_dynamic_real)의
    각 공변량 수준을 ±1σ 이동한 후 재추론하여, 모델 예측이 과거 공변량
    수준 변화에 얼마나 민감한지 측정한다.

    주의: BISTRO/MOIRAI에서 공변량은 past_feat_dynamic_real로 투입되므로
    이 분석은 "미래 유가가 오르면?" 질문이 아니라,
    "과거 유가가 달랐더라면 모델이 어떻게 예측했을까?"에 대한 답이다.

    Returns
    -------
    dict with keys:
        cf_variates   : list[str]  — 공변량 이름 (variate 기준)
        cf_impacts    : np.ndarray (n_cov,)  — 평균 |Δforecast| in pp
        cf_sigmas     : np.ndarray (n_cov,)  — 사용된 σ 값
        cf_preds_plus : np.ndarray (n_cov, PDT)
        cf_preds_minus: np.ndarray (n_cov, PDT)
    """
    from preprocessing_util import (
        prepare_long_df_monthly_for_daily_inference,
        aggregate_daily_forecast_to_monthly,
    )
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split

    covar_cols = list(COVARIATE_MAP.keys())   # ["rate", "oil"]
    n_cov      = len(covar_cols)

    ctx_start  = prep_baseline.train_end - (ctx_patches - 1)

    cf_variates    = [COVARIATE_MAP[c] for c in covar_cols]
    cf_sigmas      = np.zeros(n_cov)
    cf_impacts     = np.zeros(n_cov)
    cf_preds_plus  = np.zeros((n_cov, PDT))
    cf_preds_minus = np.zeros((n_cov, PDT))

    for ci, covar_col in enumerate(covar_cols):
        ctx_series = df.loc[ctx_start:prep_baseline.train_end, covar_col]
        sigma      = float(ctx_series.std())
        cf_sigmas[ci] = sigma
        print(f"\n  [{covar_col}]  σ = {sigma:.4f}")

        preds_dir = []
        for sign in [+1, -1]:
            df_cf = df.copy()
            df_cf.loc[ctx_start:prep_baseline.train_end, covar_col] += sign * sigma

            prep_cf = prepare_long_df_monthly_for_daily_inference(
                df_cf,
                item_id_col="item_id",
                target_col="target",
                past_dynamic_real_cols=["rate", "oil"],
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
                past_feat_dynamic_real=["rate", "oil"],
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
            samples_cf   = np.asarray(forecasts_cf[0].samples, dtype=float)
            inp_cf       = np.asarray(list(test_data_cf.input)[0]["target"], dtype=float)
            last_cf      = float(inp_cf[-1]) if inp_cf.size > 0 else None

            preds_cf, _, _ = aggregate_daily_forecast_to_monthly(
                samples_cf, np.zeros(PDT * PSZ), last_cf,
                steps_per_period=PSZ, expected_periods=PDT,
            )
            preds_dir.append(preds_cf)
            label = f"+{sign}σ" if sign > 0 else f"-1σ"
            print(f"    {label}  median[0]={preds_cf[0]:.4f}  (baseline={preds_baseline[0]:.4f})")

        cf_preds_plus[ci]  = preds_dir[0]   # +1σ
        cf_preds_minus[ci] = preds_dir[1]   # -1σ

        impact_plus  = np.mean(np.abs(preds_dir[0] - preds_baseline))
        impact_minus = np.mean(np.abs(preds_dir[1] - preds_baseline))
        cf_impacts[ci] = (impact_plus + impact_minus) / 2.0
        print(f"    CF Impact = {cf_impacts[ci]:.4f} pp")

    return {
        "cf_variates":    cf_variates,
        "cf_impacts":     cf_impacts,
        "cf_sigmas":      cf_sigmas,
        "cf_preds_plus":  cf_preds_plus,
        "cf_preds_minus": cf_preds_minus,
    }


if __name__ == "__main__":
    run_inference()
