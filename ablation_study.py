"""
BISTRO Ablation + Incremental Addition Study
=============================================
1. Ablation: 변수를 하나씩 제거하고 재추론 → ΔRMSE
2. Incremental: attention 순서대로 변수를 추가 → RMSE 커브

결과: data/ablation_results.npz

실행:
    .venv-bistro/bin/python3 ablation_study.py
"""

import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

BISTRO_REPO = "/tmp/bistro-repo"
MODEL_PATH  = f"{BISTRO_REPO}/bistro-finetuned"
DATA_DIR    = f"{BISTRO_REPO}/data"
PANEL_CSV   = os.path.join(DATA_DIR, "macro_panel.csv")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "ablation_results.npz")
REAL_NPZ    = os.path.join(os.path.dirname(__file__), "data", "real_inference_results.npz")

sys.path.insert(0, f"{BISTRO_REPO}/src")

FREQ = "M"
CTX  = 120
PDT  = 12
PSZ  = 32
BSZ  = 32
WINDOW_DISTANCE = 12
FORECAST_START_DATE = "2023-01-01"
TARGET_COL = "CPI_KR_YoY"


def forecast_with_covariates(panel, covariates, module, n_layers):
    """
    주어진 공변량 리스트로 BISTRO 추론, monthly median 예측 반환.
    covariates가 빈 리스트면 target-only (univariate) 추론.
    """
    from preprocessing_util import (
        prepare_long_df_monthly_for_daily_inference,
        aggregate_daily_forecast_to_monthly,
    )
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast

    df = panel[[TARGET_COL] + covariates].copy()
    df.columns = ["target"] + covariates
    df["item_id"] = "cpi_kr_yoy"

    prep = prepare_long_df_monthly_for_daily_inference(
        df,
        item_id_col="item_id", target_col="target",
        past_dynamic_real_cols=covariates if covariates else [],
        freq=FREQ, forecast_start_date=FORECAST_START_DATE,
        pdt_patches=PDT, ctx_patches=CTX, steps_per_period=PSZ,
        rolling_windows=1, window_distance_patches=WINDOW_DISTANCE,
    )

    if covariates:
        ds = PandasDataset.from_long_dataframe(
            prep.daily_long_df, item_id="item_id",
            past_feat_dynamic_real=covariates, feat_dynamic_real=[],
        )
    else:
        ds = PandasDataset.from_long_dataframe(
            prep.daily_long_df, item_id="item_id",
        )

    _, test_template = split(ds, date=prep.cutoff_period_daily)
    test_data = test_template.generate_instances(
        prediction_length=prep.pdt_steps, windows=1,
        distance=prep.dist_steps, max_history=prep.ctx_steps,
    )

    model = MoiraiForecast(
        module=module,
        prediction_length=prep.pdt_steps,
        context_length=prep.ctx_steps,
        patch_size=PSZ, num_samples=100, target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    predictor = model.create_predictor(batch_size=BSZ)

    labels = list(test_data.label)
    forecasts = list(predictor.predict(test_data.input))

    samples = np.asarray(forecasts[0].samples, dtype=float)
    label_target = np.asarray(labels[0]["target"], dtype=float)
    inputs = list(test_data.input)
    inp_target = np.asarray(inputs[0]["target"], dtype=float)
    last_input = float(inp_target[-1]) if inp_target.size > 0 else None

    preds, gts, ci = aggregate_daily_forecast_to_monthly(
        samples, label_target, last_input,
        steps_per_period=PSZ, expected_periods=PDT,
    )

    pred_index = pd.period_range(start=prep.forecast_start, periods=PDT, freq=FREQ)
    cpi_monthly = panel[[TARGET_COL]]
    cpi_monthly.columns = ["cpi_yoy"]
    actual = cpi_monthly["cpi_yoy"].reindex(pred_index).values

    return preds, actual, ci


def compute_rmse(preds, actual):
    valid = ~np.isnan(actual)
    if not valid.any():
        return np.nan
    return float(np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2)))


def compute_mae(preds, actual):
    valid = ~np.isnan(actual)
    if not valid.any():
        return np.nan
    return float(np.mean(np.abs(preds[valid] - actual[valid])))


def main():
    from uni2ts.model.moirai import MoiraiModule

    # 패널 로딩
    panel = pd.read_csv(PANEL_CSV, index_col=0)
    panel.index = pd.PeriodIndex(panel.index, freq=FREQ)
    panel = panel.ffill().bfill()

    # Stage 2 변수 확인
    real_data = np.load(REAL_NPZ, allow_pickle=True)
    all_variates = [str(v) for v in real_data["variates"]]
    covariates = [v for v in all_variates if v != TARGET_COL]
    print(f"Stage 2 covariates ({len(covariates)}): {covariates}")

    # Attention 랭킹 (Stage 2 기준)
    from bistro_core import BISTROConfig, AttentionHookManager, AttentionAnalyzer
    attn_arrays = real_data["attn_arrays"]
    layer_names = [str(s) for s in real_data["layer_names"]]
    n_var = int(real_data["n_variates"])
    ctx_p = int(real_data["ctx_patches"])

    cfg_temp = BISTROConfig(variates=all_variates, target_idx=0,
                            ctx_patches=ctx_p, pdt_patches=PDT, patch_size=PSZ)
    hook_temp = AttentionHookManager()
    for i, name in enumerate(layer_names):
        hook_temp.attention_maps[name] = attn_arrays[i][np.newaxis, np.newaxis, ...]
    ana_temp = AttentionAnalyzer(cfg_temp, hook_temp)
    imp = ana_temp.target_importance()
    attn_ranking = imp.drop(TARGET_COL).sort_values(ascending=False).index.tolist()
    print(f"Attention ranking: {attn_ranking}")

    # 모델 로딩
    print("\nLoading BISTRO model...")
    module = MoiraiModule.from_pretrained(MODEL_PATH)
    module.eval()
    n_layers = len(list(module.encoder.layers))
    print(f"  {n_layers} layers\n")

    # ============================================================
    # 0. Baseline: 전체 10개 공변량
    # ============================================================
    print("=" * 60)
    print("BASELINE: all 10 covariates")
    print("=" * 60)
    preds_full, actual, _ = forecast_with_covariates(panel, covariates, module, n_layers)
    rmse_full = compute_rmse(preds_full, actual)
    mae_full = compute_mae(preds_full, actual)
    print(f"  RMSE = {rmse_full:.4f} pp,  MAE = {mae_full:.4f} pp\n")

    # ============================================================
    # 1. Ablation: 변수 하나씩 제거
    # ============================================================
    print("=" * 60)
    print("ABLATION: remove one covariate at a time")
    print("=" * 60)

    abl_vars = []
    abl_rmse = []
    abl_mae = []
    abl_delta_rmse = []

    for cov in attn_ranking:
        remaining = [c for c in covariates if c != cov]
        print(f"\n  Remove [{cov}] → {len(remaining)} covariates...", end=" ")
        preds, _, _ = forecast_with_covariates(panel, remaining, module, n_layers)
        rmse = compute_rmse(preds, actual)
        mae = compute_mae(preds, actual)
        delta = rmse - rmse_full
        print(f"RMSE={rmse:.4f} (Δ={delta:+.4f}),  MAE={mae:.4f}")

        abl_vars.append(cov)
        abl_rmse.append(rmse)
        abl_mae.append(mae)
        abl_delta_rmse.append(delta)

    print(f"\n  Ablation ranking (by ΔRMSE, higher = more important):")
    abl_sorted = sorted(zip(abl_vars, abl_delta_rmse), key=lambda x: -x[1])
    for i, (v, d) in enumerate(abl_sorted):
        marker = "↑" if d > 0 else "↓"
        print(f"    {i+1:>2}. {v:<20s}  ΔRMSE={d:+.4f} {marker}")

    # ============================================================
    # 2. Incremental: CPI-only → +top1 → +top2 → ... → all
    # ============================================================
    print(f"\n{'=' * 60}")
    print("INCREMENTAL: add covariates in attention order")
    print("=" * 60)

    inc_labels = []
    inc_n_vars = []
    inc_rmse = []
    inc_mae = []

    # BISTRO는 최소 1개 공변량이 필요하므로 1개부터 시작
    for k in range(1, len(attn_ranking) + 1):
        subset = attn_ranking[:k]
        label = f"+{attn_ranking[k-1]}"
        print(f"\n  [{k} covariates] {', '.join(subset)}...", end=" ")
        preds_k, _, _ = forecast_with_covariates(panel, subset, module, n_layers)
        rmse_k = compute_rmse(preds_k, actual)
        mae_k = compute_mae(preds_k, actual)
        print(f"RMSE={rmse_k:.4f},  MAE={mae_k:.4f}")

        inc_labels.append(label)
        inc_n_vars.append(1 + k)
        inc_rmse.append(rmse_k)
        inc_mae.append(mae_k)

    # ============================================================
    # 저장
    # ============================================================
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    np.savez(
        OUTPUT_FILE,
        # baseline
        baseline_rmse=np.array(rmse_full),
        baseline_mae=np.array(mae_full),
        baseline_covariates=np.array(covariates),
        actual=actual,
        preds_full=preds_full,
        # attention ranking
        attn_ranking=np.array(attn_ranking),
        attn_values=np.array([float(imp[v]) for v in attn_ranking]),
        # ablation
        abl_vars=np.array(abl_vars),
        abl_rmse=np.array(abl_rmse),
        abl_mae=np.array(abl_mae),
        abl_delta_rmse=np.array(abl_delta_rmse),
        # incremental
        inc_labels=np.array(inc_labels),
        inc_n_vars=np.array(inc_n_vars),
        inc_rmse=np.array(inc_rmse),
        inc_mae=np.array(inc_mae),
    )

    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Full (10 cov) RMSE: {rmse_full:.4f}")
    print(f"  1-var RMSE:         {inc_rmse[0]:.4f} ({inc_labels[0]})")
    print(f"  Best incremental:   {min(inc_rmse):.4f} ({inc_labels[inc_rmse.index(min(inc_rmse))]})")
    print(f"\n✅ Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
