"""
CF만 재실행: 구간 교란 (최근 12개월만 ±1σ)
기존 npz의 attention/forecast는 유지하고 CF 결과만 교체.

실행:
    .venv-bistro/bin/python3 rerun_cf.py
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
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "data", "real_inference_results.npz")

sys.path.insert(0, f"{BISTRO_REPO}/src")

FREQ = "M"
CTX  = 120
PDT  = 12
PSZ  = 32
BSZ  = 32
WINDOW_DISTANCE = 12
FORECAST_START_DATE = "2023-01-01"
TARGET_COL = "CPI_KR_YoY"
PERTURB_MONTHS = 12


def main():
    from preprocessing_util import (
        prepare_long_df_monthly_for_daily_inference,
        aggregate_daily_forecast_to_monthly,
    )
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

    # 기존 npz 로딩
    data = np.load(OUTPUT_FILE, allow_pickle=True)
    variates = [str(v) for v in data["variates"]]
    covariates = [v for v in variates if v != TARGET_COL]
    print(f"Covariates: {covariates}")

    # 패널 로딩
    panel = pd.read_csv(PANEL_CSV, index_col=0)
    panel.index = pd.PeriodIndex(panel.index, freq=FREQ)
    panel = panel[[TARGET_COL] + covariates].ffill().bfill()

    # 모델 로딩
    print("\nLoading BISTRO model...")
    module = MoiraiModule.from_pretrained(MODEL_PATH)
    module.eval()

    # Baseline predictor 구성 (attention hook 불필요)
    df_base = panel[[TARGET_COL] + covariates].copy()
    df_base.columns = ["target"] + covariates
    df_base["item_id"] = "cpi_kr_yoy"

    prep = prepare_long_df_monthly_for_daily_inference(
        df_base,
        item_id_col="item_id", target_col="target",
        past_dynamic_real_cols=covariates,
        freq=FREQ, forecast_start_date=FORECAST_START_DATE,
        pdt_patches=PDT, ctx_patches=CTX, steps_per_period=PSZ,
        rolling_windows=1, window_distance_patches=WINDOW_DISTANCE,
    )

    ds_base = PandasDataset.from_long_dataframe(
        prep.daily_long_df, item_id="item_id",
        past_feat_dynamic_real=covariates, feat_dynamic_real=[],
    )

    model = MoiraiForecast(
        module=module,
        prediction_length=prep.pdt_steps,
        context_length=prep.ctx_steps,
        patch_size=PSZ, num_samples=100, target_dim=1,
        feat_dynamic_real_dim=ds_base.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds_base.num_past_feat_dynamic_real,
    )
    predictor = model.create_predictor(batch_size=BSZ)

    # Baseline 예측
    _, test_template = split(ds_base, date=prep.cutoff_period_daily)
    test_data = test_template.generate_instances(
        prediction_length=prep.pdt_steps, windows=1,
        distance=prep.dist_steps, max_history=prep.ctx_steps,
    )
    forecasts_bl = list(predictor.predict(test_data.input))
    samples_bl = np.asarray(forecasts_bl[0].samples, dtype=float)
    inp_bl = np.asarray(list(test_data.input)[0]["target"], dtype=float)
    last_bl = float(inp_bl[-1]) if inp_bl.size > 0 else None
    preds_baseline, _, _ = aggregate_daily_forecast_to_monthly(
        samples_bl, np.zeros(PDT * PSZ), last_bl,
        steps_per_period=PSZ, expected_periods=PDT,
    )
    print(f"\nBaseline median[0] = {preds_baseline[0]:.4f}")

    # ── 구간 교란 CF ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Partial Perturbation CF (last {PERTURB_MONTHS} months only, ±1σ)")
    print(f"{'='*60}")

    perturb_start = prep.train_end - (PERTURB_MONTHS - 1)
    ctx_start = prep.train_end - (CTX - 1)

    cf_variates = []
    cf_impacts = []
    cf_sigmas = []
    cf_preds_plus = []
    cf_preds_minus = []

    for cov_name in covariates:
        ctx_series = panel.loc[ctx_start:prep.train_end, cov_name]
        sigma = float(ctx_series.std())
        print(f"\n  [{cov_name}] σ={sigma:.4f}, perturb: {perturb_start}~{prep.train_end}")

        preds_dir = []
        for sign in [+1, -1]:
            df_cf = panel[[TARGET_COL] + covariates].copy()
            df_cf.columns = ["target"] + covariates
            df_cf["item_id"] = "cpi_kr_yoy"
            df_cf.loc[perturb_start:prep.train_end, cov_name] += sign * sigma

            prep_cf = prepare_long_df_monthly_for_daily_inference(
                df_cf,
                item_id_col="item_id", target_col="target",
                past_dynamic_real_cols=covariates,
                freq=FREQ, forecast_start_date=FORECAST_START_DATE,
                pdt_patches=PDT, ctx_patches=CTX, steps_per_period=PSZ,
                rolling_windows=1, window_distance_patches=WINDOW_DISTANCE,
            )

            ds_cf = PandasDataset.from_long_dataframe(
                prep_cf.daily_long_df, item_id="item_id",
                past_feat_dynamic_real=covariates, feat_dynamic_real=[],
            )
            _, test_cf = split(ds_cf, date=prep_cf.cutoff_period_daily)
            test_data_cf = test_cf.generate_instances(
                prediction_length=prep_cf.pdt_steps, windows=1,
                distance=prep_cf.dist_steps, max_history=prep_cf.ctx_steps,
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
            print(f"    {label}  med[0]={preds_cf[0]:.4f}  (bl={preds_baseline[0]:.4f}  Δ={preds_cf[0]-preds_baseline[0]:+.4f})")

        cf_variates.append(cov_name)
        cf_sigmas.append(sigma)
        cf_preds_plus.append(preds_dir[0])
        cf_preds_minus.append(preds_dir[1])
        impact = (np.mean(np.abs(preds_dir[0] - preds_baseline)) +
                  np.mean(np.abs(preds_dir[1] - preds_baseline))) / 2.0
        cf_impacts.append(impact)
        print(f"    Impact = {impact:.4f} pp")

    # ── 기존 npz에 CF 결과만 교체 저장 ────────────────────────
    save_dict = {k: data[k] for k in data.files}
    save_dict.update({
        "cf_variates":    np.array(cf_variates),
        "cf_impacts":     np.array(cf_impacts),
        "cf_sigmas":      np.array(cf_sigmas),
        "cf_preds_plus":  np.array(cf_preds_plus),
        "cf_preds_minus": np.array(cf_preds_minus),
        "cf_perturb_months": np.array(PERTURB_MONTHS),
    })
    np.savez(OUTPUT_FILE, **save_dict)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    ranked = sorted(zip(cf_variates, cf_impacts), key=lambda x: -x[1])
    for i, (v, imp) in enumerate(ranked):
        print(f"  {i+1:>2}. {v:<20s}  impact={imp:.4f} pp")
    print(f"\n✅ Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
