"""
CTX=36 (3년) vs CTX=120 (10년) 비교.
최적 18변수로 2023, 2024 각각 추론.
"""
import sys, os
import numpy as np
import pandas as pd

BISTRO_REPO = "/tmp/bistro-repo"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

import bistro_runner_30var as brm

# Tournament panel (288 vars including optimal 18)
TOURNAMENT_PANEL = os.path.join(os.path.dirname(__file__), "data", "macro_panel_tournament_daily.csv")
brm.PANEL_DAILY_CSV = TOURNAMENT_PANEL

from bistro_runner_30var import (
    load_daily_panel, run_bistro_inference_daily,
    TARGET_COL, PDT, PSZ, MODEL_PATH,
)

OPTIMAL_18 = [
    "AUD_USD", "CN_Interbank3M", "US_UnempRate", "JP_REER",
    "JP_Interbank3M", "JP_CoreCPI", "KC_FSI", "KR_MfgProd",
    "Pork", "US_NFP", "US_TradeTransEmp", "THB_USD",
    "PPI_CopperNickel", "CN_PPI", "US_Mortgage15Y", "UK_10Y_Bond",
    "US_ExportPI", "US_DepInstCredit",
]

# Load model once
from uni2ts.model.moirai import MoiraiModule
print("Loading BISTRO model...")
module = MoiraiModule.from_pretrained(MODEL_PATH)
module.eval()
n_layers = len(list(module.encoder.layers))
print(f"  {n_layers} layers loaded.\n")

panel, covariates = load_daily_panel(covariate_subset=OPTIMAL_18)
n_var = 1 + len(covariates)

results = {}

for year, fstart in [("2023", "2023-01-01"), ("2024", "2024-01-01")]:
    for ctx in [36, 120]:
        label = f"{year}_CTX{ctx}"
        print(f"{'='*60}")
        print(f"  {label}: forecast_start={fstart}, CTX={ctx}")
        print(f"{'='*60}")

        brm.FORECAST_START_DATE = fstart
        brm.CTX = ctx

        # Re-import to pick up changed CTX
        forecast_df, _, _, _, _, _ = run_bistro_inference_daily(
            panel, covariates, module, n_layers, capture_attention=False
        )

        med = forecast_df["bistro_med"].values.astype(float)
        actual = forecast_df["actual"].values.astype(float)
        ar1 = forecast_df["ar1"].values.astype(float)
        valid = ~np.isnan(actual)

        rmse = float(np.sqrt(np.mean((med[valid] - actual[valid]) ** 2)))
        rmse_ar1 = float(np.sqrt(np.mean((ar1[valid] - actual[valid]) ** 2))) if not all(np.isnan(ar1)) else None

        results[label] = {
            "rmse": rmse,
            "rmse_ar1": rmse_ar1,
            "forecast_df": forecast_df,
        }
        print(f"  RMSE: {rmse:.4f}")
        if rmse_ar1:
            print(f"  AR(1) RMSE: {rmse_ar1:.4f}")
        print()

# Summary
print("\n" + "="*60)
print("  COMPARISON SUMMARY")
print("="*60)
print(f"{'':20s} {'CTX=36':>10s} {'CTX=120':>10s} {'Diff':>10s}")
print("-"*60)
for year in ["2023", "2024"]:
    r36 = results[f"{year}_CTX36"]["rmse"]
    r120 = results[f"{year}_CTX120"]["rmse"]
    diff = r36 - r120
    pct = (diff / r120) * 100
    print(f"  {year} BISTRO RMSE   {r36:10.4f} {r120:10.4f} {diff:+10.4f} ({pct:+.1f}%)")
    a36 = results[f"{year}_CTX36"]["rmse_ar1"]
    a120 = results[f"{year}_CTX120"]["rmse_ar1"]
    if a36 and a120:
        diff_a = a36 - a120
        print(f"  {year} AR(1) RMSE   {a36:10.4f} {a120:10.4f} {diff_a:+10.4f}")
