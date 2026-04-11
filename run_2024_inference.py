"""
2024 Forecast: context up to 2023-12, forecast 2024-01 ~ 2024-12.
Uses the optimal 18-variable set from Stage 0→1 screening.
Saves to data/forecast_optimal18_2024.npz for dashboard overlay.
"""
import sys, os
import numpy as np
import pandas as pd

BISTRO_REPO = "/tmp/bistro-repo"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

import bistro_runner_30var as brm

# ── Override forecast start to 2024 ──
brm.FORECAST_START_DATE = "2024-01-01"
# ── Override daily panel path to tournament panel (288 vars incl. optimal 18) ──
TOURNAMENT_PANEL = os.path.join(os.path.dirname(__file__), "data", "macro_panel_tournament_daily.csv")
brm.PANEL_DAILY_CSV = TOURNAMENT_PANEL

from bistro_runner_30var import (
    load_daily_panel, run_bistro_inference_daily,
    TARGET_COL, CTX, PDT, PSZ, MODEL_PATH,
)

# Optimal 18 covariates (from Stage 0→1 screening)
OPTIMAL_18 = [
    "AUD_USD", "CN_Interbank3M", "US_UnempRate", "JP_REER",
    "JP_Interbank3M", "JP_CoreCPI", "KC_FSI", "KR_MfgProd",
    "Pork", "US_NFP", "US_TradeTransEmp", "THB_USD",
    "PPI_CopperNickel", "CN_PPI", "US_Mortgage15Y", "UK_10Y_Bond",
    "US_ExportPI", "US_DepInstCredit",
]

print(f"Forecast start: {brm.FORECAST_START_DATE}")
print(f"Context: ~2013-01 ~ 2023-12 ({CTX} patches)")
print(f"Prediction: 2024-01 ~ 2024-12 ({PDT} patches)")
print(f"Panel: {TOURNAMENT_PANEL}")
print(f"Covariates ({len(OPTIMAL_18)}): {OPTIMAL_18}")

from uni2ts.model.moirai import MoiraiModule
print("\nLoading BISTRO model...")
module = MoiraiModule.from_pretrained(MODEL_PATH)
module.eval()
n_layers = len(list(module.encoder.layers))
print(f"  {n_layers} layers loaded.")

panel, covariates = load_daily_panel(covariate_subset=OPTIMAL_18)
n_var = 1 + len(covariates)
print(f"  {n_var} variates (1 target + {len(covariates)} covariates)")

forecast_df, preds_bl, prep, pred, captured, layers = \
    run_bistro_inference_daily(panel, covariates, module, n_layers, capture_attention=False)

print("\nForecast:")
print(forecast_df.to_string(index=False))

out_path = os.path.join(os.path.dirname(__file__), "data", "forecast_optimal18_2024.npz")
np.savez(out_path,
    forecast_date=np.array(forecast_df["date"].values),
    forecast_med=np.array(forecast_df["bistro_med"].values, dtype=float),
    forecast_ci_lo=np.array(forecast_df["ci_lo"].values, dtype=float),
    forecast_ci_hi=np.array(forecast_df["ci_hi"].values, dtype=float),
    forecast_actual=np.array(forecast_df["actual"].values, dtype=float),
    forecast_ar1=np.array(forecast_df["ar1"].values, dtype=float),
    covariates=np.array(OPTIMAL_18),
    n_variates=np.array(n_var),
)
print(f"\nSaved: {out_path}")
