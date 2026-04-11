"""
Recent-attention-based 9-variable inference.
Top 9 covariates by last-12-patch (≈1 year) attention from Stage 1.
Saves forecast to data/forecast_recent9.npz for dashboard overlay.
"""
import sys, os
import numpy as np
import pandas as pd

BISTRO_REPO = "/tmp/bistro-repo"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

from bistro_runner_30var import (
    load_daily_panel, run_bistro_inference_daily,
    TARGET_COL, MODEL_PATH,
)

# Recent 1-year attention top 9 (from Stage 1 last 12 patches)
RECENT_TOP9 = [
    "CNY_USD", "US_ConsConf", "US_M2", "US_YieldSpread",
    "JP_Interbank3M", "China_CPI", "DXY_Broad", "US_Unemp", "FedFunds",
]

print(f"Recent-attention top 9: {RECENT_TOP9}")

from uni2ts.model.moirai import MoiraiModule
print("Loading BISTRO model...")
module = MoiraiModule.from_pretrained(MODEL_PATH)
module.eval()
n_layers = len(list(module.encoder.layers))
print(f"  {n_layers} layers loaded.")

panel, covariates = load_daily_panel(covariate_subset=RECENT_TOP9)
n_var = 1 + len(covariates)
print(f"  {n_var} variates (1 target + {len(covariates)} covariates)")

forecast_df, preds_bl, prep, pred, captured, layers = \
    run_bistro_inference_daily(panel, covariates, module, n_layers, capture_attention=False)

print("\nForecast:")
print(forecast_df.to_string(index=False))

out_path = os.path.join(os.path.dirname(__file__), "data", "forecast_recent9.npz")
np.savez(out_path,
    forecast_date=np.array(forecast_df["date"].values),
    forecast_med=np.array(forecast_df["bistro_med"].values, dtype=float),
    forecast_ci_lo=np.array(forecast_df["ci_lo"].values, dtype=float),
    forecast_ci_hi=np.array(forecast_df["ci_hi"].values, dtype=float),
    covariates=np.array(RECENT_TOP9),
    n_variates=np.array(n_var),
)
print(f"\nSaved: {out_path}")
