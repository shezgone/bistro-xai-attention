"""
Tournament winner inference: 9 variables selected from 287 candidates.
Saves forecast to data/forecast_tournament9.npz for dashboard overlay.
"""
import sys, os
import numpy as np
import pandas as pd

BISTRO_REPO = "/tmp/bistro-repo"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

from bistro_runner_30var import (
    run_bistro_inference_daily, TARGET_COL, MODEL_PATH,
)

# Tournament final 9 variables
TOURNAMENT_WINNERS = [
    "TED_Spread", "Moodys_BAA", "US_MonBase", "US_UnempNSA",
    "US_MichSentiment", "HardLogs", "JP_Interbank3M", "JP_CoreCPI",
    "US_UnempRate",
]

print(f"Tournament winners (9): {TOURNAMENT_WINNERS}")

# Load tournament daily panel (has all 288 vars)
panel_path = os.path.join(os.path.dirname(__file__), "data", "macro_panel_tournament_daily.csv")
panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
panel = panel.replace([np.inf, -np.inf], np.nan).ffill().bfill()
panel = panel[[TARGET_COL] + TOURNAMENT_WINNERS]
covariates = TOURNAMENT_WINNERS

from uni2ts.model.moirai import MoiraiModule
print("Loading BISTRO model...")
module = MoiraiModule.from_pretrained(MODEL_PATH)
module.eval()
n_layers = len(list(module.encoder.layers))
print(f"  {n_layers} layers loaded.")

n_var = 1 + len(covariates)
print(f"  {n_var} variates (1 target + {len(covariates)} covariates)")

forecast_df, preds_bl, prep, pred, captured, layers = \
    run_bistro_inference_daily(panel, covariates, module, n_layers, capture_attention=False)

print("\nForecast:")
print(forecast_df.to_string(index=False))

# RMSE
actual = forecast_df["actual"].values.astype(float)
preds = forecast_df["bistro_med"].values.astype(float)
valid = ~np.isnan(actual)
rmse = np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2))
print(f"\nRMSE: {rmse:.4f}")

out_path = os.path.join(os.path.dirname(__file__), "data", "forecast_tournament9.npz")
np.savez(out_path,
    forecast_date=np.array(forecast_df["date"].values),
    forecast_med=np.array(preds, dtype=float),
    forecast_ci_lo=np.array(forecast_df["ci_lo"].values, dtype=float),
    forecast_ci_hi=np.array(forecast_df["ci_hi"].values, dtype=float),
    covariates=np.array(TOURNAMENT_WINNERS),
    n_variates=np.array(n_var),
)
print(f"\nSaved: {out_path}")
