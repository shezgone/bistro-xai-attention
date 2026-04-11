"""
9-variable inference: exclude US_Unemp and CNY_USD from the 11-var Stage 2 set.
Saves forecast to data/forecast_9var.npz for dashboard overlay.
"""
import sys, os
import numpy as np
import pandas as pd

BISTRO_REPO = "/tmp/bistro-repo"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

from bistro_runner_30var import (
    load_daily_panel, run_bistro_inference_daily,
    TARGET_COL, CTX, PDT, PSZ, MODEL_PATH,
)

EXCLUDE = ["US_Unemp", "CNY_USD"]

# 11-var Stage 2 set (from real_inference_results.npz)
s2 = np.load(os.path.join(os.path.dirname(__file__), "data", "real_inference_results.npz"), allow_pickle=True)
s2_covariates = [str(v) for v in s2["variates"] if str(v) != TARGET_COL]

# 9-var subset
subset = [v for v in s2_covariates if v not in EXCLUDE]
print(f"9-var covariates: {subset}")

from uni2ts.model.moirai import MoiraiModule
print("Loading BISTRO model...")
module = MoiraiModule.from_pretrained(MODEL_PATH)
module.eval()
n_layers = len(list(module.encoder.layers))
print(f"  {n_layers} layers loaded.")

panel, covariates = load_daily_panel(covariate_subset=subset)
n_var = 1 + len(covariates)
print(f"  {n_var} variates (1 target + {len(covariates)} covariates)")

forecast_df, preds_bl, prep, pred, captured, layers = \
    run_bistro_inference_daily(panel, covariates, module, n_layers, capture_attention=False)

print("\nForecast:")
print(forecast_df.to_string(index=False))

out_path = os.path.join(os.path.dirname(__file__), "data", "forecast_9var.npz")
np.savez(out_path,
    forecast_date=np.array(forecast_df["date"].values),
    forecast_med=np.array(forecast_df["bistro_med"].values, dtype=float),
    forecast_ci_lo=np.array(forecast_df["ci_lo"].values, dtype=float),
    forecast_ci_hi=np.array(forecast_df["ci_hi"].values, dtype=float),
    covariates=np.array(subset),
    excluded=np.array(EXCLUDE),
    n_variates=np.array(n_var),
)
print(f"\nSaved: {out_path}")
