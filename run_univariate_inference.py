"""
Univariate inference: CPI_KR_YoY only, no covariates.
Saves forecast to data/forecast_univariate.npz for dashboard overlay.
"""
import sys, os
import numpy as np
import pandas as pd

BISTRO_REPO = "/tmp/bistro-repo"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

from bistro_runner_30var import (
    TARGET_COL, CTX, PDT, PSZ, BSZ, MODEL_PATH,
    FORECAST_START_DATE, FREQ, WINDOW_DISTANCE,
)

from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from preprocessing_util import aggregate_daily_forecast_to_monthly

print("Loading BISTRO model...")
module = MoiraiModule.from_pretrained(MODEL_PATH)
module.eval()
print("  Model loaded.")

# Load daily panel, target only
panel_path = os.path.join(os.path.dirname(__file__), "data", "macro_panel_daily.csv")
panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
panel = panel[[TARGET_COL]].ffill().bfill()

df = panel.copy()
df.columns = ["target"]
df = df.sort_index()

forecast_start_dt = pd.Timestamp(FORECAST_START_DATE)
cutoff_dt = forecast_start_dt - pd.Timedelta(days=1)

pdt_steps = PSZ * PDT
ctx_steps = PSZ * CTX
dist_steps = PSZ * WINDOW_DISTANCE

# Padding
df_after = df.loc[forecast_start_dt:]
n_needed = pdt_steps
if len(df_after) < n_needed:
    last_date = df.index[-1]
    pad_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                              periods=n_needed - len(df_after), freq="D")
    pad_vals = [-1 if i % 2 == 0 else -2 for i in range(len(pad_dates))]
    pad_df = pd.DataFrame({"target": pad_vals}, index=pad_dates)
    df = pd.concat([df, pad_df])
    df = df[~df.index.duplicated(keep="first")].sort_index()

df["item_id"] = "cpi_kr_yoy"
cutoff_period = pd.Period(cutoff_dt.strftime("%Y-%m-%d"), freq="D")

ds = PandasDataset.from_long_dataframe(df, item_id="item_id")
_, test_template = split(ds, date=cutoff_period)
test_data = test_template.generate_instances(
    prediction_length=pdt_steps, windows=1,
    distance=dist_steps, max_history=ctx_steps,
)

model = MoiraiForecast(
    module=module,
    prediction_length=pdt_steps,
    context_length=ctx_steps,
    patch_size=PSZ,
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
)
predictor = model.create_predictor(batch_size=BSZ)

inputs = list(test_data.input)
labels = list(test_data.label)
forecasts = list(predictor.predict(test_data.input))

samples = np.asarray(forecasts[0].samples, dtype=float)
label_target = np.asarray(labels[0]["target"], dtype=float)
inp_target = np.asarray(inputs[0]["target"], dtype=float)
last_input = float(inp_target[-1]) if inp_target.size > 0 else None

preds, gts, ci = aggregate_daily_forecast_to_monthly(
    samples, label_target, last_input,
    steps_per_period=PSZ, expected_periods=PDT,
)

forecast_start = pd.Period(FORECAST_START_DATE, freq=FREQ)
pred_index = pd.period_range(start=forecast_start, periods=PDT, freq=FREQ)

dates = [str(p) for p in pred_index]
print("\nForecast (univariate):")
for d, p in zip(dates, preds):
    print(f"  {d}: {p:.4f}")

out_path = os.path.join(os.path.dirname(__file__), "data", "forecast_univariate.npz")
np.savez(out_path,
    forecast_date=np.array(dates),
    forecast_med=np.array(preds, dtype=float),
    forecast_ci_lo=np.array(ci[:, 0], dtype=float),
    forecast_ci_hi=np.array(ci[:, 1], dtype=float),
    n_variates=np.array(1),
)
print(f"\nSaved: {out_path}")
