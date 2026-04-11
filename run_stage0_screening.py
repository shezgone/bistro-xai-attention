"""
Stage 0: Full-variable screening with short context (CTX=10).
288 variables in a single pass → attention ranking only.
Then Stage 1: top 25 with full context (CTX=120) → attention + ablation.
"""
import sys, os
import json
import time
import numpy as np
import pandas as pd

BISTRO_REPO = "/tmp/bistro-repo"
MODEL_PATH  = f"{BISTRO_REPO}/bistro-finetuned"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

from bistro_runner_30var import (
    TARGET_COL, PDT, PSZ, BSZ, FORECAST_START_DATE, FREQ,
    WINDOW_DISTANCE, install_attention_hooks,
)
from bistro_core import BISTROConfig, AttentionHookManager, AttentionAnalyzer

TOURNAMENT_PANEL = os.path.join(os.path.dirname(__file__), "data", "macro_panel_tournament_daily.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "stage0")


# ============================================================
# Stage 0: Short-context full-variable screening
# ============================================================

def run_stage0(ctx_patches=10):
    """CTX=10으로 288개 변수 전부 한 번에 추론 → attention 순위."""
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
    from preprocessing_util import aggregate_daily_forecast_to_monthly

    print("=" * 60)
    print(f"STAGE 0: Full screening (CTX={ctx_patches}, all variables)")
    print("=" * 60)

    # Model
    print("Loading model...")
    module = MoiraiModule.from_pretrained(MODEL_PATH)
    module.eval()
    n_layers = len(list(module.encoder.layers))
    print(f"  {n_layers} layers")

    # Panel
    panel = pd.read_csv(TOURNAMENT_PANEL, index_col=0, parse_dates=True)
    panel = panel.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    covariates = [c for c in panel.columns if c != TARGET_COL]
    n_variates = 1 + len(covariates)

    print(f"  {n_variates} variates (1 target + {len(covariates)} covariates)")
    print(f"  Tokens: {n_variates} × {ctx_patches} = {n_variates * ctx_patches}")

    # Prepare data
    df = panel[[TARGET_COL] + covariates].copy()
    df.columns = ["target"] + covariates
    df = df.sort_index()

    forecast_start_dt = pd.Timestamp(FORECAST_START_DATE)
    cutoff_dt = forecast_start_dt - pd.Timedelta(days=1)

    pdt_steps = PSZ * PDT
    ctx_steps = PSZ * ctx_patches
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

    ds = PandasDataset.from_long_dataframe(
        df, item_id="item_id",
        past_feat_dynamic_real=covariates, feat_dynamic_real=[],
    )
    _, test_template = split(ds, date=cutoff_period)
    test_data = test_template.generate_instances(
        prediction_length=pdt_steps, windows=1,
        distance=dist_steps, max_history=ctx_steps,
    )

    # Attention hooks
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

    print(f"\n  Running inference (CTX={ctx_patches})...")
    t0 = time.time()
    forecasts = list(predictor.predict(test_data.input))
    t_infer = time.time() - t0
    restore_fn()
    print(f"  Inference done: {t_infer:.1f}s")

    # Attention ranking
    layer_names = sorted(captured.keys(),
                         key=lambda x: int(x.split(".")[2]))
    attn_arrays = np.stack([captured[k] for k in layer_names], axis=0)
    actual_ctx = attn_arrays.shape[-1] // n_variates

    print(f"  Attention shape: {attn_arrays.shape}")
    print(f"  Actual ctx_patches: {actual_ctx}")

    cfg = BISTROConfig(
        variates=[TARGET_COL] + covariates, target_idx=0,
        ctx_patches=actual_ctx, pdt_patches=PDT, patch_size=PSZ,
    )
    hook_mgr = AttentionHookManager()
    for i, name in enumerate(layer_names):
        hook_mgr.attention_maps[name] = attn_arrays[i][np.newaxis, np.newaxis, ...]
    ana = AttentionAnalyzer(cfg, hook_mgr)

    imp = ana.target_importance()
    self_attn = float(imp[TARGET_COL])
    cov_imp = imp.drop(TARGET_COL).sort_values(ascending=False)

    print(f"\n  Self-attention: {self_attn:.1%}")
    print(f"\n  Top 30 covariates by attention:")
    for i, (v, a) in enumerate(cov_imp.head(30).items(), 1):
        print(f"    {i:>3}. {v:<30s} {a:.4f} ({a:.2%})")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "stage0_ranking.npz")
    np.savez(out_path,
        variates=np.array([TARGET_COL] + covariates),
        n_variates=np.array(n_variates),
        ctx_patches=np.array(actual_ctx),
        ranking_vars=np.array(cov_imp.index.tolist()),
        ranking_attn=np.array(cov_imp.values),
        self_attn=np.array(self_attn),
    )
    print(f"\n  Saved: {out_path}")

    return cov_imp, self_attn, module, n_layers


# ============================================================
# Stage 1: Full-context re-inference with top N
# ============================================================

def run_stage1(top_vars, module, n_layers, ctx_patches=120):
    """CTX=120으로 선별된 변수만 재추론 + ablation."""
    from bistro_runner_30var import run_bistro_inference_daily

    print(f"\n{'=' * 60}")
    print(f"STAGE 1: Re-inference with {len(top_vars)} vars (CTX={ctx_patches})")
    print(f"  Variables: {top_vars}")
    print("=" * 60)

    panel = pd.read_csv(TOURNAMENT_PANEL, index_col=0, parse_dates=True)
    panel = panel.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    panel = panel[[TARGET_COL] + top_vars]

    # Full inference
    forecast_df, preds_bl, prep, pred, captured, layers = \
        run_bistro_inference_daily(panel, top_vars, module, n_layers,
                                   capture_attention=True)

    actual = forecast_df["actual"].values.astype(float)
    preds = forecast_df["bistro_med"].values.astype(float)
    valid = ~np.isnan(actual)
    baseline_rmse = float(np.sqrt(np.mean((preds[valid] - actual[valid]) ** 2)))

    print(f"\n  Baseline RMSE: {baseline_rmse:.4f}")
    print(f"\n  Forecast:")
    print(forecast_df.to_string(index=False))

    # Attention ranking (full context)
    variates = [TARGET_COL] + top_vars
    attn_arrays = np.stack([captured[k] for k in layers], axis=0)
    actual_ctx = attn_arrays.shape[-1] // len(variates)

    cfg = BISTROConfig(
        variates=variates, target_idx=0,
        ctx_patches=actual_ctx, pdt_patches=PDT, patch_size=PSZ,
    )
    hook_mgr = AttentionHookManager()
    for i, name in enumerate(layers):
        hook_mgr.attention_maps[name] = attn_arrays[i][np.newaxis, np.newaxis, ...]
    ana = AttentionAnalyzer(cfg, hook_mgr)

    imp = ana.target_importance()
    self_attn = float(imp[TARGET_COL])
    cov_imp = imp.drop(TARGET_COL).sort_values(ascending=False)

    print(f"\n  Stage 1 self-attention: {self_attn:.1%}")
    print(f"  Stage 1 attention ranking:")
    for i, (v, a) in enumerate(cov_imp.items(), 1):
        print(f"    {i:>2}. {v:<30s} {a:.4f} ({a:.2%})")

    # Ablation
    print(f"\n  Ablation (leave-one-out):")
    ablation_delta = {}
    for var in top_vars:
        remaining = [c for c in top_vars if c != var]
        try:
            panel_lo = panel[[TARGET_COL] + remaining].copy()
            fc_lo, _, _, _, _, _ = run_bistro_inference_daily(
                panel_lo, remaining, module, n_layers, capture_attention=False
            )
            actual_lo = fc_lo["actual"].values.astype(float)
            preds_lo = fc_lo["bistro_med"].values.astype(float)
            valid_lo = ~np.isnan(actual_lo)
            rmse_lo = float(np.sqrt(np.mean((preds_lo[valid_lo] - actual_lo[valid_lo]) ** 2)))
            delta = rmse_lo - baseline_rmse
            ablation_delta[var] = delta
            status = "harmful" if delta <= 0 else "helpful"
            print(f"    {var:<30s} ΔRMSE={delta:+.4f} [{status}]")
        except Exception as e:
            print(f"    {var:<30s} FAILED: {e}")
            ablation_delta[var] = 0.0

    # Final selection: remove harmful
    harmful = [v for v, d in ablation_delta.items() if d <= 0]
    final = [v for v in top_vars if v not in harmful]

    print(f"\n  Harmful (removed): {harmful}")
    print(f"  Final selection: {len(final)} vars — {final}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "stage1_results.npz")
    np.savez(out_path,
        variates=np.array(variates),
        n_variates=np.array(len(variates)),
        top_vars=np.array(top_vars),
        final_vars=np.array(final),
        harmful_vars=np.array(harmful),
        baseline_rmse=np.array(baseline_rmse),
        ablation_vars=np.array(list(ablation_delta.keys())),
        ablation_delta=np.array(list(ablation_delta.values())),
        ranking_vars=np.array(cov_imp.index.tolist()),
        ranking_attn=np.array(cov_imp.values),
        self_attn=np.array(self_attn),
        forecast_date=np.array(forecast_df["date"].values),
        forecast_med=np.array(preds, dtype=float),
        forecast_actual=np.array(actual, dtype=float),
    )
    print(f"\n  Saved: {out_path}")

    return final, baseline_rmse


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-ctx", type=int, default=10)
    parser.add_argument("--top-n", type=int, default=25)
    args = parser.parse_args()

    # Stage 0
    cov_imp, self_attn, module, n_layers = run_stage0(ctx_patches=args.stage0_ctx)

    # Top N for Stage 1
    top_n = args.top_n
    top_vars = cov_imp.head(top_n).index.tolist()

    print(f"\n{'=' * 60}")
    print(f"Stage 0 → Stage 1: top {top_n} selected")
    print(f"{'=' * 60}")

    # Stage 1
    final, rmse = run_stage1(top_vars, module, n_layers)

    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"  Stage 0: 288 vars (CTX={args.stage0_ctx}) → top {top_n}")
    print(f"  Stage 1: {top_n} vars (CTX=120) → {len(final)} final")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Final: {final}")
    print(f"{'=' * 60}")
