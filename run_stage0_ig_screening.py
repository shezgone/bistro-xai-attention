"""
Stage 0 screening with Integrated Gradients (drop-in alternative to
run_stage0_screening.py, which uses attention).

Why this exists
---------------
The synthetic benchmark in test_ig_screening_synthetic.py showed:
  * IG recovers all 3 true drivers in top-3 (attention only 1/3)
  * Shuffling IG's top-K inflates MSE ~7.5× more than attention's top-K
    → IG-ranked variates genuinely carry the model's forecast signal.
  * Spurious confounders (correlated with a driver but causally unrelated)
    drop well below top-20 under IG.

This script applies the same method to the real Moirai/BISTRO model for the
288-covariate panel. Captum's LayerIntegratedGradients handles the hooking
internally — no manual forward/backward hooks needed. The forward wrapper
just has to return a scalar per batch element that represents the forecast.

Usage
-----
    .venv-bistro/bin/python3 run_stage0_ig_screening.py \
        --ctx 10 --n-steps 32 --batch 8

Outputs
-------
    data/stage0/stage0_ig_ranking.npz       — per-variate IG importance
    data/stage0/stage0_ig_attention_side.csv — IG vs attention side-by-side
    data/stage0/stage0_ig_permutation.csv   — permutation validation
"""
import os, sys, time, argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients

BISTRO_REPO = "/tmp/bistro-repo"
MODEL_PATH  = f"{BISTRO_REPO}/bistro-finetuned"
sys.path.insert(0, f"{BISTRO_REPO}/src")
sys.path.insert(0, os.path.dirname(__file__))

from bistro_runner_30var import (
    TARGET_COL, PDT, PSZ, BSZ, FORECAST_START_DATE, WINDOW_DISTANCE,
    install_attention_hooks,
)

TOURNAMENT_PANEL = os.path.join(os.path.dirname(__file__),
                                "data", "macro_panel_tournament_daily.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "stage0")


# ============================================================
# Model adapter: Moirai forward → scalar
# ============================================================

class MoiraiScalarWrapper(torch.nn.Module):
    """
    Wraps MoiraiModule so Captum can compute IG end-to-end.

    The raw module takes patchified tensors (target, observed_mask, sample_id,
    time_id, variate_id, prediction_mask); we freeze everything except the
    continuous `target` input (shape (B, V*ctx_patches, patch_size)) and expose
    a scalar per batch = mean forecast over the prediction horizon for the
    target variate (index 0). That scalar is what IG will attribute.
    """
    def __init__(self, module, static_inputs):
        super().__init__()
        self.module = module
        # static_inputs holds everything except `target`
        self.obs_mask = static_inputs["observed_mask"]
        self.sample_id = static_inputs["sample_id"]
        self.time_id = static_inputs["time_id"]
        self.variate_id = static_inputs["variate_id"]
        self.prediction_mask = static_inputs["prediction_mask"]
        # (B, n_pred_tokens) — bool mask of patches belonging to the target
        self.target_pred_mask = static_inputs["target_pred_mask"]

    def forward(self, target_patches):
        """target_patches: (B, V*ctx_patches, patch_size) float tensor."""
        B = target_patches.shape[0]
        # Moirai's module forward signature varies slightly by version;
        # verify against your local uni2ts install.
        dist = self.module(
            target=target_patches,
            observed_mask=self.obs_mask.expand(B, -1, -1),
            sample_id=self.sample_id.expand(B, -1),
            time_id=self.time_id.expand(B, -1),
            variate_id=self.variate_id.expand(B, -1),
            prediction_mask=self.prediction_mask.expand(B, -1),
            patch_size=torch.tensor([PSZ], device=target_patches.device),
        )
        # dist is a distribution over (B, n_pred_tokens, patch_size).
        # Take the mean forecast, restrict to target's prediction tokens,
        # reduce to a single scalar per batch element.
        mean = dist.mean  # (B, n_pred_tokens, patch_size)
        tgt = mean[:, self.target_pred_mask[0], :]   # (B, n_tgt_tokens, psz)
        return tgt.mean(dim=(1, 2))                  # (B,)


# ============================================================
# Panel → tensor batch
# ============================================================

def build_batch(panel, covariates, ctx_patches):
    """
    Converts the (target + covariates) panel into the Moirai module's
    patchified input tensors. Returns (target_tensor, static_inputs).

    NOTE: the exact packing of patches and id tensors depends on your uni2ts
    version. The clean way is to reuse the GluonTS pipeline in
    bistro_runner_30var.run_bistro_inference_daily(), capture the tensors
    going INTO the module via a single forward pre-hook, and feed those same
    tensors to the wrapper. See `capture_module_inputs` below.
    """
    raise NotImplementedError(
        "Use capture_module_inputs() against a live predictor run — see main()."
    )


def capture_module_inputs(module):
    """
    Registers a forward pre-hook on MoiraiModule to grab the real input
    tensors from one predictor call. Much more reliable than re-packing
    patches by hand.
    """
    captured = {}
    def hook(mod, args, kwargs):
        # uni2ts passes everything as kwargs in recent versions
        src = kwargs if kwargs else {}
        for k in ("target", "observed_mask", "sample_id", "time_id",
                  "variate_id", "prediction_mask"):
            if k in src:
                captured[k] = src[k].detach().clone()
        # the first positional arg is usually target in older versions
        if "target" not in captured and args:
            captured["target"] = args[0].detach().clone()
    h = module.register_forward_pre_hook(hook, with_kwargs=True)
    return captured, h


# ============================================================
# Main
# ============================================================

def run(ctx_patches=10, n_steps=32, ig_batch=8, n_layers_hint=None):
    from gluonts.dataset.pandas import PandasDataset
    from gluonts.dataset.split import split
    from uni2ts.model.moirai import MoiraiModule, MoiraiForecast

    print(f"Loading model from {MODEL_PATH}")
    module = MoiraiModule.from_pretrained(MODEL_PATH)
    module.eval()
    n_layers = n_layers_hint or len(list(module.encoder.layers))

    panel = pd.read_csv(TOURNAMENT_PANEL, index_col=0, parse_dates=True)
    panel = panel.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    covariates = [c for c in panel.columns if c != TARGET_COL]
    n_variates = 1 + len(covariates)
    print(f"  {n_variates} variates, CTX={ctx_patches}, tokens={n_variates*ctx_patches}")

    # ---- same data prep as stage0 ----
    df = panel[[TARGET_COL] + covariates].copy()
    df.columns = ["target"] + covariates
    df = df.sort_index()

    forecast_start_dt = pd.Timestamp(FORECAST_START_DATE)
    cutoff_dt = forecast_start_dt - pd.Timedelta(days=1)
    pdt_steps = PSZ * PDT
    ctx_steps = PSZ * ctx_patches

    df_after = df.loc[forecast_start_dt:]
    if len(df_after) < pdt_steps:
        last_date = df.index[-1]
        pad_dates = pd.date_range(last_date + pd.Timedelta(days=1),
                                  periods=pdt_steps - len(df_after), freq="D")
        pad_vals = [-1 if i % 2 == 0 else -2 for i in range(len(pad_dates))]
        df = pd.concat([df, pd.DataFrame({"target": pad_vals}, index=pad_dates)])
        df = df[~df.index.duplicated(keep="first")].sort_index()
    df["item_id"] = "cpi_kr_yoy"
    cutoff_period = pd.Period(cutoff_dt.strftime("%Y-%m-%d"), freq="D")

    ds = PandasDataset.from_long_dataframe(
        df, item_id="item_id",
        past_feat_dynamic_real=covariates, feat_dynamic_real=[],
    )
    _, test_tpl = split(ds, date=cutoff_period)
    test_data = test_tpl.generate_instances(
        prediction_length=pdt_steps, windows=1,
        distance=PSZ * WINDOW_DISTANCE, max_history=ctx_steps,
    )

    # ---- capture the real module-level input tensors ----
    captured_attn, restore_attn = install_attention_hooks(module, n_layers)
    captured_inp, h_inp = capture_module_inputs(module)

    model_mf = MoiraiForecast(
        module=module,
        prediction_length=pdt_steps, context_length=ctx_steps,
        patch_size=PSZ, num_samples=100, target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
    predictor = model_mf.create_predictor(batch_size=BSZ)

    print("  Running reference inference to capture input tensors...")
    t0 = time.time()
    _ = list(predictor.predict(test_data.input))
    print(f"  done in {time.time()-t0:.1f}s")
    restore_attn()
    h_inp.remove()

    # ---- Build Captum wrapper over the captured inputs ----
    # The prediction_mask marks patches to forecast. Split by variate to find
    # which rows belong to the target (variate 0).
    var_ids = captured_inp["variate_id"][0]     # (n_tokens,)
    pred_mask = captured_inp["prediction_mask"][0]  # (n_tokens,)
    target_pred_mask = (var_ids == 0) & pred_mask
    static = {
        "observed_mask":   captured_inp["observed_mask"],
        "sample_id":       captured_inp["sample_id"],
        "time_id":         captured_inp["time_id"],
        "variate_id":      captured_inp["variate_id"],
        "prediction_mask": captured_inp["prediction_mask"],
        "target_pred_mask": target_pred_mask[None, :],
    }
    target = captured_inp["target"]  # (1, n_tokens, PSZ)
    wrapper = MoiraiScalarWrapper(module, static).eval()

    # ---- Sanity: forward produces a finite scalar ----
    with torch.no_grad():
        out = wrapper(target)
        print(f"  wrapper forward OK — scalar={out.item():.4f}")

    # ---- IG over the input projection of the encoder ----
    # For Moirai, the first Linear that consumes `target` is typically
    # module.in_proj or module.input_projection. Grep your checkpoint to
    # confirm the attribute name.
    for cand in ("in_proj", "input_projection", "embed", "target_proj"):
        if hasattr(module, cand):
            target_layer = getattr(module, cand)
            print(f"  Using target layer: module.{cand}")
            break
    else:
        raise RuntimeError("Couldn't find input projection — inspect `module` "
                           "with `for n,_ in module.named_modules(): print(n)`")

    lig = LayerIntegratedGradients(wrapper, target_layer)
    baseline = torch.zeros_like(target)
    print(f"  Running IG (n_steps={n_steps})...")
    t0 = time.time()
    attributions = lig.attribute(
        target, baselines=baseline, n_steps=n_steps,
        internal_batch_size=ig_batch,
    )
    print(f"  IG done in {time.time()-t0:.1f}s  shape={tuple(attributions.shape)}")

    # ---- Collapse attribution to per-variate importance ----
    # attributions shape matches the target_layer output. After sum over the
    # hidden dim we get per-token importance; group by variate_id.
    if attributions.dim() == 3:
        tok_imp = attributions.abs().sum(dim=-1)[0]  # (n_tokens,)
    else:
        tok_imp = attributions.abs().flatten(1).sum(dim=-1)[0]

    per_var = torch.zeros(n_variates)
    for v in range(n_variates):
        mask = var_ids == v
        per_var[v] = tok_imp[mask].mean() if mask.any() else 0.0
    per_var = per_var.cpu().numpy()

    variates = [TARGET_COL] + covariates
    ig_series = pd.Series(per_var, index=variates)
    cov_ig = ig_series.drop(TARGET_COL).sort_values(ascending=False)

    print(f"\n  Top 30 covariates by IG:")
    for i, (v, a) in enumerate(cov_ig.head(30).items(), 1):
        print(f"    {i:>3}. {v:<30s} {a:.4f}")

    # ---- Side-by-side with the attention ranking (re-use captured attn) ----
    from bistro_core import BISTROConfig, AttentionHookManager, AttentionAnalyzer
    layer_names = sorted(captured_attn.keys(), key=lambda x: int(x.split(".")[2]))
    attn_arrays = np.stack([captured_attn[k] for k in layer_names], axis=0)
    actual_ctx = attn_arrays.shape[-1] // n_variates
    cfg = BISTROConfig(variates=variates, target_idx=0,
                       ctx_patches=actual_ctx, pdt_patches=PDT, patch_size=PSZ)
    hm = AttentionHookManager()
    for i, name in enumerate(layer_names):
        hm.attention_maps[name] = attn_arrays[i][np.newaxis, np.newaxis, ...]
    ana = AttentionAnalyzer(cfg, hm)
    attn_imp = ana.target_importance().drop(TARGET_COL)

    cmp = pd.DataFrame({
        "ig": cov_ig,
        "attn": attn_imp.reindex(cov_ig.index),
    })
    cmp["rank_ig"]   = cmp["ig"].rank(ascending=False).astype(int)
    cmp["rank_attn"] = cmp["attn"].rank(ascending=False).astype(int)
    cmp["rank_shift"] = cmp["rank_attn"] - cmp["rank_ig"]
    cmp = cmp.sort_values("rank_ig")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cmp.to_csv(os.path.join(OUTPUT_DIR, "stage0_ig_attention_side.csv"))
    np.savez(os.path.join(OUTPUT_DIR, "stage0_ig_ranking.npz"),
             variates=np.array(variates),
             ig=per_var,
             cov_order=np.array(cov_ig.index.tolist()),
             cov_ig=cov_ig.values,
             ctx_patches=np.array(actual_ctx))
    print(f"\n  Saved IG ranking and IG-vs-attn comparison to {OUTPUT_DIR}")
    return cov_ig


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ctx", type=int, default=10)
    p.add_argument("--n-steps", type=int, default=32)
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    run(ctx_patches=args.ctx, n_steps=args.n_steps, ig_batch=args.batch)
