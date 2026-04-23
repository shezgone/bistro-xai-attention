"""
Synthetic IG-vs-Attention screening benchmark.

Goal
----
Answer, with ground truth known, the three questions:
  (1) Does IG actually rank variables that drive the model's prediction?
  (2) Does IG exclude "spurious-looking" variables that attention latches onto
      (e.g. a covariate that is statistically correlated with a true driver
      but has no causal effect on the target — the "AU CPI" type trap)?
  (3) Can we visualize IG results per-variate and per-time-step?

Setup
-----
- V variates × T time steps, shape (B, V, T)
- Ground-truth drivers: three variate indices (A, B, C) contribute to y
- Confounder S: correlated with A (via shared noise) but NOT used to build y
- Remaining V-4 variates: pure noise
- Tiny variate-level transformer trained to regress y from the panel
- Compare:
    * attention ranking (target → each variate, last layer, head-avg)
    * IG ranking via Captum LayerIntegratedGradients on the input projection
- Validate with a permutation test (shuffle top-K → measure MSE spike)
"""
from __future__ import annotations
import math, os, json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import LayerIntegratedGradients

OUT_DIR = Path(__file__).parent / "data" / "ig_synthetic"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

V = 289          # 1 target + 288 covariates — matches the real BISTRO screen
T = 32           # time steps per variate
D = 32           # embedding dim
N_LAYERS = 3
N_HEADS = 4
TRUE_IDX = (10, 75, 160)     # A, B, C — the actual drivers of y
CONFOUNDER_IDX = 200         # "AU CPI": correlated with A but no causal path to y
N_TRAIN = 4096
N_VAL   = 1024
BATCH   = 128
EPOCHS  = 40
LR      = 1e-3


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def make_panel(n, rng):
    """
    Returns:
        x: (n, V, T) input panel
        y: (n,) scalar targets
    """
    x = rng.standard_normal((n, V, T)).astype(np.float32)

    # Inject temporal autocorrelation into every series (realism)
    for v in range(V):
        x[:, v] = np.cumsum(x[:, v], axis=-1) / np.sqrt(T)

    A, B, C = TRUE_IDX

    # Confounder: S = A + small noise → statistically correlated with A
    x[:, CONFOUNDER_IDX] = x[:, A] + 0.15 * rng.standard_normal((n, T)).astype(np.float32)

    # Target: y depends on A[-1], B[-3], C[-2] — NOT on S
    y = (0.7 * x[:, A, -1]
         + 0.4 * x[:, B, -3]
         + 0.5 * np.tanh(x[:, C, -2])
         + 0.10 * rng.standard_normal(n).astype(np.float32))
    return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class MHA(nn.Module):
    """Custom multi-head attention that exposes attention weights per call."""
    def __init__(self, d, n_heads):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.last_attn = None  # (B, h, V, V)

    def forward(self, x):
        B, V_, D_ = x.shape
        q = self.q(x).view(B, V_, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, V_, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, V_, self.h, self.dh).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.dh)
        raw = torch.matmul(q, k.transpose(-2, -1)) * scale
        w = torch.softmax(raw, dim=-1)
        self.last_attn = w.detach()
        out = torch.matmul(w, v).transpose(1, 2).contiguous().view(B, V_, D_)
        return self.o(out)


class Block(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MHA(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class VariateTransformer(nn.Module):
    """
    Mirrors the BISTRO-style variate-level attention: each variate → one token,
    all tokens self-attend, target token's representation → scalar forecast.
    """
    def __init__(self, n_vars=V, t=T, d=D, n_layers=N_LAYERS, n_heads=N_HEADS):
        super().__init__()
        self.n_vars = n_vars
        self.embed = nn.Linear(t, d)
        self.var_emb = nn.Embedding(n_vars, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.target_idx = 0

    def forward(self, x):  # x: (B, V, T)
        tok = self.embed(x)
        ids = torch.arange(self.n_vars, device=x.device)
        tok = tok + self.var_emb(ids).unsqueeze(0)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.ln_f(tok)
        return self.head(tok[:, self.target_idx]).squeeze(-1)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_model(model, x_tr, y_tr, x_va, y_va):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = x_tr.shape[0]
    for ep in range(EPOCHS):
        perm = torch.randperm(n)
        model.train()
        tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = x_tr[idx], y_tr[idx]
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.shape[0]
        model.eval()
        with torch.no_grad():
            val = F.mse_loss(model(x_va), y_va).item()
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:3d}  train_mse={tot/n:.4f}  val_mse={val:.4f}")
    return val


# ------------------------------------------------------------
# Attention ranking (capture last layer, target → all variates)
# ------------------------------------------------------------

def attention_ranking(model, x):
    """
    Run forward, read each block's captured attention weights, average over
    heads + batch, take the target row: target→variate.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)
    last = model.blocks[-1].attn.last_attn  # (B, h, V, V)
    A = last.mean(dim=(0, 1)).cpu().numpy() # (V, V)
    return A[model.target_idx]


# ------------------------------------------------------------
# IG ranking via Captum
# ------------------------------------------------------------

def ig_ranking(model, x_ref, n_steps=32):
    """
    LayerIntegratedGradients on the input projection.
    Baseline = zero panel (mean-ish after instance-norm-like assumption).
    """
    model.eval()
    lig = LayerIntegratedGradients(model, model.embed)
    baseline = torch.zeros_like(x_ref)
    # Captum's layer-IG attributes in the embed OUTPUT space: (B, V, D)
    attr = lig.attribute(x_ref, baselines=baseline, n_steps=n_steps,
                         internal_batch_size=32)
    # Per-variate importance = L1 mass across the embed dim, then mean across batch
    imp = attr.abs().sum(dim=-1).mean(dim=0).detach().cpu().numpy()
    return imp


def ig_time_resolved(model, x_ref, n_steps=32):
    """
    Plain IG on the raw input (B, V, T) to get time-resolved attribution.
    Useful for the "per-variate heatmap over time" visualization.
    """
    from captum.attr import IntegratedGradients
    model.eval()
    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x_ref)
    attr = ig.attribute(x_ref, baselines=baseline, n_steps=n_steps,
                        internal_batch_size=32)
    return attr.detach().cpu().numpy()  # (B, V, T)


# ------------------------------------------------------------
# Permutation test: do top-K ranked vars actually matter?
# ------------------------------------------------------------

def permutation_delta(model, x, y, var_indices, n_shuffles=5, seed=SEED):
    """
    Shuffle the listed variates across the batch dimension (break their
    information). Returns ΔMSE = mse_shuffled - mse_baseline.
    Larger ΔMSE = these variates were truly important to the model.
    """
    model.eval()
    with torch.no_grad():
        base = F.mse_loss(model(x), y).item()
    rng = np.random.default_rng(seed)
    deltas = []
    for s in range(n_shuffles):
        x_perm = x.clone()
        for v in var_indices:
            perm = torch.from_numpy(rng.permutation(x.shape[0]))
            x_perm[:, v] = x[perm, v]
        with torch.no_grad():
            m = F.mse_loss(model(x_perm), y).item()
        deltas.append(m - base)
    return float(np.mean(deltas)), float(np.std(deltas)), base


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    rng = np.random.default_rng(SEED)
    print(f"Setup: V={V} (1 target + {V-1} covariates), T={T}")
    print(f"  True drivers (A,B,C) = {TRUE_IDX}")
    print(f"  Confounder (AU-CPI type) = {CONFOUNDER_IDX}  (correlated with var {TRUE_IDX[0]})")
    print(f"  Training a {N_LAYERS}-layer variate transformer...\n")

    x_tr, y_tr = make_panel(N_TRAIN, rng)
    x_va, y_va = make_panel(N_VAL,   rng)

    # Sanity: correlation of each var with y (linear statistical link)
    with torch.no_grad():
        x_flat = x_va.mean(dim=-1).numpy()   # (N, V) — crude per-var summary
    corr = np.array([np.corrcoef(x_flat[:, v], y_va.numpy())[0, 1] for v in range(V)])
    print(f"  |corr(S, y)|  = {abs(corr[CONFOUNDER_IDX]):.3f}  "
          f"(confounder looks meaningful to a naive screen)")
    print(f"  |corr(A, y)|  = {abs(corr[TRUE_IDX[0]]):.3f}")

    model = VariateTransformer()
    val = train_model(model, x_tr, y_tr, x_va, y_va)
    print(f"\n  Final val MSE: {val:.4f}")
    print(f"  (Naive mean-predictor MSE: {y_va.var().item():.4f})\n")

    # Rank by attention
    print("Computing attention ranking (last layer, target→variates)...")
    attn = attention_ranking(model, x_va[:256])

    # Rank by IG
    print("Computing IG ranking (LayerIntegratedGradients on embed)...")
    ig_imp = ig_ranking(model, x_va[:256])

    # Drop the target itself from the rankings (we want the best covariate)
    cov_mask = np.ones(V, dtype=bool); cov_mask[0] = False
    var_ids = np.arange(V)

    attn_order = np.argsort(-attn)
    ig_order   = np.argsort(-ig_imp)

    def top_covs(order, k):
        return [v for v in order if cov_mask[v]][:k]

    top_attn = top_covs(attn_order, 20)
    top_ig   = top_covs(ig_order,   20)

    true_set = set(TRUE_IDX)
    spur = CONFOUNDER_IDX

    print("\n" + "=" * 72)
    print("TOP-20 BY ATTENTION  vs  TOP-20 BY IG")
    print("=" * 72)
    print(f"{'rank':>4}  {'attn_var':>10} {'attn_val':>10}  {'|':>2}  {'ig_var':>10} {'ig_val':>10}")
    for r in range(20):
        a = top_attn[r]; i = top_ig[r]
        mark_a = " ★" if a in true_set else (" ✗" if a == spur else "  ")
        mark_i = " ★" if i in true_set else (" ✗" if i == spur else "  ")
        print(f"{r+1:>4}  {a:>8}{mark_a}  {attn[a]:>10.4f}  |  {i:>8}{mark_i}  {ig_imp[i]:>10.4f}")
    print("  ★ = true driver    ✗ = spurious confounder (AU-CPI type)")

    # Precision@K for recovering true drivers
    def precision_at(lst, k):
        return len(set(lst[:k]) & true_set) / min(k, len(true_set))

    rows = []
    for k in [3, 5, 10, 20]:
        rows.append({
            "K": k,
            "attn_precision": precision_at(top_attn, k),
            "ig_precision":   precision_at(top_ig,   k),
            "attn_has_spurious": spur in top_attn[:k],
            "ig_has_spurious":   spur in top_ig[:k],
        })
    prec_df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("GROUND-TRUTH RECOVERY")
    print("=" * 72)
    print(prec_df.to_string(index=False))

    # Rank of the spurious confounder under each method
    spur_rank_attn = np.where(np.array(top_covs(attn_order, V))[:] == spur)[0]
    spur_rank_ig   = np.where(np.array(top_covs(ig_order,   V))[:] == spur)[0]
    print(f"\n  Spurious var #{spur} rank — attention: {int(spur_rank_attn[0])+1}, "
          f"IG: {int(spur_rank_ig[0])+1}")

    # Permutation test: shuffle top-K under each method, watch MSE rise
    print("\n" + "=" * 72)
    print("PERMUTATION TEST — do top-K selected vars actually move the model?")
    print("=" * 72)
    rows = []
    for K in [3, 5, 10]:
        for label, top in [("attention", top_attn[:K]), ("IG", top_ig[:K])]:
            dmean, dstd, base = permutation_delta(model, x_va, y_va, top)
            rows.append({"method": label, "K": K,
                         "baseline_mse": round(base, 4),
                         "mse_delta": round(dmean, 4),
                         "mse_delta_std": round(dstd, 4)})
        # also a control: shuffle K random noise vars
        noise_pool = [v for v in range(1, V) if v not in true_set and v != spur]
        random_vars = list(np.random.default_rng(SEED).choice(noise_pool, K, replace=False))
        dmean, dstd, base = permutation_delta(model, x_va, y_va, random_vars)
        rows.append({"method": f"random-noise (K={K})", "K": K,
                     "baseline_mse": round(base, 4),
                     "mse_delta": round(dmean, 4),
                     "mse_delta_std": round(dstd, 4)})
    perm_df = pd.DataFrame(rows)
    print(perm_df.to_string(index=False))

    # Save numerical outputs
    prec_df.to_csv(OUT_DIR / "precision_at_k.csv", index=False)
    perm_df.to_csv(OUT_DIR / "permutation_test.csv", index=False)
    pd.DataFrame({
        "var": var_ids, "attention": attn, "ig": ig_imp, "corr_with_y": corr,
        "is_true_driver": [v in true_set for v in var_ids],
        "is_confounder": var_ids == spur,
    }).to_csv(OUT_DIR / "full_ranking.csv", index=False)

    # ------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------
    print("\nRendering plots...")

    # (1) Side-by-side top-20 bars
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, vals, order, title in [
        (axes[0], attn, top_attn, "Attention (last layer, target row)"),
        (axes[1], ig_imp, top_ig, "Integrated Gradients (embed layer)"),
    ]:
        colors = ["#2ca02c" if v in true_set
                  else ("#d62728" if v == spur else "#7f7f7f")
                  for v in order]
        ax.barh(range(len(order))[::-1], [vals[v] for v in order], color=colors)
        ax.set_yticks(range(len(order))[::-1])
        ax.set_yticklabels([f"v{v}" for v in order], fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("importance")
    fig.suptitle("Top-20 variate ranking — green=true driver, red=spurious confounder")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ranking_bars.png", dpi=130)
    plt.close(fig)

    # (2) Scatter: attention vs IG, highlight true + spurious
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(attn[cov_mask], ig_imp[cov_mask], s=12, alpha=0.35, color="#7f7f7f", label="noise")
    for v in true_set:
        ax.scatter([attn[v]], [ig_imp[v]], s=90, color="#2ca02c",
                   edgecolor="black", label=f"true v{v}")
    ax.scatter([attn[spur]], [ig_imp[spur]], s=90, color="#d62728",
               edgecolor="black", marker="X", label=f"spurious v{spur}")
    ax.set_xlabel("attention importance"); ax.set_ylabel("IG importance")
    ax.set_title("Attention vs IG — where do the confounders land?")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "attn_vs_ig_scatter.png", dpi=130)
    plt.close(fig)

    # (3) Time-resolved IG for a handful of samples, focused on top-IG vars
    print("  Computing time-resolved IG for heatmap...")
    t_ig = ig_time_resolved(model, x_va[:32])  # (32, V, T)
    # average over samples, take top 15 IG covariates
    t_avg = np.abs(t_ig).mean(axis=0)  # (V, T)
    vars_to_show = top_ig[:15]
    sub = t_avg[vars_to_show]
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(sub, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(vars_to_show)))
    ax.set_yticklabels([f"v{v}{' ★' if v in true_set else (' ✗' if v == spur else '')}"
                        for v in vars_to_show])
    ax.set_xlabel("time step"); ax.set_ylabel("variate (top 15 by IG)")
    ax.set_title("Time-resolved |IG| — which timesteps of each variate drove the forecast")
    plt.colorbar(im, ax=ax, label="|IG|")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ig_time_heatmap.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved to {OUT_DIR}:")
    for p in sorted(OUT_DIR.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
