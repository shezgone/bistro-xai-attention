"""
Real-panel IG screening test.

Data: macro_panel_daily.csv  (28 real macro covariates + CPI_KR_YoY, 2003-2025)
Model: same tiny variate transformer as the synthetic test
Task: predict CPI_KR_YoY[t + HORIZON days] from a T-day window of all 29 vars
Goal: answer "are IG-top covariates economically interpretable?"

Caveat: This is NOT the Moirai foundation model. It's a small trained-from-
scratch variate transformer. But it uses the same IG machinery on real macro
data, which is sufficient to show whether IG's top picks line up with
well-known CPI drivers (oil, FX, US CPI, rates, etc.) on an honest dataset.
"""
from __future__ import annotations
import math, os, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from captum.attr import LayerIntegratedGradients

ROOT = Path(__file__).parent
PANEL = ROOT / "data" / "macro_panel_daily.csv"
OUT   = ROOT / "data" / "ig_real"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED)

TARGET = "CPI_KR_YoY"
T = 90                  # lookback window in days (~3 months)
HORIZON = 30            # predict CPI at t + 30d (~1 month ahead)
D = 32
N_LAYERS = 3
N_HEADS = 4
BATCH = 128
EPOCHS = 60
LR = 1e-3


# ============================================================
# Data
# ============================================================

def load_windows():
    df = pd.read_csv(PANEL, index_col=0, parse_dates=True).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    # Put target first (mirrors BISTRO: target at index 0)
    cols = [TARGET] + [c for c in df.columns if c != TARGET]
    df = df[cols]

    # z-score per variate using only pre-2020 stats (no leakage)
    norm_end = pd.Timestamp("2020-01-01")
    train_mask = df.index < norm_end
    mean = df.loc[train_mask].mean()
    std  = df.loc[train_mask].std().replace(0, 1)
    dfz = (df - mean) / std

    V = dfz.shape[1]
    arr = dfz.values.astype(np.float32)
    dates = dfz.index

    # Build (window, target) pairs — window is past T days of all V vars,
    # target is the (normalized) CPI value at t + HORIZON days later
    windows, targets, window_end = [], [], []
    for end in range(T - 1, len(arr) - HORIZON):
        x = arr[end - T + 1 : end + 1]            # (T, V)
        y = arr[end + HORIZON, 0]                  # target variate
        windows.append(x); targets.append(y); window_end.append(dates[end])

    X = np.stack(windows).transpose(0, 2, 1)       # (N, V, T)
    y = np.array(targets, dtype=np.float32)
    window_end = pd.DatetimeIndex(window_end)

    # Time-based split
    tr_mask = window_end <  pd.Timestamp("2019-01-01")
    va_mask = (window_end >= pd.Timestamp("2019-01-01")) & (window_end < pd.Timestamp("2022-01-01"))
    te_mask = window_end >= pd.Timestamp("2022-01-01")

    return (X[tr_mask], y[tr_mask],
            X[va_mask], y[va_mask],
            X[te_mask], y[te_mask],
            cols, mean, std, window_end)


# ============================================================
# Model (identical to synthetic test)
# ============================================================

class MHA(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q = nn.Linear(d, d); self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d); self.o = nn.Linear(d, d)
        self.last_attn = None
    def forward(self, x):
        B, V, D_ = x.shape
        q = self.q(x).view(B, V, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, V, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, V, self.h, self.dh).transpose(1, 2)
        w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh), dim=-1)
        self.last_attn = w.detach()
        return self.o(torch.matmul(w, v).transpose(1, 2).contiguous().view(B, V, D_))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.attn = MHA(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class VT(nn.Module):
    def __init__(self, n_vars, t, d=D, n_layers=N_LAYERS, n_heads=N_HEADS):
        super().__init__()
        self.n_vars = n_vars
        self.embed = nn.Linear(t, d)
        self.var_emb = nn.Embedding(n_vars, d)
        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
    def forward(self, x):
        tok = self.embed(x)
        ids = torch.arange(self.n_vars, device=x.device)
        tok = tok + self.var_emb(ids).unsqueeze(0)
        for b in self.blocks:
            tok = b(tok)
        return self.head(self.ln_f(tok)[:, 0]).squeeze(-1)


# ============================================================
# Train
# ============================================================

def train(m, xt, yt, xv, yv):
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-4)
    n = xt.shape[0]; best = 1e9; best_state = None
    for ep in range(EPOCHS):
        perm = torch.randperm(n); m.train(); tot = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i+BATCH]
            xb, yb = xt[idx], yt[idx]
            p = m(xb); l = F.mse_loss(p, yb)
            opt.zero_grad(); l.backward(); opt.step()
            tot += l.item() * xb.shape[0]
        m.eval()
        with torch.no_grad():
            vl = F.mse_loss(m(xv), yv).item()
        if vl < best:
            best = vl; best_state = {k: v.clone() for k, v in m.state_dict().items()}
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"    epoch {ep+1:3d}  train={tot/n:.4f}  val={vl:.4f}")
    m.load_state_dict(best_state)
    return best


# ============================================================
# IG + attention rankings
# ============================================================

def attention_row(m, x):
    m.eval()
    with torch.no_grad():
        _ = m(x)
    A = m.blocks[-1].attn.last_attn.mean(dim=(0, 1)).cpu().numpy()
    return A[0]  # target row


def ig_per_variate(m, x, n_steps=32):
    m.eval()
    lig = LayerIntegratedGradients(m, m.embed)
    attr = lig.attribute(x, baselines=torch.zeros_like(x), n_steps=n_steps,
                         internal_batch_size=32)
    return attr.abs().sum(dim=-1).mean(dim=0).detach().cpu().numpy()


def permutation_delta(m, x, y, var_ids, n_shuf=5):
    m.eval()
    with torch.no_grad():
        base = F.mse_loss(m(x), y).item()
    rng = np.random.default_rng(SEED); deltas = []
    for _ in range(n_shuf):
        xp = x.clone()
        for v in var_ids:
            xp[:, v] = x[torch.from_numpy(rng.permutation(x.shape[0])), v]
        with torch.no_grad():
            deltas.append(F.mse_loss(m(xp), y).item() - base)
    return float(np.mean(deltas)), base


# ============================================================
# Main
# ============================================================

def main():
    print("Loading real panel...")
    Xtr, ytr, Xva, yva, Xte, yte, cols, mean, std, _ = load_windows()
    V = Xtr.shape[1]
    print(f"  Target: {cols[0]}   |   {V-1} covariates")
    print(f"  Window shape (N, V, T) = {Xtr.shape}  train={len(Xtr)}  val={len(Xva)}  test={len(Xte)}")
    print(f"  (normalization fit on 2003-2019; test windows end ≥2022)\n")

    xt = torch.from_numpy(Xtr); yt = torch.from_numpy(ytr)
    xv = torch.from_numpy(Xva); yv = torch.from_numpy(yva)
    xte = torch.from_numpy(Xte); yte_t = torch.from_numpy(yte)

    model = VT(n_vars=V, t=T)
    print(f"Training {N_LAYERS}-layer variate transformer...")
    best_val = train(model, xt, yt, xv, yv)
    with torch.no_grad():
        test_mse = F.mse_loss(model(xte), yte_t).item()
    print(f"\n  best val MSE: {best_val:.4f}")
    print(f"  test MSE:    {test_mse:.4f}  (target is z-scored; naive=1.0 would be mean)")
    naive = float(yte_t.var().item())
    print(f"  test variance (naive mean-predictor): {naive:.4f}")
    print(f"  R² on test:   {1 - test_mse / naive:.3f}")

    # Use a chunk of val data for attribution (train is too large)
    x_for_attr = xv[:256] if len(xv) >= 256 else xv

    print("\nComputing rankings...")
    attn = attention_row(model, x_for_attr)
    ig   = ig_per_variate(model, x_for_attr, n_steps=32)

    rank = pd.DataFrame({"var": cols, "ig": ig, "attn": attn})
    # Exclude the target variate (position 0) from the covariate ranking
    rank_cov = rank.iloc[1:].copy()
    rank_cov["rank_ig"] = rank_cov["ig"].rank(ascending=False).astype(int)
    rank_cov["rank_attn"] = rank_cov["attn"].rank(ascending=False).astype(int)
    rank_cov = rank_cov.sort_values("rank_ig")

    print(f"\n  Target's own self-importance (IG): {ig[0]:.3f}  "
          f"({ig[0] / ig.sum() * 100:.1f}% of total mass)")
    print(f"  Covariate IG mass: {ig[1:].sum() / ig.sum() * 100:.1f}%\n")

    print("=" * 72)
    print("TOP-10 BY IG  (real macro variable names, CPI_KR_YoY target)")
    print("=" * 72)
    print(rank_cov.head(10).to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    print("\n" + "=" * 72)
    print("TOP-10 BY ATTENTION (for comparison)")
    print("=" * 72)
    print(rank_cov.sort_values("rank_attn").head(10).to_string(index=False, float_format=lambda x: f"{x:8.4f}"))

    # Permutation validation for IG top-5
    top5_ig_idx = [cols.index(v) for v in rank_cov.sort_values("rank_ig").head(5)["var"]]
    top5_at_idx = [cols.index(v) for v in rank_cov.sort_values("rank_attn").head(5)["var"]]
    rng_pool = [i for i in range(1, V) if i not in top5_ig_idx + top5_at_idx]
    rng_pick = list(np.random.default_rng(SEED).choice(rng_pool, 5, replace=False))

    print("\n" + "=" * 72)
    print("PERMUTATION TEST on held-out val set (shuffle top-5 across samples)")
    print("=" * 72)
    for label, idxs in [("IG top-5", top5_ig_idx),
                        ("Attn top-5", top5_at_idx),
                        ("5 random covs", rng_pick)]:
        dm, base = permutation_delta(model, xv, yv, idxs)
        print(f"  {label:<16s}  baseline MSE={base:.4f}  ΔMSE={dm:+.4f}")

    # Save
    rank_cov.to_csv(OUT / "real_ig_ranking.csv", index=False)
    rank.to_csv(OUT / "real_ig_full.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    top20 = rank_cov.sort_values("rank_ig").head(20)[::-1]
    ax.barh(top20["var"], top20["ig"], color="#2ca02c")
    ax.set_title("Top-20 covariates by IG — real CPI_KR_YoY forecasting model")
    ax.set_xlabel("IG importance (|attribution| summed over embed dim, avg over samples)")
    fig.tight_layout()
    fig.savefig(OUT / "real_ig_top20.png", dpi=130)
    plt.close(fig)

    print(f"\nSaved to {OUT}:")
    for p in sorted(OUT.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
