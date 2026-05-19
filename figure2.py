"""figure2.py — n-sweep stress test of the local-law scaling.

We have 50 cluster assignments per question. For each n in {10, 20, 30, 40, 50}
we subsample the assignments (S subsamples per Q, averaged) and recompute:

  conf_naive_n = top-cluster frequency on n samples
  conf_sssc_n  = average of agreement-with-modal across R random half-splits
                  of size  s = n // 2 / m = n - s

Then we compute ECE per n on:
  (a) all questions  →  Theorem 3.1 (bias bound) and Theorem 3.2 (ECE bound)
  (b) the low-margin subset  { q : Δ_q < √(log K / n) }
        ⇔  m̃_q = √n · Δ_q  <  √log K  →  Corollary 3.3
       The threshold tightens with n; subset is recomputed per n.

K is dataset-specific (matching Figure 3): SimpleQA 7.48, HLE 7.49, PopQA 6.05.

Theory predicts ECE_1 − ECE_2 = O(1/√n) on this low-margin subset; the figure
plots gap vs n on log-log axes with a reference slope of −1/2.

Outputs
  figures/n_sweep_full.{pdf,png}          — top row: ECE vs n on full population
  figures/n_sweep_lowdelta.{pdf,png}      — bottom row: gap on low-margin subset
  figure2.{pdf,png}                       — combined 2×3 figure for the paper
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OUT  = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.10,
})

C_SEM1 = "#D55E00"
C_SEM2 = "#0072B2"
C_GAP    = "#444444"
C_REF    = "#bd1f1f"

# (dataset_label, phase, provider) — provider="pooled" concatenates all 5 providers.
# Each (qid, provider) pair is treated as an independent row for the n-sweep
# (cluster_assignments are per-(qid, provider), so subsampling stays well-defined).
# Pooling tightens the slope fit and tests the local-law universality across
# models rather than cherry-picking one.
ALL_PROVIDERS = ["openai", "anthropic", "gemini", "xai", "mistral"]
PANEL_SPECS = [
    ("SimpleQA", "phase1", "pooled"),
    ("HLE",      "phase2", "pooled"),
    ("PopQA",    "phase4", "pooled"),
]
N_VALUES   = [10, 20, 30, 40, 50]
R_SPLITS   = 10        # number of half-splits for Sem_2
N_SUBSAMP  = 8         # subsample draws per Q per n (smooths out subsample noise)
ECE_BINS_FULL = 10

# Dataset-specific K (pooled across 5 providers, matches figure3.py).
# Low-margin subset at sweep-step n: { q : Δ_q < √(log K / n) }.
K_BY_DATASET = {
    "SimpleQA": 7.48,
    "HLE":      7.49,
    "PopQA":    6.05,
}


def low_margin_threshold(K: float, n: int) -> float:
    return float(np.sqrt(np.log(max(K, 1.0001)) / n))

# Adaptive bin count for low-Δ: each bin should hold ≥ ~MIN_PER_BIN samples,
# capped at ECE_BINS_FULL. Avoids bin-noise blow-up when N_low is small.
MIN_PER_BIN = 30


def cluster_path(phase: str, provider: str) -> Path | None:
    candidates = sorted((DATA_ROOT / phase / "clustering").glob(f"{provider}_*_clustered.jsonl"))
    candidates = [c for c in candidates if "bak" not in c.name]
    return candidates[0] if candidates else None


def graded_ids(phase: str, provider: str) -> dict[str, int]:
    out = {}
    paths = sorted((DATA_ROOT / phase / "grading").glob(f"{provider}_*_graded.jsonl"))
    if not paths:
        return out
    with open(paths[0]) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("Y") is not None:
                out[r["id"]] = int(r["Y"])
    return out


def load_cell(phase: str, provider: str) -> list[dict]:
    """Load Q rows for one provider, or pooled across all 5 providers.
    Each row keeps its own (qid, provider, cluster_assignments)."""
    if provider == "pooled":
        out = []
        for prov in ALL_PROVIDERS:
            out.extend(load_cell(phase, prov))
        return out

    grades = graded_ids(phase, provider)
    cp = cluster_path(phase, provider)
    if cp is None:
        return []
    out = []
    with open(cp) as fh:
        for line in fh:
            r = json.loads(line)
            if r["id"] not in grades:
                continue
            ca = r.get("cluster_assignments")
            cs = r.get("cluster_sizes")
            if not ca or not cs:
                continue
            ca = np.asarray(ca, dtype=np.int32)
            n_samples = int(r.get("n_samples", len(ca)))
            if n_samples < max(N_VALUES):
                continue   # need at least 50 samples to do the full sweep
            # Empirical top-two margin (on full 50 samples) for the low-Δ split
            sizes = np.sort(np.asarray(cs))[::-1]
            top1 = sizes[0] / n_samples
            top2 = (sizes[1] / n_samples) if len(sizes) > 1 else 0.0
            out.append({
                "id":  f"{provider}__{r['id']}",   # disambiguate same qid across providers
                "ca":  ca,
                "Y":   grades[r["id"]],
                "delta_full": float(top1 - top2),
            })
    return out


# ---------------------------------------------------------------------------
# Confidence estimators on a (sub)sample of cluster IDs
# ---------------------------------------------------------------------------
def _modal_freq(ca: np.ndarray) -> float:
    if len(ca) == 0:
        return 0.0
    _, counts = np.unique(ca, return_counts=True)
    return float(counts.max() / len(ca))


def _sssc(ca: np.ndarray, R: int, rng: np.random.Generator) -> float:
    """Sem_2: average over R half-splits of agreement-with-modal-on-S."""
    n = len(ca)
    if n < 2:
        return _modal_freq(ca)
    s = n // 2
    confs = np.empty(R)
    for r in range(R):
        idx = rng.permutation(n)
        S = ca[idx[:s]]
        E = ca[idx[s:]]
        u, c = np.unique(S, return_counts=True)
        z_S = int(u[np.argmax(c)])
        confs[r] = (E == z_S).mean()
    return float(confs.mean())


def n_sweep_per_q(qs: list[dict], n_vals: list[int],
                  R: int = R_SPLITS, S: int = N_SUBSAMP,
                  seed: int = 7) -> dict[int, dict]:
    """Returns {n: {'c1': arr, 'c2': arr, 'y': arr, 'delta_full': arr}}."""
    rng = np.random.default_rng(seed)
    out = {n: {"c1": [], "c2": [], "y": [], "delta_full": []} for n in n_vals}
    for q in qs:
        ca = q["ca"]
        n_total = len(ca)
        for n in n_vals:
            c1_acc = 0.0
            c2_acc = 0.0
            for s in range(S):
                idx = rng.permutation(n_total)[:n]
                sub = ca[idx]
                c1_acc += _modal_freq(sub)
                c2_acc += _sssc(sub, R=R, rng=rng)
            out[n]["c1"].append(c1_acc / S)
            out[n]["c2"].append(c2_acc / S)
            out[n]["y"].append(q["Y"])
            out[n]["delta_full"].append(q["delta_full"])
    for n in n_vals:
        for k in out[n]:
            out[n][k] = np.asarray(out[n][k], dtype=float)
    return out


def ece(conf: np.ndarray, y: np.ndarray, n_bins: int = ECE_BINS_FULL) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    N = len(conf)
    e = 0.0
    for k in range(n_bins):
        mask = bin_idx == k
        m = int(mask.sum())
        if m:
            e += (m / N) * abs(np.mean(conf[mask]) - np.mean(y[mask]))
    return float(e)


def adaptive_bins(N: int) -> int:
    """Pick a bin count so every bin holds ≥ MIN_PER_BIN samples on average."""
    return max(3, min(ECE_BINS_FULL, N // MIN_PER_BIN))


# ---------------------------------------------------------------------------
# Run sweep + plot
# ---------------------------------------------------------------------------
print("Loading and sweeping (this takes ~1–2 min per dataset)...")
print("\nLow-margin threshold per dataset:  Δ_thr(n) = √(log K / n)\n")
results = {}
for ds_label, phase, prov in PANEL_SPECS:
    qs = load_cell(phase, prov)
    if not qs:
        print(f"[warn] no data for {ds_label}/{prov}")
        continue
    K = K_BY_DATASET.get(ds_label, np.nan)
    sqrt_logK = float(np.sqrt(np.log(max(K, 1.0001))))
    print(f"  {ds_label}/{prov}: N={len(qs)} questions, "
          f"K={K:.2f}, √log K = {sqrt_logK:.4f}")
    sweep = n_sweep_per_q(qs, N_VALUES)

    # ECE per n on (a) full set and (b) low-margin subset (n-specific threshold)
    rows_full, rows_lo = [], []
    delta_full_vec = sweep[N_VALUES[0]]["delta_full"]   # same Δ across n
    n_lo_at_max_n  = None
    print(f"      n   Δ_thr      n_q   ECE_1     ECE_2     ECE_1−ECE_2")
    for n in N_VALUES:
        d = sweep[n]
        rows_full.append({
            "n": n,
            "ece1": ece(d["c1"], d["y"], n_bins=ECE_BINS_FULL),
            "ece2": ece(d["c2"], d["y"], n_bins=ECE_BINS_FULL),
        })
        thr_n = low_margin_threshold(K, n)
        lo_mask_n = delta_full_vec < thr_n
        n_lo_n    = int(lo_mask_n.sum())
        if n == N_VALUES[-1]:
            n_lo_at_max_n = n_lo_n
        if n_lo_n >= MIN_PER_BIN:
            bins_lo_n = adaptive_bins(n_lo_n)
            e1 = ece(d["c1"][lo_mask_n], d["y"][lo_mask_n], n_bins=bins_lo_n)
            e2 = ece(d["c2"][lo_mask_n], d["y"][lo_mask_n], n_bins=bins_lo_n)
            rows_lo.append({"n": n, "ece1": e1, "ece2": e2,
                            "n_lo": n_lo_n, "thr": thr_n})
            print(f"   {n:>3d}   {thr_n:.4f}   {n_lo_n:>5d}   "
                  f"{e1:.4f}    {e2:.4f}    {e1 - e2:+.4f}")
        else:
            print(f"   {n:>3d}   {thr_n:.4f}   {n_lo_n:>5d}   "
                  f"(<{MIN_PER_BIN}, skipped)")

    results[ds_label] = {
        "rows_full": rows_full,
        "rows_lo":   rows_lo,
        "n_lo":      n_lo_at_max_n if n_lo_at_max_n is not None else 0,
        "n_total":   len(qs),
        "K":         K,
        "provider":  prov,
    }
    print()


# ---------------------------------------------------------------------------
# Figure 1 of 2: ECE vs n on the full set (linear–linear)
# ---------------------------------------------------------------------------
fig_full, axes_full = plt.subplots(
    1, len(results), figsize=(13.5, 4.0),
    gridspec_kw={"wspace": 0.25, "left": 0.07, "right": 0.985,
                  "top": 0.88, "bottom": 0.16},
)
if len(results) == 1:
    axes_full = [axes_full]

for j, (ds_label, R) in enumerate(results.items()):
    ax = axes_full[j]
    ns   = [r["n"] for r in R["rows_full"]]
    e1   = [r["ece1"] for r in R["rows_full"]]
    e2   = [r["ece2"] for r in R["rows_full"]]
    ax.plot(ns, e1, color=C_SEM1, lw=2.0, marker="o", markersize=6,
            markerfacecolor=C_SEM1, markeredgecolor="black",
            markeredgewidth=0.5, label=r"Sem$_1$")
    ax.plot(ns, e2, color=C_SEM2, lw=2.0, marker="o", markersize=6,
            markerfacecolor=C_SEM2, markeredgecolor="black",
            markeredgewidth=0.5, label=r"Sem$_2$")
    ax.set_xticks(ns)
    ax.set_xlabel(r"$n$ (samples per question)")
    if j == 0:
        ax.set_ylabel("ECE")
    ax.set_title(rf"{ds_label}  ($N{{=}}{R['n_total']:,d}$)",
                 fontsize=15.5, pad=6)
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.35)
    if j == 0:
        ax.legend(frameon=False, fontsize=13)

fig_full.savefig(OUT / "n_sweep_full.pdf")
fig_full.savefig(OUT / "n_sweep_full.png", dpi=300)
plt.close(fig_full)

# ---------------------------------------------------------------------------
# Figure 2 of 2: gap vs n on the low-Δ subset, log–log + 1/√n reference
# ---------------------------------------------------------------------------
print("\nFitted log-log slopes of (ECE_1 − ECE_2) vs n on the low-Δ subset:")
fig_lo, axes_lo = plt.subplots(
    1, len(results), figsize=(13.5, 4.0),
    gridspec_kw={"wspace": 0.25, "left": 0.07, "right": 0.985,
                  "top": 0.88, "bottom": 0.16},
)
if len(results) == 1:
    axes_lo = [axes_lo]

for j, (ds_label, R) in enumerate(results.items()):
    ax = axes_lo[j]
    if not R["rows_lo"]:
        ax.text(0.5, 0.5, f"low-Δ n={R['n_lo']}\n(<30, skipped)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="#888")
        continue

    ns   = np.array([r["n"] for r in R["rows_lo"]], dtype=float)
    gaps = np.array([r["ece1"] - r["ece2"] for r in R["rows_lo"]],
                    dtype=float)
    pos  = gaps > 0  # only positive gaps log-loggable

    slope = np.nan
    if pos.sum() >= 3:
        slope, intercept = np.polyfit(np.log(ns[pos]), np.log(gaps[pos]), 1)
        n_fit = np.linspace(ns.min(), ns.max(), 50)
        ax.plot(n_fit, np.exp(intercept) * n_fit ** slope,
                color=C_GAP, lw=1.0, ls="--", alpha=0.7,
                label="fit")

    ref = gaps[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax.plot(ns, ref, color=C_REF, lw=1.4, ls=":",
            label=r"$\propto 1/\sqrt{n}$ reference")

    ax.plot(ns, gaps, color="black", lw=1.8, marker="o", markersize=6,
            markerfacecolor=C_SEM2, markeredgecolor="black",
            markeredgewidth=0.6, zorder=5,
            label=r"empirical gap (ECE$_1$ − ECE$_2$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"$n$ (log scale)")
    if j == 0:
        ax.set_ylabel(r"ECE$_1$ $-$ ECE$_2$  (log scale)")
    ax.set_title(rf"{ds_label}  ($n_q{{=}}{R['n_lo']:,d}$)",
                 fontsize=15, pad=6)
    ax.grid(which="both", linestyle=":", linewidth=0.4, alpha=0.35)
    # Per-panel slope annotation (always visible).  First panel keeps the
    # full legend; the other panels just get a slope tag in the lower-left.
    if j == 0:
        ax.legend(frameon=False, fontsize=9, loc="lower left")

    print(f"  {ds_label}: slope = {slope:.3f}  (theory: −0.5)")

fig_lo.savefig(OUT / "n_sweep_lowdelta.pdf")
fig_lo.savefig(OUT / "n_sweep_lowdelta.png", dpi=300)
plt.close(fig_lo)


# ---------------------------------------------------------------------------
# Combined figure2.pdf  (2 rows × 3 datasets) for the paper
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(
    2, len(results), figsize=(13.5, 8.4),
    gridspec_kw={"wspace": 0.25, "hspace": 0.42,
                 "left": 0.07, "right": 0.985,
                 "top": 0.93, "bottom": 0.08},
)
if len(results) == 1:
    axes2 = axes2.reshape(2, 1)

# Top row (a): ECE vs n on full population
for j, (ds_label, R) in enumerate(results.items()):
    ax  = axes2[0, j]
    ns  = [r["n"] for r in R["rows_full"]]
    e1  = [r["ece1"] for r in R["rows_full"]]
    e2  = [r["ece2"] for r in R["rows_full"]]
    ax.plot(ns, e1, color=C_SEM1, lw=2.0, marker="o", markersize=6,
            markerfacecolor=C_SEM1, markeredgecolor="black",
            markeredgewidth=0.5, label=r"Sem$_1$")
    ax.plot(ns, e2, color=C_SEM2, lw=2.0, marker="o", markersize=6,
            markerfacecolor=C_SEM2, markeredgecolor="black",
            markeredgewidth=0.5, label=r"Sem$_2$")
    ax.set_xticks(ns)
    ax.set_xlabel(r"$n$ (samples per question)")
    if j == 0:
        ax.set_ylabel("ECE  (full population)")
    ax.set_title(rf"(a) {ds_label}  ($N{{=}}{R['n_total']:,d}$)",
                 fontsize=15.5, pad=6)
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.35)
    if j == 0:
        ax.legend(frameon=False, fontsize=12)

# Bottom row (b): gap on low-margin subset, log-log
for j, (ds_label, R) in enumerate(results.items()):
    ax = axes2[1, j]
    if not R["rows_lo"]:
        ax.text(0.5, 0.5, f"low-margin n={R['n_lo']}\n(<{MIN_PER_BIN}, skipped)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=14, color="#888")
        continue

    ns   = np.array([r["n"]    for r in R["rows_lo"]], dtype=float)
    gaps = np.array([r["ece1"] - r["ece2"] for r in R["rows_lo"]], dtype=float)
    pos  = gaps > 0

    slope = np.nan
    if pos.sum() >= 3:
        slope, intercept = np.polyfit(np.log(ns[pos]), np.log(gaps[pos]), 1)
        n_fit = np.linspace(ns.min(), ns.max(), 50)
        ax.plot(n_fit, np.exp(intercept) * n_fit ** slope,
                color=C_GAP, lw=1.0, ls="--", alpha=0.7,
                label="fit")
    ref = gaps[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax.plot(ns, ref, color=C_REF, lw=1.4, ls=":",
            label=r"$\propto 1/\sqrt{n}$ reference")
    ax.plot(ns, gaps, color="black", lw=1.8, marker="o", markersize=6,
            markerfacecolor=C_SEM2, markeredgecolor="black",
            markeredgewidth=0.6, zorder=5,
            label=r"empirical gap (ECE$_1$ − ECE$_2$)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(r"$n$ (log scale)")
    if j == 0:
        ax.set_ylabel(r"ECE$_1$ $-$ ECE$_2$  (low-margin subset)")
    ax.set_title(rf"(b) {ds_label}  ($n_q{{=}}{R['n_lo']:,d}$)",
                 fontsize=15, pad=6)
    ax.grid(which="both", linestyle=":", linewidth=0.4, alpha=0.35)
    if j == 0:
        ax.legend(frameon=False, fontsize=9, loc="lower left")

fig2.savefig(ROOT / "figure2.pdf")
fig2.savefig(ROOT / "figure2.png", dpi=300)
plt.close(fig2)

print(f"\nSaved 3 figures:")
print(f"  {OUT / 'n_sweep_full.pdf'}      +  .png")
print(f"  {OUT / 'n_sweep_lowdelta.pdf'}  +  .png")
print(f"  {ROOT / 'figure2.pdf'}                 +  .png   (combined 2×3 for paper)")
