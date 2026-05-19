"""figure3.py — ECE stratified by per-question margin Δ_q.

Pooled across all 5 providers per dataset.  Two theory-derived boundaries
partition the question population into three regimes (matching Figure 1b):

      JDR              :  Δ_q  <  2·tilde_lambda_star / sqrt(n)  =  0.612 / sqrt(n)
      low-margin (non-JDR) :  0.612/√n  ≤  Δ_q  <  sqrt(log K / n)
      large-margin     :  Δ_q  ≥  sqrt(log K / n)

Sem₁-ECE (orange) and Sem₂-ECE (blue) are computed in equal-width bins on
Δ_q ∈ [0, 1.0].  A two-colour (orange + blue) translucent histogram of the
empirical Δ_q distribution sits at the bottom of each panel; a right-side
density y-axis on the rightmost panel.  Outputs:  figure3.pdf, figure3.png.

Reads the SAME per-question data sources as make_ece_figure.py
(*_clustered.jsonl, *_graded.jsonl, *_confidence.jsonl).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib import rcParams
from scipy.stats import norm


# ---------- Style -----------------------------------------------------------
rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.titlesize":   17,
    "axes.labelsize":   16,
    "legend.fontsize":  13,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":   1.0,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.10,
})


# ---------- Constants -------------------------------------------------------
TL_STAR  = 0.306                   # tilde-lambda star (theoretical)
N        = 50                      # samples per question

# Pooled mean K (concat across all 5 providers, per dataset).
K_BY_DATASET = {
    "SimpleQA": 7.48,
    "HLE":      7.49,
    "PopQA":    6.05,
}

# X-axis: raw Δ_q, full range
X_MAX  = 1.0
N_BINS = 12      # for ECE binning
N_DBINS = 25     # for density bars
MIN_PER_BIN = 30
BOTTOM_FRAC = 0.24

# Colours per spec
C_SEM1   = "#D85F37"          # orange  (Sem₁-ECE)
C_SEM2   = "#3B7BB8"          # blue    (Sem₂-ECE)
C_JDR    = "#C0392B"          # red dashed (universal JDR boundary)
C_LARGE  = "#7F4F1F"          # brown solid (low/large boundary)
C_REGION_JDR   = "#B8E0C8"    # light green
C_REGION_MID   = "#FAE3A0"    # light yellow
C_REGION_LARGE = "#D8D8D8"    # light gray
REG_ALPHA = 0.18

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
PROVIDERS = ["openai", "anthropic", "gemini", "xai", "mistral"]
DATASETS = [
    # (display label, phase, clustered-file ds suffix)
    ("SimpleQA", "phase1", "simpleqa"),
    ("HLE",      "phase2", "simpleqa"),    # legacy filename
    ("PopQA",    "phase4", "popqa"),
]


# ---------- Data loading ----------------------------------------------------
def load_pooled(phase: str, ds_suffix: str) -> np.ndarray:
    """Returns (M, 4) array with [Δ_q, conf_naive, conf_sssc, Y]
    pooled across all 5 providers (each (qid, prov) pair = one row)."""
    rows = []
    for prov in PROVIDERS:
        cluster_path = DATA_ROOT / phase / "clustering" / f"{prov}_{ds_suffix}_clustered.jsonl"
        if not cluster_path.exists():
            continue
        deltas_per_qid: dict[str, float] = {}
        with open(cluster_path) as fh:
            for line in fh:
                r = json.loads(line)
                cs = r.get("cluster_sizes")
                if not cs:
                    continue
                tot = float(sum(cs))
                if tot <= 0:
                    continue
                cs_sorted = sorted(cs, reverse=True)
                top1 = cs_sorted[0] / tot
                top2 = cs_sorted[1] / tot if len(cs_sorted) > 1 else 0.0
                deltas_per_qid[r["id"]] = max(0.0, top1 - top2)

        grades: dict[str, int] = {}
        for gp in (DATA_ROOT / phase / "grading").glob(f"{prov}_*_graded.jsonl"):
            with open(gp) as fh:
                for line in fh:
                    g = json.loads(line)
                    if g.get("Y") is not None:
                        grades[g["id"]] = int(g["Y"])
            break

        conf_path = DATA_ROOT / phase / "confidence" / f"{prov}_confidence.jsonl"
        if not conf_path.exists():
            continue
        with open(conf_path) as fh:
            for line in fh:
                r = json.loads(line)
                qid = r["id"]
                if qid not in grades or qid not in deltas_per_qid:
                    continue
                rows.append((
                    float(deltas_per_qid[qid]),
                    float(r.get("conf_naive", 0.0) or 0.0),
                    float(r.get("conf_sssc",  0.0) or 0.0),
                    int(grades[qid]),
                ))
    return np.asarray(rows) if rows else np.empty((0, 4))


def bin_ece(delta: np.ndarray, conf: np.ndarray, y: np.ndarray,
             n_bins: int = N_BINS, x_max: float = X_MAX,
             min_count: int = MIN_PER_BIN):
    """Per-bin |mean(conf) - mean(y)| on equal-width Δ bins."""
    edges = np.linspace(0.0, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(delta, edges[1:-1]), 0, n_bins - 1)
    valid_centers, valid_eces, counts = [], [], []
    for k in range(n_bins):
        m = bin_idx == k
        nm = int(m.sum())
        if nm < min_count:
            continue
        ece_k = float(abs(conf[m].mean() - y[m].mean()))
        valid_centers.append(centers[k])
        valid_eces.append(ece_k)
        counts.append(nm)
    return np.asarray(valid_centers), np.asarray(valid_eces), np.asarray(counts)


def overlay_density(ax, values, ymax, color, alpha=0.42, n_bins=N_DBINS,
                    bottom_frac=BOTTOM_FRAC, offset=0.0, width_scale=0.46,
                    x_max=X_MAX):
    """Bottom-strip density bars."""
    edges   = np.linspace(0.0, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw      = (edges[1] - edges[0]) * width_scale
    h, _    = np.histogram(values, bins=edges)
    if h.max() == 0:
        return
    f = h / h.max()
    ax.bar(centers + offset, f * ymax * bottom_frac, width=bw,
           color=color, alpha=alpha, edgecolor="none", zorder=0)


def gap_empirical(delta, c1, c2, y, boundary, half_width):
    """ECE_1 - ECE_2 in a thin band  |Δ_q - boundary| < half_width."""
    mask = np.abs(delta - boundary) < half_width
    n_in = int(mask.sum())
    if n_in == 0:
        return float("nan"), float("nan"), float("nan"), 0
    e1 = float(abs(c1[mask].mean() - y[mask].mean()))
    e2 = float(abs(c2[mask].mean() - y[mask].mean()))
    return e1, e2, e1 - e2, n_in


def gap_theory(tilde_m_star, n):
    """Corollary 3.3 leading-order universal form (p_q = 1):
        gap_theory = phi(m̃*) / sqrt(n)."""
    return float(norm.pdf(tilde_m_star) / np.sqrt(n))


def add_density_yaxis(ax, bottom_frac=BOTTOM_FRAC, label="density",
                      show_label=True):
    ax_d = ax.twinx()
    ax_d.set_ylim(0.0, 1.0 / bottom_frac)
    ax_d.set_yticks([0.0, 0.5, 1.0])
    ax_d.set_yticklabels(["0", "", "1"])
    ax_d.spines["right"].set_bounds(0, 1)
    ax_d.spines["top"].set_visible(False)
    ax_d.tick_params(axis="y", which="major", length=3, pad=2,
                     labelsize=11)
    if show_label:
        ax_d.set_ylabel(label, rotation=270, labelpad=14, va="bottom",
                        fontsize=13, y=bottom_frac / 2)
    return ax_d


# ---------- Plot ------------------------------------------------------------
def main():
    sqrt_n = float(np.sqrt(N))
    delta_jdr = (2.0 * TL_STAR) / sqrt_n     # 0.612 / √n

    print("=" * 70)
    print(f"Boundaries (n = {N}, x-axis = Δ_q):")
    print(f"  JDR boundary  Δ = 2·tl*/√n  = 0.612/√{N}  =  {delta_jdr:.4f}  (universal)")
    for ds, _, _ in DATASETS:
        K = K_BY_DATASET[ds]
        d_lg = float(np.sqrt(np.log(K) / N))
        print(f"  {ds:<10}  K = {K:.2f}   √(log K / n)  =  {d_lg:.4f}")
    print("=" * 70)
    print()

    gap_th_jdr = gap_theory(2.0 * TL_STAR, N)
    print(f"Theory gaps  (gap_theory = phi(m̃*) / sqrt(n), leading-order universal form):")
    print(f"  JDR  m̃* = 0.612 → phi/√{N} = {gap_th_jdr:.4f}  (universal)")
    for ds in K_BY_DATASET:
        tm_star = float(np.sqrt(np.log(K_BY_DATASET[ds])))
        g       = gap_theory(tm_star, N)
        print(f"  {ds:<10}  m̃* = √log K = {tm_star:.4f}   gap_th = {g:.4f}")
    print()

    fig, axes = plt.subplots(
        1, 3, figsize=(14.0, 4.8),
        gridspec_kw={"wspace": 0.22, "left": 0.06, "right": 0.985,
                      "top": 0.78, "bottom": 0.15},
    )

    for j, (ds, phase, ds_suffix) in enumerate(DATASETS):
        ax = axes[j]
        data = load_pooled(phase, ds_suffix)
        if len(data) == 0:
            print(f"WARN: no data for {ds}")
            continue

        delta = data[:, 0]
        c1    = data[:, 1]
        c2    = data[:, 2]
        y     = data[:, 3]

        K = K_BY_DATASET[ds]
        d_large = float(np.sqrt(np.log(K) / N))

        n_total  = len(delta)
        frac_jdr = float((delta < delta_jdr).mean())
        frac_mid = float(((delta >= delta_jdr) & (delta < d_large)).mean())
        frac_lg  = float((delta >= d_large).mean())
        print(f"  {ds}  (N = {n_total:,}):  "
              f"JDR {frac_jdr*100:5.1f}%  |  "
              f"low-mid {frac_mid*100:5.1f}%  |  "
              f"large {frac_lg*100:5.1f}%")

        # ---- Theory-vs-experiment comparison at the two boundaries ----
        # Window: ±10% of the boundary value (widen if too sparse).
        def widen(boundary, frac=0.10, min_count=50):
            hw = boundary * frac
            for _ in range(6):
                _, _, _, n_in = gap_empirical(delta, c1, c2, y, boundary, hw)
                if n_in >= min_count:
                    break
                hw *= 1.5
            return hw

        hw_jdr = widen(delta_jdr)
        hw_lg  = widen(d_large)
        e1_j, e2_j, gap_emp_j, n_j = gap_empirical(delta, c1, c2, y, delta_jdr, hw_jdr)
        e1_l, e2_l, gap_emp_l, n_l = gap_empirical(delta, c1, c2, y, d_large,   hw_lg)
        gap_th_j = gap_th_jdr
        gap_th_l = gap_theory(float(np.sqrt(np.log(K))), N)

        print(f"\n  --- {ds}: theory vs experiment at boundaries ---")
        print(f"    {'':32s}{'JDR':>14s}{'low/large':>15s}")
        print(f"    {'Δ_q at boundary':32s}{delta_jdr:>14.4f}{d_large:>15.4f}")
        print(f"    {'half-width window':32s}{hw_jdr:>14.4f}{hw_lg:>15.4f}")
        print(f"    {'n_q in band':32s}{n_j:>14d}{n_l:>15d}")
        print(f"    {'gap_empirical (ECE_1−ECE_2)':32s}{gap_emp_j:>14.4f}{gap_emp_l:>15.4f}")
        print(f"    {'gap_theory   (phi/√n)':32s}{gap_th_j:>14.4f}{gap_th_l:>15.4f}")
        print(f"    {'ratio (emp / theory)':32s}{gap_emp_j/gap_th_j:>14.2f}{gap_emp_l/gap_th_l:>15.2f}")
        print()

        # ---- ECE curves first (size y-axis) ----
        x1, e1, _ = bin_ece(delta, c1, y)
        x2, e2, _ = bin_ece(delta, c2, y)
        if len(e1) and len(e2):
            ymax = max(np.concatenate([e1, e2]).max() * 1.18 + 0.02, 0.10)
        else:
            ymax = 1.0
        ymax = min(1.0, ymax)
        ax.set_ylim(0.0, ymax)

        # ---- 3 background regimes (Figure-1b colour scheme) ----
        ax.axvspan(0.0,        delta_jdr, color=C_REGION_JDR,
                   alpha=REG_ALPHA, zorder=0)
        ax.axvspan(delta_jdr,  d_large,   color=C_REGION_MID,
                   alpha=REG_ALPHA, zorder=0)
        ax.axvspan(d_large,    X_MAX,     color=C_REGION_LARGE,
                   alpha=REG_ALPHA, zorder=0)

        # ---- two-colour density underlay (orange + blue, side-by-side,
        #      narrow bars with a clear gap between the two colours) ----
        bin_w = X_MAX / N_DBINS
        offs  = 0.25 * bin_w
        overlay_density(ax, delta, ymax=ymax, color=C_SEM1,
                        alpha=0.55, n_bins=N_DBINS, offset=-offs,
                        width_scale=0.40)
        overlay_density(ax, delta, ymax=ymax, color=C_SEM2,
                        alpha=0.55, n_bins=N_DBINS, offset=+offs,
                        width_scale=0.40)

        # ---- boundaries ----
        ax.axvline(delta_jdr, color=C_JDR,   lw=1.6, ls=(0, (5, 3)), zorder=3)
        ax.axvline(d_large,   color=C_LARGE, lw=1.7, ls="-",          zorder=3)

        # ---- ECE curves ----
        ax.plot(x1, e1, color=C_SEM1, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM1, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4)
        ax.plot(x2, e2, color=C_SEM2, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM2, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4)

        ax.set_xlim(0.0, X_MAX)
        ax.set_xlabel(r"per-question margin $\Delta_q$")
        if j == 0:
            ax.set_ylabel("ECE")
        ax.set_title(rf"{ds}  ($\sqrt{{\log K\,/\,n}}={d_large:.3f}$)", pad=6)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
        add_density_yaxis(ax, bottom_frac=BOTTOM_FRAC,
                          show_label=(j == len(DATASETS) - 1))

    # ----- Shared legend -----
    handles = [
        Line2D([], [], color=C_SEM1, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM1, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_1$-ECE"),
        Line2D([], [], color=C_SEM2, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM2, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_2$-ECE"),
        Line2D([], [], color=C_JDR, lw=1.7, ls=(0, (5, 3)),
               label=r"JDR boundary  $\Delta_q = 2\tilde\lambda^{\!\star}/\sqrt{n}$"),
        Line2D([], [], color=C_LARGE, lw=1.7, ls="-",
               label=r"low / large boundary  $\Delta_q = \sqrt{\log K / n}$"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False, fontsize=12.5,
               handlelength=2.4, handletextpad=0.6, columnspacing=2.0)

    out_pdf = ROOT / "figure3.pdf"
    out_png = ROOT / "figure3.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print()
    print(f"saved {out_pdf}, {out_png}")


if __name__ == "__main__":
    main()
