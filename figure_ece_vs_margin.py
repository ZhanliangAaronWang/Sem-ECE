"""figure_ece_vs_margin.py — per-(dataset, model) versions of figure3.

Same 3-regime / 2-boundary layout as figure3.py:
    JDR              :  Δ_q  <  0.612 / sqrt(n)        (universal)
    low-margin       :  0.612/√n  ≤  Δ_q  <  √(log K / n)
    large-margin     :  Δ_q  ≥  √(log K / n)

K is computed *per cell* (provider × dataset, or pooled across the 5 providers
for the pooled figure).  Theory comparison uses the leading-order universal
form  gap_theory = phi(m̃*) / sqrt(n).

Outputs in ./figures/ (replacing the legacy ece_vs_margin_*):
    ece_vs_margin_pooled.{pdf,png}        1×3, all 5 providers concatenated
    ece_vs_margin_{provider}.{pdf,png}    1×3, single provider
    ece_vs_margin_all.{pdf,png}           5×3, every (provider, dataset)
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
TL_STAR = 0.306
N       = 50

X_MAX        = 1.0
N_BINS       = 12
N_DBINS      = 25
MIN_PER_BIN  = 30
BOTTOM_FRAC  = 0.24

C_SEM1   = "#D85F37"
C_SEM2   = "#3B7BB8"
C_JDR    = "#C0392B"
C_LARGE  = "#7F4F1F"
C_REGION_JDR   = "#B8E0C8"
C_REGION_MID   = "#FAE3A0"
C_REGION_LARGE = "#D8D8D8"
REG_ALPHA = 0.18

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OUT_DIR = ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)

PROVIDERS = ["openai", "anthropic", "gemini", "xai", "mistral"]
PROV_LABEL = {
    "openai":    "OpenAI gpt-5.4",
    "anthropic": "Anthropic claude-opus-4-6",
    "gemini":    "Gemini 3.1-flash-lite",
    "xai":       "xAI grok-4.20",
    "mistral":   "Mistral large-latest",
}
DATASETS = [
    ("SimpleQA", "phase1", "simpleqa"),
    ("HLE",      "phase2", "simpleqa"),
    ("PopQA",    "phase4", "popqa"),
]


# ---------- Data loading ----------------------------------------------------
def _load_provider_rows(phase, ds_suffix, provider):
    """Returns list of (Δ_q, conf_naive, conf_sssc, Y, n_clusters_for_Q)."""
    out = []
    cluster_path = DATA_ROOT / phase / "clustering" / f"{provider}_{ds_suffix}_clustered.jsonl"
    if not cluster_path.exists():
        return out
    delta_per_qid: dict[str, float] = {}
    nclust_per_qid: dict[str, int] = {}
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
            delta_per_qid[r["id"]]  = max(0.0, top1 - top2)
            nclust_per_qid[r["id"]] = sum(1 for c in cs if c > 0)

    grades: dict[str, int] = {}
    for gp in (DATA_ROOT / phase / "grading").glob(f"{provider}_*_graded.jsonl"):
        with open(gp) as fh:
            for line in fh:
                g = json.loads(line)
                if g.get("Y") is not None:
                    grades[g["id"]] = int(g["Y"])
        break

    conf_path = DATA_ROOT / phase / "confidence" / f"{provider}_confidence.jsonl"
    if not conf_path.exists():
        return out
    with open(conf_path) as fh:
        for line in fh:
            r = json.loads(line)
            qid = r["id"]
            if qid not in grades or qid not in delta_per_qid:
                continue
            out.append((
                float(delta_per_qid[qid]),
                float(r.get("conf_naive", 0.0) or 0.0),
                float(r.get("conf_sssc",  0.0) or 0.0),
                int(grades[qid]),
                int(nclust_per_qid[qid]),
            ))
    return out


def load_cell(phase, ds_suffix, providers):
    """Returns (M, 5) array  [Δ_q, conf_naive, conf_sssc, Y, n_clusters]
    pooled across the requested providers (one row per (qid, prov))."""
    rows = []
    for prov in providers:
        rows.extend(_load_provider_rows(phase, ds_suffix, prov))
    return np.asarray(rows) if rows else np.empty((0, 5))


# ---------- Math helpers ----------------------------------------------------
def bin_ece(delta, conf, y, n_bins=N_BINS, x_max=X_MAX, min_count=MIN_PER_BIN):
    edges   = np.linspace(0.0, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(delta, edges[1:-1]), 0, n_bins - 1)
    xs, es, ns = [], [], []
    for k in range(n_bins):
        m  = bin_idx == k
        nm = int(m.sum())
        if nm < min_count:
            continue
        xs.append(centers[k])
        es.append(float(abs(conf[m].mean() - y[m].mean())))
        ns.append(nm)
    return np.asarray(xs), np.asarray(es), np.asarray(ns)


def overlay_density(ax, values, ymax, color, *, alpha=0.55, n_bins=N_DBINS,
                    bottom_frac=BOTTOM_FRAC, offset=0.0, width_scale=0.40,
                    x_max=X_MAX):
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
    mask = np.abs(delta - boundary) < half_width
    n_in = int(mask.sum())
    if n_in == 0:
        return float("nan"), 0
    e1 = float(abs(c1[mask].mean() - y[mask].mean()))
    e2 = float(abs(c2[mask].mean() - y[mask].mean()))
    return e1 - e2, n_in


def gap_theory(tilde_m_star, n):
    return float(norm.pdf(tilde_m_star) / np.sqrt(n))


def add_density_yaxis(ax, *, bottom_frac=BOTTOM_FRAC, label="density",
                      show_label=True):
    ax_d = ax.twinx()
    ax_d.set_ylim(0.0, 1.0 / bottom_frac)
    ax_d.set_yticks([0.0, 0.5, 1.0])
    ax_d.set_yticklabels(["0", "", "1"])
    ax_d.spines["right"].set_bounds(0, 1)
    ax_d.spines["top"].set_visible(False)
    ax_d.tick_params(axis="y", which="major", length=3, pad=2, labelsize=11)
    if show_label:
        ax_d.set_ylabel(label, rotation=270, labelpad=14, va="bottom",
                        fontsize=13, y=bottom_frac / 2)
    return ax_d


# ---------- Per-panel rendering ---------------------------------------------
def widen_band(boundary, delta, c1, c2, y, frac=0.10, min_count=50):
    hw = boundary * frac
    for _ in range(6):
        _, n_in = gap_empirical(delta, c1, c2, y, boundary, hw)
        if n_in >= min_count:
            break
        hw *= 1.5
    return hw


def render_panel(ax, data, K, *, ds_label, show_ylabel=True,
                 show_density_yaxis=True, sqrt_n=float(np.sqrt(N))):
    """Draw one ECE-vs-Δ panel.  Returns a dict of summary stats."""
    delta = data[:, 0]; c1 = data[:, 1]; c2 = data[:, 2]; y = data[:, 3]
    delta_jdr = (2.0 * TL_STAR) / sqrt_n
    d_large   = float(np.sqrt(np.log(max(K, 1.0001)) / N))

    # Top regimes
    f_j  = float((delta < delta_jdr).mean())
    f_m  = float(((delta >= delta_jdr) & (delta < d_large)).mean())
    f_l  = float((delta >= d_large).mean())

    # Theory at boundaries
    gap_th_jdr = gap_theory(2.0 * TL_STAR, N)
    gap_th_lg  = gap_theory(float(np.sqrt(np.log(max(K, 1.0001)))), N)

    # ECE curves
    x1, e1, _ = bin_ece(delta, c1, y)
    x2, e2, _ = bin_ece(delta, c2, y)
    if len(e1) and len(e2):
        ymax = max(np.concatenate([e1, e2]).max() * 1.18 + 0.02, 0.10)
    else:
        ymax = 1.0
    ymax = min(1.0, ymax)
    ax.set_ylim(0.0, ymax)

    # Backgrounds
    ax.axvspan(0.0,        delta_jdr, color=C_REGION_JDR,   alpha=REG_ALPHA, zorder=0)
    ax.axvspan(delta_jdr,  d_large,   color=C_REGION_MID,   alpha=REG_ALPHA, zorder=0)
    ax.axvspan(d_large,    X_MAX,     color=C_REGION_LARGE, alpha=REG_ALPHA, zorder=0)

    # Density bars
    bin_w = X_MAX / N_DBINS
    offs  = 0.25 * bin_w
    overlay_density(ax, delta, ymax=ymax, color=C_SEM1, offset=-offs)
    overlay_density(ax, delta, ymax=ymax, color=C_SEM2, offset=+offs)

    # Boundaries
    ax.axvline(delta_jdr, color=C_JDR,   lw=1.6, ls=(0, (5, 3)), zorder=3)
    ax.axvline(d_large,   color=C_LARGE, lw=1.7, ls="-",          zorder=3)

    # ECE curves
    ax.plot(x1, e1, color=C_SEM1, lw=2.0, marker="o", markersize=5.5,
            markerfacecolor=C_SEM1, markeredgecolor="black",
            markeredgewidth=0.5, zorder=4)
    ax.plot(x2, e2, color=C_SEM2, lw=2.0, marker="o", markersize=5.5,
            markerfacecolor=C_SEM2, markeredgecolor="black",
            markeredgewidth=0.5, zorder=4)

    ax.set_xlim(0.0, X_MAX)
    ax.set_xlabel(r"per-question margin $\Delta_q$")
    if show_ylabel:
        ax.set_ylabel("ECE")
    ax.set_title(rf"{ds_label}  ($K={K:.2f},\ \sqrt{{\log K/n}}={d_large:.3f}$)",
                 pad=6)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
    if show_density_yaxis:
        add_density_yaxis(ax, show_label=True)

    # Empirical comparison at boundaries
    hw_j = widen_band(delta_jdr, delta, c1, c2, y)
    hw_l = widen_band(d_large,   delta, c1, c2, y)
    g_emp_j, n_j = gap_empirical(delta, c1, c2, y, delta_jdr, hw_j)
    g_emp_l, n_l = gap_empirical(delta, c1, c2, y, d_large,   hw_l)

    return dict(
        n_total=len(delta), K=K,
        delta_jdr=delta_jdr, d_large=d_large,
        f_j=f_j, f_m=f_m, f_l=f_l,
        gap_emp_j=g_emp_j, gap_emp_l=g_emp_l,
        gap_th_j=gap_th_jdr, gap_th_l=gap_th_lg,
        n_j=n_j, n_l=n_l,
    )


def shared_legend(fig):
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


# ---------- High-level builders ---------------------------------------------
def build_1x3(providers, *, title_suffix, out_stem):
    """1 row × 3 datasets figure."""
    fig, axes = plt.subplots(
        1, 3, figsize=(14.0, 4.8),
        gridspec_kw={"wspace": 0.22, "left": 0.06, "right": 0.985,
                     "top": 0.78, "bottom": 0.15},
    )
    print("=" * 80)
    print(f"  Figure: {out_stem}   (providers: {','.join(providers)})")
    print("=" * 80)
    for j, (ds_label, phase, ds_suffix) in enumerate(DATASETS):
        ax   = axes[j]
        data = load_cell(phase, ds_suffix, providers)
        if len(data) == 0:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        K = float(np.nanmean(data[:, 4]))
        info = render_panel(ax, data, K, ds_label=ds_label,
                            show_ylabel=(j == 0),
                            show_density_yaxis=(j == 2))
        print(f"  [{ds_label:<8}]  N={info['n_total']:>5,d}  K={K:.2f}  "
              f"Δ_lg={info['d_large']:.3f}   "
              f"frac (JDR/mid/lg) = {info['f_j']*100:4.1f}/"
              f"{info['f_m']*100:4.1f}/{info['f_l']*100:4.1f}%")
        print(f"     JDR    : Δ={info['delta_jdr']:.4f}  "
              f"emp={info['gap_emp_j']:+.4f}  th={info['gap_th_j']:+.4f}  "
              f"n={info['n_j']}")
        print(f"     low/lg : Δ={info['d_large']:.4f}  "
              f"emp={info['gap_emp_l']:+.4f}  th={info['gap_th_l']:+.4f}  "
              f"n={info['n_l']}")

    shared_legend(fig)
    if title_suffix:
        fig.suptitle(title_suffix, y=1.06, fontsize=15)

    out_pdf = OUT_DIR / f"{out_stem}.pdf"
    out_png = OUT_DIR / f"{out_stem}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  saved {out_pdf.name}  /  {out_png.name}\n")


def build_5x3_all():
    """5 providers × 3 datasets grid."""
    n_rows = len(PROVIDERS)
    n_cols = len(DATASETS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(14.0, 4.5 * n_rows),
        gridspec_kw={"wspace": 0.22, "hspace": 0.55,
                     "left": 0.07, "right": 0.985,
                     "top": 0.96, "bottom": 0.05},
    )
    print("=" * 80)
    print(f"  Figure: ece_vs_margin_all   (5 × 3 grid)")
    print("=" * 80)
    for i, prov in enumerate(PROVIDERS):
        for j, (ds_label, phase, ds_suffix) in enumerate(DATASETS):
            ax   = axes[i, j]
            data = load_cell(phase, ds_suffix, [prov])
            if len(data) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            K = float(np.nanmean(data[:, 4]))
            info = render_panel(ax, data, K,
                                ds_label=f"{PROV_LABEL[prov]}  ·  {ds_label}",
                                show_ylabel=(j == 0),
                                show_density_yaxis=(j == n_cols - 1))
            print(f"  [{prov:<10} | {ds_label:<8}]  N={info['n_total']:>5,d}  "
                  f"K={K:.2f}  emp_J={info['gap_emp_j']:+.4f}  "
                  f"emp_L={info['gap_emp_l']:+.4f}  "
                  f"th_J={info['gap_th_j']:+.4f}  th_L={info['gap_th_l']:+.4f}")

    shared_legend(fig)
    out_pdf = OUT_DIR / "ece_vs_margin_all.pdf"
    out_png = OUT_DIR / "ece_vs_margin_all.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  saved {out_pdf.name}  /  {out_png.name}\n")


# ---------- Main ------------------------------------------------------------
def main():
    # Pooled (all 5 providers)
    build_1x3(PROVIDERS, title_suffix=None, out_stem="ece_vs_margin_pooled")

    # Per provider
    for prov in PROVIDERS:
        build_1x3([prov], title_suffix=PROV_LABEL[prov],
                  out_stem=f"ece_vs_margin_{prov}")

    # Big grid
    build_5x3_all()


if __name__ == "__main__":
    main()
