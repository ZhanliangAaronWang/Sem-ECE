"""figure_per_provider.py — sample-level calibration figures (one row per dataset, one figure per provider).

For each of the 5 providers (OpenAI, Anthropic, Gemini, xAI, Mistral) we
produce two separate figures:

  figures/ece_vs_margin_<provider>.{pdf,png}
      ECE stratified by per-question margin Δ, two methods (Sem_1 vs Sem_2),
      with a two-colour Δ density underlay at the bottom of each panel.

  figures/reliability_<provider>.{pdf,png}
      Sample-level reliability diagram, three methods (Sem_1, Sem_2,
      Verbalized) plus a three-colour predicted-confidence density underlay.

Each figure is 1 row × 3 columns (one column per dataset: SimpleQA, HLE, PopQA).
Total: 5 × 2 = 10 figure files.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"
OUT  = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.10,
})

# Wong / Okabe-Ito colours
C_SEM1 = "#D55E00"   # vermillion
C_SEM2 = "#0072B2"   # blue
C_VERB   = "#009E73"   # bluish green
C_GOLD   = "#E0A100"
C_DIAG   = "#999999"

DATASETS = [
    ("SimpleQA", "phase1"),
    ("HLE",      "phase2"),
    ("PopQA",    "phase4"),
]
PROVIDERS = ["openai", "anthropic", "gemini", "xai", "mistral"]
PROVIDER_DISPLAY = {
    "openai":    "OpenAI",
    "anthropic": "Anthropic",
    "gemini":    "Gemini",
    "xai":       "xAI",
    "mistral":   "Mistral",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def common_qids(phase: str) -> set[str]:
    """Intersection of question IDs across all providers in this phase.
    Ensures every provider contributes the exact same question set, so
    comparisons across methods/providers are like-for-like.
    """
    qid_sets = []
    for prov in PROVIDERS:
        conf_path  = DATA_ROOT / phase / "confidence" / f"{prov}_confidence.jsonl"
        grade_dir  = DATA_ROOT / phase / "grading"
        grade_paths = list(grade_dir.glob(f"{prov}_*_graded.jsonl"))
        if not conf_path.exists() or not grade_paths:
            continue
        graded_ids = set()
        with open(grade_paths[0]) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("Y") is not None:
                    graded_ids.add(r["id"])
        ids = set()
        with open(conf_path) as fh:
            for line in fh:
                r = json.loads(line)
                if r["id"] in graded_ids and r.get("empirical_margin") is not None:
                    ids.add(r["id"])
        qid_sets.append(ids)
    if not qid_sets:
        return set()
    return set.intersection(*qid_sets)


def load_cell(phase: str, provider: str, qid_filter: set[str] | None = None):
    conf_path = DATA_ROOT / phase / "confidence" / f"{provider}_confidence.jsonl"
    if not conf_path.exists():
        return None
    grade_dir   = DATA_ROOT / phase / "grading"
    grade_paths = list(grade_dir.glob(f"{provider}_*_graded.jsonl"))
    if not grade_paths:
        return None
    grades = {}
    with open(grade_paths[0]) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("Y") is not None:
                grades[r["id"]] = int(r["Y"])
    deltas, c1s, c2s, cvs, ys = [], [], [], [], []
    with open(conf_path) as fh:
        for line in fh:
            r = json.loads(line)
            qid = r["id"]
            if qid not in grades:
                continue
            if qid_filter is not None and qid not in qid_filter:
                continue
            d = r.get("empirical_margin")
            if d is None:
                continue
            cv = r.get("conf_verbalized")
            cv = 1.0 if cv is None else float(cv)
            deltas.append(float(d))
            c1s.append(float(r["conf_naive"]))
            c2s.append(float(r["conf_sssc"]))
            cvs.append(cv)
            ys.append(grades[qid])
    return (np.array(deltas), np.array(c1s), np.array(c2s),
            np.array(cvs),    np.array(ys, dtype=float))


def calibration_curve_by_delta(delta, conf, y, n_bins=12, min_count=30):
    edges = np.unique(np.quantile(delta, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return np.array([]), np.array([]), np.array([])
    bin_idx = np.clip(np.digitize(delta, edges[1:-1]), 0, len(edges) - 2)
    centers, errs, counts = [], [], []
    for k in range(len(edges) - 1):
        mask = bin_idx == k
        m = int(mask.sum())
        if m < min_count:
            continue
        centers.append(float(np.mean(delta[mask])))
        errs.append(float(abs(np.mean(conf[mask]) - np.mean(y[mask]))))
        counts.append(m)
    return np.array(centers), np.array(errs), np.array(counts, dtype=int)


def reliability_curve(conf, y, n_bins=10, min_count=30):
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    centers, accs, counts = [], [], []
    for k in range(n_bins):
        mask = bin_idx == k
        m = int(mask.sum())
        if m < min_count:
            continue
        centers.append(float(np.mean(conf[mask])))
        accs.append(float(np.mean(y[mask])))
        counts.append(m)
    return np.array(centers), np.array(accs), np.array(counts, dtype=int)


def overall_ece(conf, y, n_bins=10):
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    N = len(conf)
    e = 0.0
    for k in range(n_bins):
        mask = bin_idx == k
        m = int(mask.sum())
        if m == 0:
            continue
        e += (m / N) * abs(np.mean(conf[mask]) - np.mean(y[mask]))
    return e


def overlay_density(ax, values, ymax: float, color: str = "#888888",
                    alpha: float = 0.45, n_bins: int = 30,
                    bottom_frac: float = 0.24, offset: float = 0.0,
                    width_scale: float = 1.0):
    """In-axis density: bars rise from the bottom, capped at
    `bottom_frac` of `ymax`, drawn behind the main curves."""
    edges   = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bw      = (edges[1] - edges[0]) * width_scale
    h, _    = np.histogram(values, bins=edges)
    if h.max() == 0:
        return
    f = h / h.max()
    ax.bar(centers + offset, f * ymax * bottom_frac, width=bw,
           color=color, alpha=alpha, edgecolor="none", zorder=0)


def add_density_yaxis(ax, bottom_frac: float = 0.24,
                      label: str = "density",
                      show_label: bool = True):
    """Add a right twin y-axis showing the normalized density scale [0, 1]
    for the bottom-strip overlay produced by `overlay_density`. The twin's
    data-coord range is [0, 1/bottom_frac] so that density value 1.0 lines
    up exactly with the top of the density bars (= bottom_frac of the panel
    height in main coords). Ticks are clipped to [0, 1] so they only appear
    inside the visible density region."""
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


# ---------------------------------------------------------------------------
# Aggregate per dataset
# ---------------------------------------------------------------------------
def build_panels_for_provider(provider: str) -> list[dict]:
    panels = []
    for ds_label, phase in DATASETS:
        cell = load_cell(phase, provider)
        if cell is None:
            print(f"[warn] no data for {ds_label}/{provider}")
            continue
        d, c1, c2, cv, y = cell
        panels.append({
            "label": ds_label,
            "delta": d, "c1": c1, "c2": c2, "cv": cv, "y": y,
            "N":     len(d),
        })
    return panels


# ---------------------------------------------------------------------------
# Per-provider curve helpers (so we can macro-average across providers)
# ---------------------------------------------------------------------------
def per_provider_ece_curves(panel: dict, conf_key: str,
                             n_bins: int = 12, min_count: int = 30):
    """For each provider, compute ECE on shared quantile bins of pooled Δ.
    Returns dict prov -> (x_centers, ece_vals) and the shared bin edges.
    """
    pooled = np.concatenate([d["delta"] for d in panel["per_prov"].values()])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        return {}, edges
    out = {}
    for prov, d in panel["per_prov"].items():
        bin_idx = np.clip(np.digitize(d["delta"], edges[1:-1]),
                          0, len(edges) - 2)
        xs, es = [], []
        for k in range(len(edges) - 1):
            mask = bin_idx == k
            if int(mask.sum()) < min_count:
                xs.append(np.nan); es.append(np.nan); continue
            xs.append(float(np.mean(d["delta"][mask])))
            es.append(float(abs(np.mean(d[conf_key][mask])
                                - np.mean(d["y"][mask]))))
        out[prov] = (np.array(xs), np.array(es))
    return out, edges


def per_provider_reliability(panel: dict, conf_key: str,
                              n_bins: int = 10, min_count: int = 30):
    """Per-provider reliability curve on fixed equal-width confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = {}
    for prov, d in panel["per_prov"].items():
        bin_idx = np.clip(np.digitize(d[conf_key], edges[1:-1]),
                          0, n_bins - 1)
        xs, ys = [], []
        for k in range(n_bins):
            mask = bin_idx == k
            if int(mask.sum()) < min_count:
                xs.append(np.nan); ys.append(np.nan); continue
            xs.append(float(np.mean(d[conf_key][mask])))
            ys.append(float(np.mean(d["y"][mask])))
        out[prov] = (np.array(xs), np.array(ys))
    return out, edges


def macro_curve(curves: dict[str, tuple[np.ndarray, np.ndarray]]):
    """Stack per-provider y-values into a 2D array and return
    (x, mean, lo, hi) using nan-aware aggregation across providers."""
    if not curves:
        return np.array([]), np.array([]), np.array([]), np.array([])
    Y = np.array([y for _, y in curves.values()])           # (P, B)
    X = np.array([x for x, _ in curves.values()])           # (P, B)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(Y, axis=0)
        lo   = np.nanmin(Y, axis=0)
        hi   = np.nanmax(Y, axis=0)
        x_c  = np.nanmean(X, axis=0)
    return x_c, mean, lo, hi


def macro_overall_ece(panel: dict, conf_key: str, n_bins: int = 10):
    """Mean of per-provider overall ECE values."""
    eces = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for prov, d in panel["per_prov"].items():
        bin_idx = np.clip(np.digitize(d[conf_key], edges[1:-1]),
                          0, n_bins - 1)
        N = len(d[conf_key])
        e = 0.0
        for k in range(n_bins):
            m = int((bin_idx == k).sum())
            if m == 0: continue
            mask = bin_idx == k
            e += (m / N) * abs(np.mean(d[conf_key][mask])
                                - np.mean(d["y"][mask]))
        eces.append(e)
    return float(np.mean(eces)) if eces else float("nan")


n_ref     = 50
# Per-dataset boundary  Δ_thr = √(log K_pooled / n)  using the pooled mean K
# across all 5 providers: SimpleQA 7.48, HLE 7.49, PopQA 6.05.
# Old default was Δ_thr = 1/√n ≈ 0.141 which corresponds to K = e ≈ 2.72.
DELTA_THR_BY_DATASET = {
    "SimpleQA": np.sqrt(np.log(7.48) / n_ref),   # ≈ 0.201
    "HLE":      np.sqrt(np.log(7.49) / n_ref),   # ≈ 0.201
    "PopQA":    np.sqrt(np.log(6.05) / n_ref),   # ≈ 0.190
}
# Fallback (e.g. if a panel reports a label that's not in the map)
delta_thr = 1.0 / np.sqrt(n_ref)


def delta_thr_for(label: str) -> float:
    return DELTA_THR_BY_DATASET.get(label, delta_thr)


def render_ece_figure(panels: list[dict], provider: str,
                      provider_disp: str) -> None:
    n_cols = len(panels)
    fig1, axes1 = plt.subplots(1, n_cols, figsize=(13.5, 4.4),
                               gridspec_kw={"wspace": 0.20,
                                             "left": 0.06, "right": 0.985,
                                             "top": 0.86, "bottom": 0.16})
    if n_cols == 1:
        axes1 = [axes1]

    ece_ymax = 0.0
    for p in panels:
        _, e1, _ = calibration_curve_by_delta(p["delta"], p["c1"], p["y"])
        _, e2, _ = calibration_curve_by_delta(p["delta"], p["c2"], p["y"])
        if len(e1):
            ece_ymax = max(ece_ymax, float(np.concatenate([e1, e2]).max()))
    ece_ymax = min(1.0, ece_ymax * 1.18 + 0.02) if ece_ymax > 0 else 1.0

    for j, (ax, p) in enumerate(zip(axes1, panels)):
        x, e1, _ = calibration_curve_by_delta(p["delta"], p["c1"], p["y"])
        _, e2, n = calibration_curve_by_delta(p["delta"], p["c2"], p["y"])
        p["delta_bins"] = (x, e1, e2, n)

        overlay_density(ax, p["delta"], ymax=ece_ymax, color=C_SEM1,
                        alpha=0.42, n_bins=20, bottom_frac=0.24,
                        offset=-0.012, width_scale=0.46)
        overlay_density(ax, p["delta"], ymax=ece_ymax, color=C_SEM2,
                        alpha=0.42, n_bins=20, bottom_frac=0.24,
                        offset=+0.012, width_scale=0.46)

        ax.axvline(delta_thr_for(p["label"]), color=C_GOLD, lw=1.2,
                   alpha=0.85, zorder=1)
        ax.plot(x, e1, color=C_SEM1, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM1, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4)
        ax.plot(x, e2, color=C_SEM2, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM2, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4)

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, ece_ymax)
        ax.set_xlabel(r"per-question margin $\Delta$", fontsize=13)
        if ax is axes1[0]:
            ax.set_ylabel("ECE", fontsize=13)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
        # Right-side density y-axis (shown only on the rightmost panel)
        add_density_yaxis(ax, bottom_frac=0.24,
                          show_label=(j == len(panels) - 1))

    handles = [
        Line2D([], [], color=C_SEM1, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM1, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_1$-ECE"),
        Line2D([], [], color=C_SEM2, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM2, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_2$-ECE"),
        Line2D([], [], color=C_GOLD, lw=1.2,
               label=r"low-margin threshold  $n\Delta^{2}{=}\log K$"),
    ]
    fig1.legend(handles=handles, loc="upper center",
                bbox_to_anchor=(0.5, 0.97),
                ncol=3, frameon=False, fontsize=15.5,
                handlelength=2.0, handletextpad=0.5, columnspacing=2.2)

    fig1.savefig(OUT / f"ece_vs_margin_{provider}.pdf")
    fig1.savefig(OUT / f"ece_vs_margin_{provider}.png", dpi=300)
    plt.close(fig1)


# ===========================================================================
# Figure 2: reliability diagram  (three methods, with Verbalized added)
# ===========================================================================
def render_reliability_figure(panels: list[dict], provider: str,
                              provider_disp: str) -> None:
    n_cols = len(panels)
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(13.5, 5.0),
                               gridspec_kw={"wspace": 0.20,
                                             "left": 0.06, "right": 0.985,
                                             "top": 0.88, "bottom": 0.13})
    if n_cols == 1:
        axes2 = [axes2]

    for j, (ax, p) in enumerate(zip(axes2, panels)):
        x1, a1, n1 = reliability_curve(p["c1"], p["y"])
        x2, a2, n2 = reliability_curve(p["c2"], p["y"])
        xv, av, nv = reliability_curve(p["cv"], p["y"])
        ece1 = overall_ece(p["c1"], p["y"])
        ece2 = overall_ece(p["c2"], p["y"])
        ecev = overall_ece(p["cv"], p["y"])
        p["rel_bins"] = dict(scope1=ece1, scope2=ece2, verb=ecev)

        overlay_density(ax, p["c1"], ymax=1.0, color=C_SEM1,
                        alpha=0.42, n_bins=20, bottom_frac=0.22,
                        offset=-0.018, width_scale=0.30)
        overlay_density(ax, p["c2"], ymax=1.0, color=C_SEM2,
                        alpha=0.42, n_bins=20, bottom_frac=0.22,
                        offset=+0.000, width_scale=0.30)
        overlay_density(ax, p["cv"], ymax=1.0, color=C_VERB,
                        alpha=0.42, n_bins=20, bottom_frac=0.22,
                        offset=+0.018, width_scale=0.30)

        ax.plot([0, 1], [0, 1], color=C_DIAG, lw=1.0, ls="--", zorder=1)
        ax.plot(x1, a1, color=C_SEM1, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM1, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4,
                label=fr"Sem$_1$-ECE: ${ece1:.3f}$")
        ax.plot(x2, a2, color=C_SEM2, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM2, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4,
                label=fr"Sem$_2$-ECE: ${ece2:.3f}$")
        ax.plot(xv, av, color=C_VERB, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_VERB, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4,
                label=fr"Ver-ECE: ${ecev:.3f}$")

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"predicted confidence  $\hat c$", fontsize=13)
        if ax is axes2[0]:
            ax.set_ylabel(r"empirical accuracy  $\bar Y$", fontsize=13)
        ax.legend(loc="upper left", frameon=False, fontsize=14,
                  handlelength=1.5, handletextpad=0.4, labelspacing=0.35)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.35)
        # Right-side density y-axis (label only on rightmost)
        add_density_yaxis(ax, bottom_frac=0.20,
                          show_label=(j == len(panels) - 1))

    handles = [
        Line2D([], [], color=C_SEM1, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM1, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_1$-ECE"),
        Line2D([], [], color=C_SEM2, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM2, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_2$-ECE"),
        Line2D([], [], color=C_VERB,   lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_VERB,   markeredgecolor="black",
               markeredgewidth=0.5, label="Ver-ECE"),
        Line2D([], [], color=C_DIAG, lw=1.0, ls="--",
               label="perfect calibration"),
    ]
    fig2.legend(handles=handles, loc="upper center",
                bbox_to_anchor=(0.5, 0.97),
                ncol=4, frameon=False, fontsize=15.5,
                handlelength=2.0, handletextpad=0.5, columnspacing=2.2)

    fig2.savefig(OUT / f"reliability_{provider}.pdf")
    fig2.savefig(OUT / f"reliability_{provider}.png", dpi=300)
    plt.close(fig2)


# ===========================================================================
# Pooled figures: all 5 providers mixed into one curve per dataset.
# Uses the intersection of question IDs across providers so every provider
# contributes the same set of questions (avoids xAI/PopQA size mismatch).
# ===========================================================================
def build_pooled_panels() -> list[dict]:
    panels = []
    for ds_label, phase in DATASETS:
        qids = common_qids(phase)
        deltas, c1s, c2s, cvs, ys = [], [], [], [], []
        for prov in PROVIDERS:
            cell = load_cell(phase, prov, qid_filter=qids)
            if cell is None:
                continue
            d, c1, c2, cv, y = cell
            deltas.append(d); c1s.append(c1); c2s.append(c2)
            cvs.append(cv);   ys.append(y)
        if not deltas:
            continue
        panels.append({
            "label": ds_label,
            "delta": np.concatenate(deltas),
            "c1":    np.concatenate(c1s),
            "c2":    np.concatenate(c2s),
            "cv":    np.concatenate(cvs),
            "y":     np.concatenate(ys),
        })
        panels[-1]["N"] = len(panels[-1]["delta"])
    return panels


def render_pooled_ece(panels: list[dict]) -> None:
    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(13.5, 4.4),
                             gridspec_kw={"wspace": 0.20,
                                           "left": 0.06, "right": 0.985,
                                           "top": 0.86, "bottom": 0.16})
    if n_cols == 1:
        axes = [axes]

    ece_ymax = 0.0
    for p in panels:
        _, e1, _ = calibration_curve_by_delta(p["delta"], p["c1"], p["y"])
        _, e2, _ = calibration_curve_by_delta(p["delta"], p["c2"], p["y"])
        if len(e1):
            ece_ymax = max(ece_ymax, float(np.concatenate([e1, e2]).max()))
    ece_ymax = min(1.0, ece_ymax * 1.18 + 0.02) if ece_ymax > 0 else 1.0

    for j, (ax, p) in enumerate(zip(axes, panels)):
        x, e1, _ = calibration_curve_by_delta(p["delta"], p["c1"], p["y"])
        _, e2, _ = calibration_curve_by_delta(p["delta"], p["c2"], p["y"])

        overlay_density(ax, p["delta"], ymax=ece_ymax, color=C_SEM1,
                        alpha=0.42, n_bins=20, bottom_frac=0.24,
                        offset=-0.012, width_scale=0.46)
        overlay_density(ax, p["delta"], ymax=ece_ymax, color=C_SEM2,
                        alpha=0.42, n_bins=20, bottom_frac=0.24,
                        offset=+0.012, width_scale=0.46)

        ax.axvline(delta_thr_for(p["label"]), color=C_GOLD, lw=1.2,
                   alpha=0.85, zorder=1)
        ax.plot(x, e1, color=C_SEM1, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM1, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4)
        ax.plot(x, e2, color=C_SEM2, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM2, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4)

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, ece_ymax)
        ax.set_xlabel(r"per-question margin $\Delta$", fontsize=13)
        if ax is axes[0]:
            ax.set_ylabel("ECE", fontsize=13)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.35)
        add_density_yaxis(ax, bottom_frac=0.22,
                          show_label=(j == len(panels) - 1))

    handles = [
        Line2D([], [], color=C_SEM1, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM1, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_1$-ECE"),
        Line2D([], [], color=C_SEM2, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM2, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_2$-ECE"),
        Line2D([], [], color=C_GOLD, lw=1.2,
               label=r"low-margin threshold  $n\Delta^{2}{=}\log K$"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.97),
               ncol=3, frameon=False, fontsize=15.5,
               handlelength=2.0, handletextpad=0.5, columnspacing=2.2)

    fig.savefig(OUT / "ece_vs_margin_pooled.pdf")
    fig.savefig(OUT / "ece_vs_margin_pooled.png", dpi=300)
    plt.close(fig)


def render_pooled_reliability(panels: list[dict]) -> None:
    n_cols = len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(13.5, 5.0),
                             gridspec_kw={"wspace": 0.20,
                                           "left": 0.06, "right": 0.985,
                                           "top": 0.88, "bottom": 0.13})
    if n_cols == 1:
        axes = [axes]

    for j, (ax, p) in enumerate(zip(axes, panels)):
        x1, a1, _ = reliability_curve(p["c1"], p["y"])
        x2, a2, _ = reliability_curve(p["c2"], p["y"])
        xv, av, _ = reliability_curve(p["cv"], p["y"])
        ece1 = overall_ece(p["c1"], p["y"])
        ece2 = overall_ece(p["c2"], p["y"])
        ecev = overall_ece(p["cv"], p["y"])

        overlay_density(ax, p["c1"], ymax=1.0, color=C_SEM1,
                        alpha=0.42, n_bins=20, bottom_frac=0.22,
                        offset=-0.018, width_scale=0.30)
        overlay_density(ax, p["c2"], ymax=1.0, color=C_SEM2,
                        alpha=0.42, n_bins=20, bottom_frac=0.22,
                        offset=+0.000, width_scale=0.30)
        overlay_density(ax, p["cv"], ymax=1.0, color=C_VERB,
                        alpha=0.42, n_bins=20, bottom_frac=0.22,
                        offset=+0.018, width_scale=0.30)

        ax.plot([0, 1], [0, 1], color=C_DIAG, lw=1.0, ls="--", zorder=1)
        ax.plot(x1, a1, color=C_SEM1, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM1, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4,
                label=fr"Sem$_1$-ECE: ${ece1:.3f}$")
        ax.plot(x2, a2, color=C_SEM2, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_SEM2, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4,
                label=fr"Sem$_2$-ECE: ${ece2:.3f}$")
        ax.plot(xv, av, color=C_VERB, lw=2.0, marker="o", markersize=5.5,
                markerfacecolor=C_VERB, markeredgecolor="black",
                markeredgewidth=0.5, zorder=4,
                label=fr"Ver-ECE: ${ecev:.3f}$")

        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"predicted confidence  $\hat c$", fontsize=13)
        if ax is axes[0]:
            ax.set_ylabel(r"empirical accuracy  $\bar Y$", fontsize=13)
        ax.legend(loc="upper left", frameon=False, fontsize=14,
                  handlelength=1.5, handletextpad=0.4, labelspacing=0.35)
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.35)
        add_density_yaxis(ax, bottom_frac=0.22,
                          show_label=(j == len(panels) - 1))

    handles = [
        Line2D([], [], color=C_SEM1, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM1, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_1$-ECE"),
        Line2D([], [], color=C_SEM2, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM2, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_2$-ECE"),
        Line2D([], [], color=C_VERB, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_VERB, markeredgecolor="black",
               markeredgewidth=0.5, label="Ver-ECE"),
        Line2D([], [], color=C_DIAG, lw=1.0, ls="--",
               label="perfect calibration"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.97),
               ncol=4, frameon=False, fontsize=15.5,
               handlelength=2.0, handletextpad=0.5, columnspacing=2.2)

    fig.savefig(OUT / "reliability_pooled.pdf")
    fig.savefig(OUT / "reliability_pooled.png", dpi=300)
    plt.close(fig)


# ===========================================================================
# Combined figures: 5 rows (providers) × 3 cols (datasets), no titles.
# Provider names are placed as rotated row labels on the left margin;
# dataset names sit as small column labels above the top row.
# ===========================================================================
def render_combined_ece(all_panels: dict[str, list[dict]]) -> None:
    n_rows = len(PROVIDERS)
    n_cols = len(DATASETS)

    # global ymax across all (provider, dataset) cells
    ece_ymax = 0.0
    for prov in PROVIDERS:
        for p in all_panels.get(prov, []):
            _, e1, _ = calibration_curve_by_delta(p["delta"], p["c1"], p["y"])
            _, e2, _ = calibration_curve_by_delta(p["delta"], p["c2"], p["y"])
            if len(e1):
                ece_ymax = max(ece_ymax,
                               float(np.concatenate([e1, e2]).max()))
    ece_ymax = min(1.0, ece_ymax * 1.18 + 0.02) if ece_ymax > 0 else 1.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 14.0),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.10, "hspace": 0.16,
                                           "left": 0.10, "right": 0.985,
                                           "top": 0.94, "bottom": 0.06})

    # Column labels (dataset names) above the top row
    for j, (ds_label, _) in enumerate(DATASETS):
        axes[0, j].annotate(
            ds_label,
            xy=(0.5, 1.02), xycoords="axes fraction",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#333",
        )

    for i, prov in enumerate(PROVIDERS):
        # Row label (provider name) on the left margin
        axes[i, 0].annotate(
            PROVIDER_DISPLAY[prov],
            xy=(-0.16, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color="#333",
        )

        panels = all_panels.get(prov, [])
        # Map ds_label -> panel for safe lookup
        by_label = {p["label"]: p for p in panels}

        for j, (ds_label, _) in enumerate(DATASETS):
            ax = axes[i, j]
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, ece_ymax)

            p = by_label.get(ds_label)
            if p is None:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center", color="#aaa", fontsize=14)
                continue

            x, e1, _ = calibration_curve_by_delta(p["delta"], p["c1"], p["y"])
            _, e2, _ = calibration_curve_by_delta(p["delta"], p["c2"], p["y"])

            overlay_density(ax, p["delta"], ymax=ece_ymax, color=C_SEM1,
                            alpha=0.42, n_bins=20, bottom_frac=0.24,
                            offset=-0.012, width_scale=0.46)
            overlay_density(ax, p["delta"], ymax=ece_ymax, color=C_SEM2,
                            alpha=0.42, n_bins=20, bottom_frac=0.24,
                            offset=+0.012, width_scale=0.46)

            ax.axvline(delta_thr_for(ds_label), color=C_GOLD, lw=1.0,
                       alpha=0.85, zorder=1)
            ax.plot(x, e1, color=C_SEM1, lw=1.7, marker="o", markersize=4.5,
                    markerfacecolor=C_SEM1, markeredgecolor="black",
                    markeredgewidth=0.4, zorder=4)
            ax.plot(x, e2, color=C_SEM2, lw=1.7, marker="o", markersize=4.5,
                    markerfacecolor=C_SEM2, markeredgecolor="black",
                    markeredgewidth=0.4, zorder=4)
            ax.grid(axis="y", linestyle=":", linewidth=0.4, alpha=0.30)
            # Right-side density y-axis: only on rightmost column, label
            # only on the bottom-rightmost panel.
            if j == len(DATASETS) - 1:
                add_density_yaxis(ax, bottom_frac=0.24,
                                  show_label=(i == len(PROVIDERS) - 1))

    # Shared axis labels
    fig.supxlabel(r"per-question margin $\Delta$", fontsize=13)
    fig.supylabel("ECE", fontsize=13, x=0.025)

    handles = [
        Line2D([], [], color=C_SEM1, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM1, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_1$-ECE"),
        Line2D([], [], color=C_SEM2, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM2, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_2$-ECE"),
        Line2D([], [], color=C_GOLD, lw=1.2,
               label=r"low-margin threshold  $n\Delta^{2}{=}\log K$"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99),
               ncol=3, frameon=False, fontsize=15.5,
               handlelength=2.0, handletextpad=0.5, columnspacing=2.2)

    fig.savefig(OUT / "ece_vs_margin_all.pdf")
    fig.savefig(OUT / "ece_vs_margin_all.png", dpi=300)
    plt.close(fig)


def render_combined_reliability(all_panels: dict[str, list[dict]]) -> None:
    n_rows = len(PROVIDERS)
    n_cols = len(DATASETS)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 18.5),
                             sharex=True, sharey=True,
                             gridspec_kw={"wspace": 0.10, "hspace": 0.18,
                                           "left": 0.10, "right": 0.985,
                                           "top": 0.95, "bottom": 0.05})

    for j, (ds_label, _) in enumerate(DATASETS):
        axes[0, j].annotate(
            ds_label,
            xy=(0.5, 1.04), xycoords="axes fraction",
            ha="center", va="bottom",
            fontsize=14, fontweight="bold", color="#333",
        )

    for i, prov in enumerate(PROVIDERS):
        axes[i, 0].annotate(
            PROVIDER_DISPLAY[prov],
            xy=(-0.20, 0.5), xycoords="axes fraction",
            rotation=90, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color="#333",
        )

        panels = all_panels.get(prov, [])
        by_label = {p["label"]: p for p in panels}

        for j, (ds_label, _) in enumerate(DATASETS):
            ax = axes[i, j]
            ax.set_xlim(0, 1.0)
            ax.set_ylim(0, 1.0)
            ax.set_box_aspect(1)   # box-shape aspect; compatible w/ twinx

            p = by_label.get(ds_label)
            if p is None:
                ax.text(0.5, 0.5, "—", transform=ax.transAxes,
                        ha="center", va="center", color="#aaa", fontsize=14)
                continue

            x1, a1, _ = reliability_curve(p["c1"], p["y"])
            x2, a2, _ = reliability_curve(p["c2"], p["y"])
            xv, av, _ = reliability_curve(p["cv"], p["y"])
            ece1 = overall_ece(p["c1"], p["y"])
            ece2 = overall_ece(p["c2"], p["y"])
            ecev = overall_ece(p["cv"], p["y"])

            overlay_density(ax, p["c1"], ymax=1.0, color=C_SEM1,
                            alpha=0.42, n_bins=20, bottom_frac=0.20,
                            offset=-0.018, width_scale=0.30)
            overlay_density(ax, p["c2"], ymax=1.0, color=C_SEM2,
                            alpha=0.42, n_bins=20, bottom_frac=0.20,
                            offset=+0.000, width_scale=0.30)
            overlay_density(ax, p["cv"], ymax=1.0, color=C_VERB,
                            alpha=0.42, n_bins=20, bottom_frac=0.20,
                            offset=+0.018, width_scale=0.30)

            ax.plot([0, 1], [0, 1], color=C_DIAG, lw=0.9, ls="--", zorder=1)
            ax.plot(x1, a1, color=C_SEM1, lw=1.7, marker="o", markersize=4.0,
                    markerfacecolor=C_SEM1, markeredgecolor="black",
                    markeredgewidth=0.4, zorder=4,
                    label=fr"S1: ${ece1:.3f}$")
            ax.plot(x2, a2, color=C_SEM2, lw=1.7, marker="o", markersize=4.0,
                    markerfacecolor=C_SEM2, markeredgecolor="black",
                    markeredgewidth=0.4, zorder=4,
                    label=fr"S2: ${ece2:.3f}$")
            ax.plot(xv, av, color=C_VERB, lw=1.7, marker="o", markersize=4.0,
                    markerfacecolor=C_VERB, markeredgecolor="black",
                    markeredgewidth=0.4, zorder=4,
                    label=fr"V: ${ecev:.3f}$")
            ax.grid(linestyle=":", linewidth=0.4, alpha=0.30)
            ax.legend(loc="upper left", frameon=False, fontsize=8.5,
                      handlelength=1.0, handletextpad=0.3, labelspacing=0.25)
            # Right-side density y-axis: only on rightmost column.
            if j == len(DATASETS) - 1:
                add_density_yaxis(ax, bottom_frac=0.20,
                                  show_label=(i == len(PROVIDERS) - 1))

    fig.supxlabel(r"predicted confidence  $\hat c$", fontsize=13)
    fig.supylabel(r"empirical accuracy  $\bar Y$", fontsize=13, x=0.025)

    handles = [
        Line2D([], [], color=C_SEM1, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM1, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_1$-ECE"),
        Line2D([], [], color=C_SEM2, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_SEM2, markeredgecolor="black",
               markeredgewidth=0.5, label=r"Sem$_2$-ECE"),
        Line2D([], [], color=C_VERB, lw=2.0, marker="o", markersize=7,
               markerfacecolor=C_VERB, markeredgecolor="black",
               markeredgewidth=0.5, label="Ver-ECE"),
        Line2D([], [], color=C_DIAG, lw=1.0, ls="--",
               label="perfect calibration"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99),
               ncol=4, frameon=False, fontsize=15.5,
               handlelength=2.0, handletextpad=0.5, columnspacing=2.2)

    fig.savefig(OUT / "reliability_all.pdf")
    fig.savefig(OUT / "reliability_all.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Drive: for each of the 5 providers, render both figures
# ---------------------------------------------------------------------------
print("=" * 78)
print("Per-provider figures: ECE-vs-Δ  +  reliability  (5 × 2 = 10 files)")
print("=" * 78)
print(f"Low-margin threshold: Δ < 1/√n = {delta_thr:.3f}  (n={n_ref})\n")

all_panels: dict[str, list[dict]] = {}
for prov in PROVIDERS:
    panels = build_panels_for_provider(prov)
    if not panels:
        continue
    all_panels[prov] = panels
    render_ece_figure(panels, prov, PROVIDER_DISPLAY[prov])
    render_reliability_figure(panels, prov, PROVIDER_DISPLAY[prov])

    print(f"--- {PROVIDER_DISPLAY[prov]} ({prov}) ---")
    for p in panels:
        rb = p["rel_bins"]
        x, e1, e2, ct = p["delta_bins"]
        lo_mask = x < delta_thr
        gap_lo = (np.average((e1 - e2)[lo_mask], weights=ct[lo_mask])
                  if lo_mask.any() else float("nan"))
        print(f"  {p['label']:<10s} N={p['N']:>5,d}  "
              f"ECE: S1={rb['scope1']:.3f}  S2={rb['scope2']:.3f}  "
              f"Verb={rb['verb']:.3f}  low-Δ gap={gap_lo:+.3f}")
    print()

# Combined 5x3 grids covering all providers in one figure each
render_combined_ece(all_panels)
render_combined_reliability(all_panels)

# Pooled figures: all 5 providers mixed (intersection of qids)
pooled = build_pooled_panels()
if pooled:
    render_pooled_ece(pooled)
    render_pooled_reliability(pooled)
    print("--- Pooled (5 providers, intersection of qids) ---")
    for p in pooled:
        ece1 = overall_ece(p["c1"], p["y"])
        ece2 = overall_ece(p["c2"], p["y"])
        ecev = overall_ece(p["cv"], p["y"])
        print(f"  {p['label']:<10s} N={p['N']:>6,d}  "
              f"ECE: S1={ece1:.3f}  S2={ece2:.3f}  Verb={ecev:.3f}")
    print()

print("Saved 10 per-provider files to figures/:")
for prov in PROVIDERS:
    print(f"  ece_vs_margin_{prov}.{{pdf,png}}   reliability_{prov}.{{pdf,png}}")
print("Saved 2 combined files (5 rows × 3 cols):")
print(f"  ece_vs_margin_all.{{pdf,png}}    reliability_all.{{pdf,png}}")
print("Saved 2 pooled files (1 row × 3 cols, all 5 providers mixed):")
print(f"  ece_vs_margin_pooled.{{pdf,png}} reliability_pooled.{{pdf,png}}")
