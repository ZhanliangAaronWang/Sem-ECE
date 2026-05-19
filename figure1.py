"""figure1.py — Figure 1 of this paper, two separate sub-figures.

Outputs
    figures/figure1a.{pdf,png}    leading constants g_A, g_B vs tilde-lambda
    figures/figure1b.{pdf,png}    regime diagram on the (tilde-m_q, K) plane

Theory:
    g_A(x) = phi(2x)
    g_B(x) = phi(2x) - 4x * Phi(-2x)
    tilde-lambda^star = 0.306         (zero of g_B)
    JDR boundary at  tilde-m_q = 2 * tilde-lambda^star = 0.612
    Low / large boundary at  tilde-m_q = sqrt(log K)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import norm


# ---------- Theory constants ------------------------------------------------
TL_STAR = 0.306
TM_JDR  = 2.0 * TL_STAR

OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ---------- Functions -------------------------------------------------------
def g_A(x: np.ndarray) -> np.ndarray:
    return norm.pdf(2.0 * x)


def g_B(x: np.ndarray) -> np.ndarray:
    return norm.pdf(2.0 * x) - 4.0 * x * norm.cdf(-2.0 * x)


# ---------- Style -----------------------------------------------------------
rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "cm",
    "font.size":        13,
    "axes.labelsize":   16,
    "axes.titlesize":   15,
    "legend.fontsize":  13,
    "xtick.labelsize":  12.5,
    "ytick.labelsize":  12.5,
    "xtick.color":      "#444",
    "ytick.color":      "#444",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "#111",
    "axes.linewidth":   1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.08,
})

# Palette  (deeper, more saturated than the first pass)
C_GA       = "#1B4332"   # deep forest green
C_GB       = "#9C2A39"   # deep brick red
C_JDR_FILL = "#9DB7D6"   # soft slate-blue (richer than the previous pale)
C_ZERO     = "#888"

C_REG_JDR   = "#D4EBD8"
C_REG_MID   = "#F7EBC9"
C_REG_LARGE = "#E8E8E8"
C_BD_JDR    = "#1F4E5F"
C_BD_LARGE  = "#C0703A"
C_REG_LABEL = "#555"


# ---------- Panel (a) -------------------------------------------------------
def render_a():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(left=0.13, right=0.965, top=0.93, bottom=0.16)

    x  = np.linspace(0.0, 0.55, 400)
    yA = g_A(x)
    yB = g_B(x)

    # Plot ranges first so we know where the spine sits.  y_max is set
    # generously above g_A's peak so the legend can sit in the upper-right
    # corner without ever overlapping the curve.
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(-0.13, 0.58)

    # JDR shading (under everything) — richer alpha so it reads on print.
    ax.axvspan(0.0, TL_STAR, color=C_JDR_FILL, alpha=0.28,
               linewidth=0, zorder=0,
               label=r"JDR  ($\tilde\lambda<\tilde\lambda^{\!\star}$)")

    # Zero baseline
    ax.axhline(0.0, color=C_ZERO, linewidth=0.6, alpha=0.6, zorder=1)

    # Curves — bolder
    ax.plot(x, yA, color=C_GA, linewidth=3.2, zorder=3,
            label=r"$g_A(\tilde\lambda)$",   solid_capstyle="round")
    ax.plot(x, yB, color=C_GB, linewidth=3.2, zorder=3,
            label=r"$g_B(\tilde\lambda)$",   solid_capstyle="round")

    # tilde-lambda^star annotation: open circle at the zero of g_B (the
    # geometrically meaningful point), with an inline label tucked into the
    # whitespace above-right.  The numeric x-axis ticks are kept intact.
    ax.scatter([TL_STAR], [0.0], s=78, facecolor="white",
               edgecolor=C_GB, linewidth=2.0, zorder=6)
    ax.annotate(r"$\tilde\lambda^{\!\star}\,\approx\,0.306$",
                xy=(TL_STAR, 0.0), xytext=(TL_STAR + 0.020, 0.085),
                fontsize=15, color="#111",
                ha="left", va="bottom", zorder=7,
                arrowprops=dict(arrowstyle="-", color="#777",
                                lw=0.8, shrinkA=0, shrinkB=2))

    # Axis labels
    ax.set_xlabel(r"$\tilde\lambda$", labelpad=5)
    ax.set_ylabel("leading constant", labelpad=7)

    # Standard numeric ticks (no special replacement at lambda*)
    ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ax.set_yticks([-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])

    # Subtle horizontal grid only
    ax.yaxis.grid(True,  linestyle="-", linewidth=0.5, alpha=0.18, color="#888")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Legend (top-right, no frame, fixed order)
    handles, labels = ax.get_legend_handles_labels()
    desired = [r"$g_A(\tilde\lambda)$",
               r"$g_B(\tilde\lambda)$",
               r"JDR  ($\tilde\lambda<\tilde\lambda^{\!\star}$)"]
    order = [labels.index(l) for l in desired]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper right", frameon=False, fontsize=14,
              handlelength=1.8, handletextpad=0.6, borderaxespad=0.6)

    # Panel label  (this panel is figure 1b)
    ax.text(-0.095, 1.04, "(b)", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="left",
            color="#111")

    out_pdf = OUT_DIR / "figure1b.pdf"
    out_png = OUT_DIR / "figure1b.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------- Panel (b) -------------------------------------------------------
def render_b():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    fig.subplots_adjust(left=0.12, right=0.965, top=0.93, bottom=0.16)

    x_max = 3.0

    # Build region fills using fill_betweenx along the K axis
    K_grid   = np.logspace(np.log10(2.0), np.log10(100.0), 500)
    tm_curve = np.sqrt(np.log(K_grid))

    ax.fill_betweenx(K_grid, np.zeros_like(K_grid),
                     np.full_like(K_grid, TM_JDR),
                     color=C_REG_JDR, alpha=0.70, linewidth=0, zorder=0)
    ax.fill_betweenx(K_grid, np.full_like(K_grid, TM_JDR),
                     np.maximum(tm_curve, TM_JDR),
                     color=C_REG_MID, alpha=0.70, linewidth=0, zorder=0)
    ax.fill_betweenx(K_grid, tm_curve, np.full_like(K_grid, x_max),
                     color=C_REG_LARGE, alpha=0.70, linewidth=0, zorder=0)

    # Boundaries.  Use a tighter dash pattern so the legend handle shows the
    # full dashed pattern instead of a single short bar.
    ax.axvline(TM_JDR, color=C_BD_JDR, linestyle=(0, (4, 2)),
               linewidth=3.0, zorder=3,
               label=r"$\tilde m_q = 2\tilde\lambda^{\!\star}$")
    ax.plot(tm_curve, K_grid, color=C_BD_LARGE, linewidth=3.2,
            zorder=3, label=r"$\tilde m_q = \sqrt{\log K}$",
            solid_capstyle="round")

    # Region labels — placed for clear separation, no overlap with the
    # boundary lines.
    # JDR strip is narrow → rotated 90° at the strip mid-line, mid-K.
    ax.text(TM_JDR / 2.0, 8.5, "Jensen-dominated",
            color=C_REG_LABEL, fontsize=13.5, fontstyle="italic",
            ha="center", va="center", rotation=90, zorder=2)
    # Low non-JDR: pick a (m, K) where the band is wide.  At K ≈ 30 the
    # band runs from 0.612 to √log 30 ≈ 1.84, so x ≈ 1.18 is well-centred.
    ax.text(1.18, 30.0, "low-margin,\nnot JDR",
            color=C_REG_LABEL, fontsize=13.5, fontstyle="italic",
            ha="center", va="center", linespacing=1.15, zorder=2)
    # Large-margin: well to the right of the curve at mid-K.
    ax.text(2.55, 8.5, "large-margin",
            color=C_REG_LABEL, fontsize=13.5, fontstyle="italic",
            ha="center", va="center", zorder=2)

    # Axes
    ax.set_yscale("log")
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(2.0, 100.0)
    ax.set_xlabel(r"$\tilde m_q$", labelpad=5)
    ax.set_ylabel(r"$K$", labelpad=7)

    yticks = [2, 5, 10, 20, 50, 100]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(t) for t in yticks])
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.minorticks_off()

    # Legend (lower-right) — placed in the empty large-margin band below the
    # "large-margin" label, where neither boundary line passes.  Long handle
    # so the dashed JDR pattern is fully visible.
    leg = ax.legend(loc="lower right", frameon=False, fontsize=14,
                    handlelength=3.6, handletextpad=0.7, borderaxespad=0.8)

    # Panel label  (this panel is figure 1a)
    ax.text(-0.095, 1.04, "(a)", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="bottom", ha="left",
            color="#111")

    out_pdf = OUT_DIR / "figure1a.pdf"
    out_png = OUT_DIR / "figure1a.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------- Main ------------------------------------------------------------
def main():
    render_a()
    render_b()
    print(f"Saved {OUT_DIR/'figure1a.pdf'}")
    print(f"Saved {OUT_DIR/'figure1b.pdf'}")


if __name__ == "__main__":
    main()
