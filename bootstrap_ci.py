"""bootstrap_ci.py — paired bootstrap CIs for the Sem_1 vs Sem_2 gap.

For every (dataset, provider) cell we resample question indices with replacement
B times and compute three statistics on each bootstrap:

    Δconf_mean   = mean(c1 - c2)                            # signed confidence gap
    ΔECE         = ECE(c1) - ECE(c2)                        # overall ECE gap
    ΔECE_lowΔ    = ECE_lowΔ(c1) - ECE_lowΔ(c2)              # low-margin bin only
                                                            #   (Δ < 1/√n)

Reports point estimate + 95% percentile CI per cell as a markdown table written
to RESULTS_bootstrap_ci.md and printed to stdout.

For PopQA we use the 466-question intersection across all 5 providers (matches
the figures and the updated PopQA table in RESULTS.md).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data"

DATASETS = [
    ("SimpleQA", "phase1"),
    ("HLE",      "phase2"),
    ("PopQA",    "phase4"),
]
PROVIDERS = ["openai", "anthropic", "gemini", "xai", "mistral"]

B            = 1000          # bootstrap replicates
DELTA_THR    = 1.0 / np.sqrt(50)
SEED         = 7
N_ECE_BINS   = 10


# ---------------------------------------------------------------------------
def graded_ids(phase: str, prov: str) -> dict[str, int]:
    out = {}
    for f in (DATA_ROOT / phase / "grading").glob(f"{prov}_*_graded.jsonl"):
        with open(f) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("Y") is not None:
                    out[r["id"]] = int(r["Y"])
        break
    return out


def conf_ids(phase: str, prov: str) -> set[str]:
    out: set[str] = set()
    p = DATA_ROOT / phase / "confidence" / f"{prov}_confidence.jsonl"
    if not p.exists():
        return out
    with open(p) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("empirical_margin") is not None:
                out.add(r["id"])
    return out


def common_qids(phase: str) -> set[str]:
    sets = []
    for prov in PROVIDERS:
        ids = set(graded_ids(phase, prov).keys()) & conf_ids(phase, prov)
        if ids:
            sets.append(ids)
    return set.intersection(*sets) if sets else set()


def load_cell(phase: str, prov: str, qid_filter: set[str] | None):
    grades = graded_ids(phase, prov)
    deltas, c1s, c2s, ys = [], [], [], []
    p = DATA_ROOT / phase / "confidence" / f"{prov}_confidence.jsonl"
    if not p.exists():
        return None
    with open(p) as fh:
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
            deltas.append(float(d))
            c1s.append(float(r["conf_naive"]))
            c2s.append(float(r["conf_sssc"]))
            ys.append(grades[qid])
    if not deltas:
        return None
    return (np.asarray(deltas), np.asarray(c1s), np.asarray(c2s),
            np.asarray(ys, dtype=float))


# ---------------------------------------------------------------------------
def ece(conf: np.ndarray, y: np.ndarray, n_bins: int = N_ECE_BINS) -> float:
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    N = len(conf)
    e = 0.0
    for k in range(n_bins):
        mask = bin_idx == k
        m = int(mask.sum())
        if m:
            e += (m / N) * abs(conf[mask].mean() - y[mask].mean())
    return float(e)


def bootstrap_cell(delta: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                    y: np.ndarray, B: int = B, seed: int = SEED) -> dict:
    """Paired bootstrap on per-Q rows; returns point + 95% CI for three stats."""
    rng = np.random.default_rng(seed)
    n = len(delta)

    point_dconf  = float((c1 - c2).mean())
    point_dece   = ece(c1, y) - ece(c2, y)
    lo_full      = delta < DELTA_THR
    if lo_full.sum() >= 30:
        point_dece_lo = ece(c1[lo_full], y[lo_full]) - ece(c2[lo_full], y[lo_full])
    else:
        point_dece_lo = float("nan")

    boot_dconf  = np.empty(B)
    boot_dece   = np.empty(B)
    boot_dece_lo = np.full(B, np.nan)

    for b in range(B):
        idx = rng.integers(0, n, n)
        c1b, c2b, yb, db = c1[idx], c2[idx], y[idx], delta[idx]
        boot_dconf[b] = (c1b - c2b).mean()
        boot_dece[b]  = ece(c1b, yb) - ece(c2b, yb)
        lo = db < DELTA_THR
        if lo.sum() >= 30:
            boot_dece_lo[b] = (ece(c1b[lo], yb[lo])
                                - ece(c2b[lo], yb[lo]))

    def ci(arr: np.ndarray) -> tuple[float, float]:
        good = arr[np.isfinite(arr)]
        if len(good) < 50:
            return float("nan"), float("nan")
        return float(np.percentile(good, 2.5)), float(np.percentile(good, 97.5))

    lo_dconf,  hi_dconf  = ci(boot_dconf)
    lo_dece,   hi_dece   = ci(boot_dece)
    lo_dece_lo, hi_dece_lo = ci(boot_dece_lo)

    return dict(
        n=n,
        n_lo=int(lo_full.sum()),
        d_conf=(point_dconf, lo_dconf, hi_dconf),
        d_ece=(point_dece,  lo_dece,  hi_dece),
        d_ece_lo=(point_dece_lo, lo_dece_lo, hi_dece_lo),
    )


# ---------------------------------------------------------------------------
print("=" * 100)
print(f"Paired bootstrap CIs (B={B}) for the Sem_1 vs Sem_2 gap")
print(f"Low-margin threshold: Δ < 1/√50 = {DELTA_THR:.3f}")
print("=" * 100)

rows = []
for ds_label, phase in DATASETS:
    qid_filter = common_qids(phase) if ds_label == "PopQA" else None
    if qid_filter:
        print(f"[{ds_label}] using common-qid intersection ({len(qid_filter)} qids)")
    for prov in PROVIDERS:
        cell = load_cell(phase, prov, qid_filter)
        if cell is None:
            continue
        delta, c1, c2, y = cell
        res = bootstrap_cell(delta, c1, c2, y)
        rows.append(dict(dataset=ds_label, provider=prov, **res))

# ---------------------------------------------------------------------------
# Pretty output
# ---------------------------------------------------------------------------
def fmt_pm(point, lo, hi, decimals=4):
    if not np.isfinite(point) or not np.isfinite(lo):
        return "n/a"
    return (f"{point:+.{decimals}f}  "
            f"[{lo:+.{decimals}f}, {hi:+.{decimals}f}]")


print(f"\n{'dataset':<10}{'provider':<11}{'N':>5}{'N_lo':>5}  "
      f"{'Δ E[c1−c2]':<28}{'ΔECE (overall)':<28}{'ΔECE (low-Δ)':<28}")
print("-" * 100)
for r in rows:
    print(f"{r['dataset']:<10}{r['provider']:<11}"
          f"{r['n']:>5,d}{r['n_lo']:>5d}  "
          f"{fmt_pm(*r['d_conf']):<28}"
          f"{fmt_pm(*r['d_ece']):<28}"
          f"{fmt_pm(*r['d_ece_lo']):<28}")


# ---------------------------------------------------------------------------
# Markdown table for RESULTS.md
# ---------------------------------------------------------------------------
md_path = ROOT / "RESULTS_bootstrap_ci.md"
with open(md_path, "w") as fh:
    fh.write("# Paired bootstrap 95% CIs on the Sem_1 → Sem_2 gap\n\n")
    fh.write(f"- Replicates: B = {B}\n")
    fh.write("- Resampling unit: per-question rows (paired by Q)\n")
    fh.write("- Statistics:\n")
    fh.write("  - **ΔE[c1−c2]** — mean signed confidence reduction (positive: Sem_2 lowers c)\n")
    fh.write("  - **ΔECE** — ECE(Sem_1) − ECE(Sem_2) on all questions  (positive: Sem_2 wins)\n")
    fh.write("  - **ΔECE (low-Δ)** — same gap restricted to per-Q margin Δ < 1/√50 ≈ 0.141\n")
    fh.write("- PopQA uses the 466-Q intersection across all 5 providers.\n\n")

    fh.write("| dataset | provider | N | N (low-Δ) | ΔE[c1−c2] (CI) | ΔECE (CI) | ΔECE low-Δ (CI) |\n")
    fh.write("|---|---|---:|---:|---|---|---|\n")
    for r in rows:
        fh.write(f"| {r['dataset']} | {r['provider']} "
                 f"| {r['n']:,d} | {r['n_lo']:,d} "
                 f"| {fmt_pm(*r['d_conf'])} "
                 f"| {fmt_pm(*r['d_ece'])} "
                 f"| {fmt_pm(*r['d_ece_lo'])} |\n")
print(f"\nMarkdown table written to: {md_path}")
