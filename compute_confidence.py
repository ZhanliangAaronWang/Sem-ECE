"""compute_confidence.py — compute SSSC (Sem_2), naive (Sem_1), and verbalized (Ver) confidence.

Local computation only, zero API calls.  Reads `<phase>/clustering/<provider>_<suffix>_clustered.jsonl`
under `data/`; writes per-provider `<provider>_confidence.jsonl` and a pooled
parquet/CSV plus a diagnostic report into `data/<phase>/confidence/`.

Usage:
    python compute_confidence.py --phase phase1                       # SimpleQA, all providers
    python compute_confidence.py --phase phase2 --providers openai    # HLE, OpenAI only
    python compute_confidence.py --phase phase4 --ds-suffix popqa     # PopQA
"""
import os, json, argparse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(ROOT, "data")
ALL_PROVIDERS = ["openai", "anthropic", "gemini", "xai", "mistral"]
DEFAULT_DS_SUFFIX = {"phase1": "simpleqa", "phase2": "simpleqa", "phase4": "popqa"}
REQUIRED_FIELDS = ["id", "cluster_assignments", "cluster_sizes", "modal_cluster",
                   "n_samples", "verbalized_mean", "verbalized_n_extracted"]


# ── Algorithm implementations (exactly as spec) ──────────────────────

def sssc_confidence(cluster_assignments, R=10, seed=0):
    """Sample-Split Semantic Consensus."""
    a = np.asarray(cluster_assignments)
    n = len(a)
    assert n % 2 == 0, f"n must be even, got {n}"
    half = n // 2
    rng = np.random.default_rng(seed)

    confidences = []
    for r in range(R):
        perm = rng.permutation(n)
        S_idx = perm[:half]
        E_idx = perm[half:]

        S_labels = a[S_idx]
        max_label = int(a.max()) + 1
        S_counts = np.bincount(S_labels, minlength=max_label)
        z_S = int(np.argmax(S_counts))

        E_labels = a[E_idx]
        c_r = float(np.mean(E_labels == z_S))
        confidences.append(c_r)

    return float(np.mean(confidences))


def naive_confidence(cluster_sizes):
    """Plug-in self-consistency: max cluster freq."""
    s = np.asarray(cluster_sizes)
    return float(s.max() / s.sum())


def top1_minus_top2(cluster_sizes):
    """Empirical margin = (top cluster freq) - (second cluster freq)."""
    s = sorted(cluster_sizes, reverse=True)
    if len(s) == 1:
        return 1.0
    return float((s[0] - s[1]) / sum(s))


# ── Main logic ───────────────────────────────────────────────────────

def load_clustered(provider, cluster_dir, ds_suffix):
    path = os.path.join(cluster_dir, f"{provider}_{ds_suffix}_clustered.jsonl")
    if not os.path.exists(path):
        return None, path
    rows = []
    skipped = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if all(k in r for k in REQUIRED_FIELDS):
                rows.append(r)
            else:
                skipped += 1
    return rows, path


def compute_provider(provider, rows):
    records = []
    for row in rows:
        cn = naive_confidence(row["cluster_sizes"])
        cs = sssc_confidence(row["cluster_assignments"], R=10, seed=0)
        mf = row["cluster_sizes"][row["modal_cluster"]] / 50.0
        records.append({
            "id": row["id"],
            "model": provider,
            "question": row["question"],
            "gold_answer": row["gold_answer"],
            "n_samples": 50,
            "n_clusters": row["n_clusters"],
            "modal_cluster": row["modal_cluster"],
            "modal_cluster_label": row["modal_cluster_label"],
            "conf_naive": cn,
            "conf_sssc": cs,
            "conf_verbalized": row["verbalized_mean"],
            "empirical_margin": top1_minus_top2(row["cluster_sizes"]),
            "modal_freq": mf,
            "verbalized_n_extracted": row["verbalized_n_extracted"],
        })
    return records


def sanity_check(records, provider):
    errors = []
    for r in records:
        if r["conf_naive"] < 1.0 / 50:
            errors.append(f"{r['id']}: conf_naive={r['conf_naive']} < 0.02")
        if r["conf_sssc"] < 0 or r["conf_sssc"] > 1:
            errors.append(f"{r['id']}: conf_sssc={r['conf_sssc']} out of [0,1]")
        if abs(r["conf_naive"] - r["modal_freq"]) > 1e-9:
            errors.append(f"{r['id']}: conf_naive={r['conf_naive']} != modal_freq={r['modal_freq']}")
    return errors


def histogram_buckets(values, edges):
    """Count values in each bucket defined by edges."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[i + 1]):
                counts[i] += 1
                break
    return counts


def generate_report(all_records):
    """Generate diagnostic_report.md content."""
    lines = ["# Phase 1 Step 3a — Confidence Diagnostic Report\n"]

    providers = sorted(set(r["model"] for r in all_records))
    summary_rows = []

    for prov in providers:
        recs = [r for r in all_records if r["model"] == prov]
        n = len(recs)
        naive_vals = [r["conf_naive"] for r in recs]
        sssc_vals = [r["conf_sssc"] for r in recs]
        verb_vals = [r["conf_verbalized"] for r in recs if r["conf_verbalized"] is not None]
        gaps = [r["conf_naive"] - r["conf_sssc"] for r in recs]

        mn = np.mean(naive_vals)
        ms = np.mean(sssc_vals)
        mv = np.mean(verb_vals) if verb_vals else None
        gap = mn - ms

        summary_rows.append({
            "provider": prov, "n_questions": n,
            "mean_naive": mn, "mean_sssc": ms,
            "gap": gap, "mean_verbalized": mv,
        })

        lines.append(f"\n## {prov} (n={n})\n")

        # 1. Mean/median
        lines.append("### Confidence means and medians\n")
        lines.append(f"| metric | mean | median |")
        lines.append(f"|--------|------|--------|")
        lines.append(f"| conf_naive | {mn:.4f} | {np.median(naive_vals):.4f} |")
        lines.append(f"| conf_sssc | {ms:.4f} | {np.median(sssc_vals):.4f} |")
        if verb_vals:
            lines.append(f"| conf_verbalized | {np.mean(verb_vals):.4f} | {np.median(verb_vals):.4f} |")
        lines.append("")

        # 2. Jensen gap
        lines.append(f"### Jensen gap: mean(naive) - mean(sssc) = **{gap:.4f}**\n")
        if gap < 0:
            lines.append("**WARNING: NEGATIVE GAP — something may be wrong.**\n")

        # 3. Histogram of (naive - sssc)
        lines.append("### Gap histogram (conf_naive - conf_sssc)\n")
        neg_count = sum(1 for g in gaps if g < 0)
        zero_count = sum(1 for g in gaps if g == 0)
        pos_count = sum(1 for g in gaps if g > 0)
        lines.append(f"- negative: {neg_count} ({neg_count*100/n:.1f}%)")
        lines.append(f"- zero: {zero_count} ({zero_count*100/n:.1f}%)")
        lines.append(f"- positive: {pos_count} ({pos_count*100/n:.1f}%)")
        lines.append(f"- mean gap: {np.mean(gaps):.4f}, std: {np.std(gaps):.4f}")
        gap_edges = [-1, -0.1, -0.05, 0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
        gap_hist = histogram_buckets(gaps, gap_edges)
        lines.append(f"- bucketed: {list(zip([f'{gap_edges[i]:.2f}-{gap_edges[i+1]:.2f}' for i in range(len(gap_edges)-1)], gap_hist))}")
        lines.append("")

        # 4. Margin vs gap
        lines.append("### Empirical margin vs gap\n")
        lines.append("| margin bucket | n | mean gap |")
        lines.append("|--------------|---|----------|")
        margin_edges = [i / 10 for i in range(11)]
        for i in range(len(margin_edges) - 1):
            lo, hi = margin_edges[i], margin_edges[i + 1]
            bucket = [r for r in recs if lo <= r["empirical_margin"] < hi or (i == 9 and r["empirical_margin"] == 1.0)]
            if bucket:
                bg = np.mean([r["conf_naive"] - r["conf_sssc"] for r in bucket])
                lines.append(f"| [{lo:.1f}, {hi:.1f}) | {len(bucket)} | {bg:.4f} |")
        lines.append("")

        # 5. Confidence histograms
        edges10 = [i / 10 for i in range(11)]
        naive_hist = histogram_buckets(naive_vals, edges10)
        sssc_hist = histogram_buckets(sssc_vals, edges10)
        lines.append("### Confidence histograms\n")
        lines.append("| bucket | conf_naive | conf_sssc |")
        lines.append("|--------|-----------|----------|")
        for i in range(10):
            lo, hi = edges10[i], edges10[i + 1]
            lines.append(f"| [{lo:.1f}, {hi:.1f}) | {naive_hist[i]} | {sssc_hist[i]} |")
        lines.append("")

    # 6. Cross-provider table
    lines.append("\n## Cross-provider comparison\n")
    lines.append("| provider | n_questions | mean_naive | mean_sssc | gap | mean_verbalized |")
    lines.append("|----------|-----------|-----------|----------|------|----------------|")
    for s in summary_rows:
        mv_str = f"{s['mean_verbalized']:.4f}" if s["mean_verbalized"] is not None else "N/A"
        lines.append(f"| {s['provider']} | {s['n_questions']} | {s['mean_naive']:.4f} | {s['mean_sssc']:.4f} | {s['gap']:.4f} | {mv_str} |")
    lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="+", default=None)
    ap.add_argument("--phase", default="phase1",
                    help="phase1=SimpleQA, phase2=HLE, phase4=PopQA")
    ap.add_argument("--ds-suffix", default=None,
                    help="cluster file suffix (default: auto from --phase)")
    ap.add_argument("--clusters-dir", default=None,
                    help="override input clustering dir (default: data/<phase>/clustering)")
    ap.add_argument("--out-dir", default=None,
                    help="override output dir (default: data/<phase>/confidence)")
    args = ap.parse_args()
    providers   = args.providers or ALL_PROVIDERS
    ds_suffix   = args.ds_suffix or DEFAULT_DS_SUFFIX.get(args.phase, "simpleqa")
    cluster_dir = args.clusters_dir or os.path.join(DATA_ROOT, args.phase, "clustering")
    out_dir     = args.out_dir      or os.path.join(DATA_ROOT, args.phase, "confidence")
    os.makedirs(out_dir, exist_ok=True)

    all_records = []
    all_errors = []

    for prov in providers:
        rows, path = load_clustered(prov, cluster_dir, ds_suffix)
        if rows is None:
            print(f"[{prov}] SKIP — {path} not found", flush=True)
            continue
        print(f"[{prov}] loaded {len(rows)} rows from {path}", flush=True)

        records = compute_provider(prov, rows)

        # Save per-provider
        out_path = os.path.join(out_dir, f"{prov}_confidence.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{prov}] wrote {len(records)} records -> {out_path}", flush=True)

        # Sanity checks
        errors = sanity_check(records, prov)
        if errors:
            print(f"[{prov}] SANITY ERRORS: {len(errors)}", flush=True)
            for e in errors[:5]:
                print(f"  {e}", flush=True)
            all_errors.extend(errors)
        else:
            print(f"[{prov}] sanity checks PASSED", flush=True)

        all_records.extend(records)

    if not all_records:
        print("No records to process.", flush=True)
        return

    # Unified table
    df = pd.DataFrame(all_records)
    parquet_path = os.path.join(out_dir, "all_confidence.parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        print(f"\nWrote {len(df)} rows -> {parquet_path}", flush=True)
    except Exception:
        csv_path = os.path.join(out_dir, "all_confidence.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nWrote {len(df)} rows -> {csv_path}", flush=True)

    # Diagnostic report
    report = generate_report(all_records)
    report_path = os.path.join(out_dir, "diagnostic_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nDiagnostic report -> {report_path}", flush=True)
    print("\n" + report)

    # Final status
    print(f"\nTotal records: {len(all_records)}", flush=True)
    if all_errors:
        print(f"SANITY ERRORS: {len(all_errors)} — see above", flush=True)
    else:
        providers_done = sorted(set(r["model"] for r in all_records))
        neg_gap = [p for p in providers_done
                   if np.mean([r["conf_naive"] for r in all_records if r["model"] == p])
                   < np.mean([r["conf_sssc"] for r in all_records if r["model"] == p])]
        if neg_gap:
            print(f"WARNING: negative mean gap for {neg_gap}", flush=True)
        else:
            print("READY FOR ECE COMPUTATION (after grading completes)", flush=True)


if __name__ == "__main__":
    main()
