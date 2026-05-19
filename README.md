# Sem-ECE: Sample-Split Semantic Consensus for Self-Consistency Confidence

This package reproduces all figures and the bootstrap-CI tables from the paper.

It contains **only the downstream analysis pipeline** (clustering JSONL →
confidence → ECE/figures).  The upstream batch-generation code (calls to
five LLM providers' batch APIs) is *not* included; the `prompts/` directory
documents the exact LLM prompts used for generation, clustering, and grading.

---

## 1. Quick start

```bash
pip install -r requirements.txt

# Theory-only figures (no data needed)
python figure1.py            # → figures/figure1a.{pdf,png}, figures/figure1b.{pdf,png}
python figure_workflow.py    # → figures/workflow_example.{pdf,png}

# Empirical figures (needs `data/` populated — see §3)
python figure2.py            # → figure2.{pdf,png}, figures/n_sweep_full.*, figures/n_sweep_lowdelta.*
python figure3.py            # → figure3.{pdf,png}
python figure_ece_vs_margin.py
python figure_per_provider.py
python bootstrap_ci.py
```

## 2. Methods (one-paragraph summary)

Given $n$ stochastic LLM samples on a question, the proposed estimator is

* **Sem₁ (plug-in)** — top-cluster frequency $\hat c_1 = (\max_k |C_k|)/n$
* **Sem₂ (Sample-Split Semantic Consensus, SSSC)** — for $R{=}10$ random
  half-splits, pick the modal cluster on the first half and report the
  agreement frequency on the second half; average the $R$ values.
* **Ver (verbalized)** — model self-reported confidence (missing values
  imputed to 1.0).

ECE is computed in 10 equal-width bins.  See `compute_confidence.py` for
the exact formulas.

## 3. Data layout

The figure scripts expect this layout under `data/`:

```
data/
├── phase1/                            # SimpleQA
│   ├── clustering/<provider>_simpleqa_clustered.jsonl
│   ├── grading/<provider>_*_graded.jsonl
│   └── confidence/<provider>_confidence.jsonl
├── phase2/                            # HLE  (uses *_simpleqa_clustered.jsonl filenames for legacy reasons)
│   └── …
└── phase4/                            # PopQA
    └── …
```

`<provider>` ∈ `{openai, anthropic, gemini, xai, mistral}`.

**Where to get the data**: SimpleQA, HLE, and PopQA are public benchmarks.
SimpleQA: https://openai.com/index/introducing-simpleqa/ ;
HLE: https://github.com/centerforaisafety/hle ;
PopQA: https://github.com/AlexTMallen/adaptive-retrieval (or HuggingFace).
The five LLM providers' batch APIs were used to draw $n{=}50$ samples per
question, which were then clustered by a judge LLM (prompts in
`prompts/`).  Generated `data/` totals roughly 1 GB compressed per
benchmark; the cluster/grade/confidence JSONL files we ship are derived
artifacts.

### File schemas

`<provider>_<suffix>_clustered.jsonl` — one JSON per question with fields:
```
id, question, gold_answer, n_samples, cluster_assignments, cluster_sizes,
modal_cluster, modal_cluster_label, n_clusters,
verbalized_mean, verbalized_n_extracted
```

`<provider>_*_graded.jsonl` — `{id, Y}` where `Y ∈ {0, 1}` is the gold
match for the question's modal answer.

`<provider>_confidence.jsonl` (output of `compute_confidence.py`) —
`{id, model, conf_naive, conf_sssc, conf_verbalized, empirical_margin,
modal_freq, …}`.

## 4. Reproducing each figure

| Script                       | Output                                | Inputs needed              |
|------------------------------|---------------------------------------|----------------------------|
| `figure1.py`                 | Figure 1a, 1b (theory regime + curves)| —                          |
| `figure2.py`                 | Figure 2 (n-sweep)                    | clustering, grading        |
| `figure3.py`                 | Figure 3 (ECE vs margin Δ_q, pooled)  | clustering, grading, confidence |
| `figure_ece_vs_margin.py`    | per-(dataset × provider) variants     | clustering, grading, confidence |
| `figure_per_provider.py`     | reliability + ECE-vs-Δ per provider   | clustering, grading, confidence |
| `bootstrap_ci.py`            | paired bootstrap 95% CI tables        | grading, confidence        |
| `compute_confidence.py`      | derives `*_confidence.jsonl`          | clustering only            |

Run order if starting from clustering JSONL:
```bash
python compute_confidence.py --phase phase1                  # SimpleQA
python compute_confidence.py --phase phase2                  # HLE
python compute_confidence.py --phase phase4 --ds-suffix popqa # PopQA
python figure3.py
…
```

## 5. Constants used in figures

* `tilde_lambda_star ≈ 0.306` (zero of $g_B(\tilde\lambda) =
  \phi(2\tilde\lambda) - 4\tilde\lambda \Phi(-2\tilde\lambda)$)
* JDR boundary: $\Delta_q = 2\tilde\lambda^\star/\sqrt{n} = 0.612/\sqrt{n}$
* Low/large boundary: $\Delta_q = \sqrt{\log K / n}$
* Pooled K per dataset: SimpleQA 7.48, HLE 7.49, PopQA 6.05 (5-provider concat)
* $n = 50$ samples per question; $R = 10$ half-splits for Sem₂

## 6. Dependencies

See `requirements.txt`.  Tested with Python 3.11.

## 7. License

MIT — see `LICENSE`.
