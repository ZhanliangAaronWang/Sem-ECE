# Sem-ECE — complete code and prompt release

Everything needed to reproduce the paper and its rebuttal experiments: 274 source
files (72,039 lines) and 12 distinct prompts.

- **`code/`** — all source, with the original directory tree preserved. The scripts use
  relative paths and `sys.path` insertions between directories, so the layout is kept
  as-is rather than reorganised; `MANIFEST.md` provides the functional map instead.
- **`prompts/`** — every prompt, extracted from the Python constants it is embedded in and
  written out verbatim as `.txt`, deduplicated with full provenance.
- **`MANIFEST.md`** — every file, its line count and its own docstring, grouped by stage.
- **`env/`** — dependencies and the API-key template.

Not included: the ~14 GB of generations, partitions and grades.

---

## Pipeline

```
    questions
        |
   [1] GENERATION            50 samples/question x 5 providers x 3 benchmarks
        |                    OpenAI Batch | Anthropic Message Batches | Gemini Batch
        |                    | Mistral Batch | xAI async (no batch API)
        v
   answer_extracted[]        regex on the enforced "Answer:/Confidence:" tail,
        |                    with a fallback for responses that break format
        |
   [2] CLUSTERING            one LLM judge call per question partitions the 50
        |                    answers into semantic classes -> cluster_assignments[]
        v
   [3] GRADING               modal answer vs gold -> Y in {0,1}
        |
        v
   [4] ESTIMATORS            Sem1 = max_k pi_hat(k)                    (same-sample)
                             Sem2 = held-out frequency of the split-selected class
                             Ver  = mean elicited confidence
        |
        v
        ECE, 10 equal-width bins
```

Stages 1-3 are provider- and dataset-specific and live under `phase1`-`phase5`; the
estimators are small and are re-implemented inside each analysis script (see
`cross_judge/compare.py` for the reference implementation that the others import).

## Reproducing each result

| result | entry point |
|---|---|
| main tables (Sem1/Sem2/Ver ECE) | `code/cross_judge/compare.py` |
| cross-judge robustness, 4 judges | `code/cross_judge/`, `code/cross_judge_claude/`, `code/cross_judge_kimi/` |
| n = 50 -> 100 convergence | `code/n100/full_table.py` |
| budget sweep B in {5,10,20,30,50} | `code/baselines/budget_sweep.py` |
| semantic-entropy baseline | `code/baselines/semantic_entropy.py` |
| temperature / top-p sensitivity | `code/temp_sweep/analyze.py` |
| clustering-prompt sensitivity | `code/prompt_sens/collect_analyze.py` |
| structural ambiguity (hedged answers) | `code/hedged/metric_stability.py`, `impact.py`, `prevalence.py` |
| clustering audit | `code/audit_cluster/build.py`, `score.py` |

## Provider quirks worth knowing before re-running

These cost real time to rediscover, so they are recorded here:

- **OpenAI** caps `n` at 8, so 50 samples requires 7 chunked requests with distinct seeds.
  `reasoning_effort=none` is required to suppress reasoning tokens.
- **Anthropic** rejects `temperature` and `top_p` together — send only one.
- **Gemini** in JSON mode truncates the closing brace on roughly 13% of responses; see the
  bracket-repair parser in `code/cross_judge/collect.py`.
- **Claude** prepends a reasoning preamble before the JSON; strip to the first `{"clusters"`.
- **xAI** does not honour native `n`; samples must be drawn with concurrent single requests.
- **Mistral** could not produce a valid exhaustive partition on ~19% of questions and was
  dropped as a judge (`code/cross_judge_mistral/` is retained for completeness).
- **QUBRID** (Kimi) has no batch API; the judge runs async at concurrency 48.

