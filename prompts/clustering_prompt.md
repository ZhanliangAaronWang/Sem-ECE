# Phase 1 Step 2 — Semantic clustering of SimpleQA generations

## Context
Step 1 (generation) produced 5 unified JSONL files at
`phase1/generation/{provider}_simpleqa.jsonl`, each with ~4326 rows × 50
generations per row. We now cluster the 50 generations per (question,
model) into semantic equivalence classes using GPT-5.2 as a single
judge LLM. Downstream steps (grading, SSSC computation) consume these
clustering outputs.

This step does ONLY clustering. Grading is a separate prompt that
follows.

## Configuration (FIXED)

| Parameter | Value |
|---|---|
| Judge model | `gpt-5.2` (non-reasoning mode) |
| Reasoning | DISABLED |
| API | OpenAI Batch API (24h window, ~50% discount) |
| Granularity | One clustering call per (question, model) |
| Total calls | ~4326 questions × 5 models = ~21,630 |
| Hard cost cap | $200 |

## Step 0 — Verify GPT-5.2 access and reasoning is OFF

Load `OPENAI_API_KEY` from `phase1/.env`. Send ONE non-batch test
request to `gpt-5.2` with reasoning explicitly disabled (use whichever
GPT-5 series parameter is current — likely `reasoning_effort="minimal"`
or `"none"`; verify against OpenAI docs).

Test prompt: "Reply with the single word: ok"

Verify in the response:
1. The reply contains "ok"
2. `response.usage.completion_tokens_details.reasoning_tokens == 0`

If reasoning cannot be disabled, STOP and report. We cannot proceed
with reasoning enabled — deterministic reasoning chains may produce
inconsistent partitions across calls.

Never echo or write the API key anywhere except `.env`.

## Step 1 — Answer and verbalized confidence extraction

For each generation in each `phase1/generation/{provider}_simpleqa.jsonl`,
extract BOTH the short answer AND the self-reported verbalized
confidence using this exact procedure:

**Answer extraction:**
```
1. Apply regex: r"Answer:\s*(.+?)(?=\nConfidence|\Z)" with re.DOTALL
2. If match found, strip whitespace from the captured group
3. If the result is non-empty AND length >= 1 AND not in
   {"see above", "as stated", "as mentioned", "as noted"} (case-insensitive),
   then extraction_method = "answer_line", answer_extracted = result
4. Otherwise, fall back: take the last 200 characters of the raw
   generation, strip whitespace.
   extraction_method = "fallback_tail", answer_extracted = last_200
```

**Verbalized confidence extraction:**
```
1. Apply regex: r"Confidence:\s*([0-9]+(?:\.[0-9]+)?)\s*%?"
2. If match found:
   - Parse the captured group as float
   - Divide by 100.0 to convert to [0, 1] range
   - Clamp to [0.0, 1.0] (in case model writes "150%" or negative)
   - confidence_verbalized = clamped_value
   - confidence_extraction_method = "regex_match"
3. If no match:
   - confidence_verbalized = None
   - confidence_extraction_method = "missing"
```

**Convention**: confidence values are stored in `[0, 1]` (e.g., 0.95
not 95) to match the convention used by downstream ECE / Brier
computation. Do NOT store as 0–100 integers.

Run this on all 5 providers' generation files. For each provider,
report:
- Fallback rate for answer extraction (`extraction_method == "fallback_tail"`)
- Missing rate for confidence extraction (`confidence_extraction_method == "missing"`)
- Mean and median verbalized confidence (over non-None values)

If any provider's answer fallback rate exceeds 5% OR confidence
missing rate exceeds 10%, print a warning but continue.

## Step 2 — Build batch input files

For each provider, build a clustering batch input file. One row per
question. The system message and user message templates are below —
use them VERBATIM.

**System message** (use exactly, do not modify):

```
You are a semantic clustering assistant for a research benchmark. You will receive a question and a list of model-generated answers, indexed 0 through N-1. Your job is to partition the answers into groups such that two answers are in the same group if and only if they express the same underlying answer to the question.

Rules:
- Cluster by SEMANTIC meaning, not surface form. "Leo Tolstoy", "Tolstoy", "Leo Tolstoy (1869)", and "the Russian novelist Leo Tolstoy" all belong to the same cluster.
- Surface variations to ignore: capitalization, punctuation, articles (a/the), trailing dates or qualifiers, formal vs informal name forms, units that express the same physical quantity ("1.5 mM" = "0.0015 M"), and explanatory parentheticals.
- Different specific answers belong to different clusters even if related: "Tolstoy" and "Dostoevsky" are different; "Paris" and "France" are different; "1869" and "1865" are different.
- Each answer index must appear in exactly one cluster. Every index 0 to N-1 must be assigned.
- Use as few clusters as faithfully captures the semantic distinctions. Do not over-fragment over trivial differences.

Output format: a single JSON object with one field, "clusters", whose value is an array of cluster objects. Each cluster object has two fields:
- "label": a short canonical string representing the cluster's answer (e.g., "Leo Tolstoy", "1869")
- "members": a sorted array of integer indices (0-indexed) of the answers in that cluster

Example output:
{"clusters": [
  {"label": "Leo Tolstoy", "members": [0, 1, 3, 5, 8, 12, 15, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49]},
  {"label": "Fyodor Dostoevsky", "members": [2, 6, 11, 17, 24, 30, 38, 44]},
  {"label": "Anton Chekhov", "members": [4, 9, 14, 20, 27, 33, 41, 47]},
  {"label": "I don't know", "members": [7, 10, 13, 16, 18, 21, 23, 26, 29, 32, 35, 36, 39, 42, 45, 48]}
]}

Output ONLY the JSON object, with no preamble, no code fences, no commentary.
```

**User message template** (substitute `{question}` and the 50 extracted
answers):

```
Question: {question}

Answers (50 total, indexed 0 to 49):
[0] {answer_0}
[1] {answer_1}
[2] {answer_2}
[3] {answer_3}
... (continue for all 50)
[49] {answer_49}
```

Where each `{answer_i}` is the `answer_extracted` from Step 1 for
sample i.

For each provider, build one batch row per question with:
- `custom_id`: `{provider}__{question_id}` (e.g. `openai__simpleqa_00042`)
- `method`: `POST`
- `url`: `/v1/chat/completions`
- `body`:
  - `model`: `gpt-5.2`
  - `messages`: `[{"role": "system", "content": <SYSTEM>}, {"role": "user", "content": <USER>}]`
  - reasoning disabled parameter (whatever current syntax is)
  - `temperature`: 0.0 (deterministic clustering)
  - `max_tokens`: 1500
  - `response_format`: `{"type": "json_object"}` (force valid JSON)

If `gpt-5.2` does not accept `response_format=json_object` together
with reasoning-off, drop the `response_format` parameter and rely on
the system prompt's "Output ONLY the JSON object" instruction. In
that case Step 5 must be more forgiving in its JSON parsing
(strip code fences, leading/trailing whitespace).

Save batch input files to:
- `phase1/clustering/batch_inputs/openai_clustering.jsonl`
- `phase1/clustering/batch_inputs/anthropic_clustering.jsonl`
- `phase1/clustering/batch_inputs/gemini_clustering.jsonl`
- `phase1/clustering/batch_inputs/xai_clustering.jsonl`
- `phase1/clustering/batch_inputs/mistral_clustering.jsonl`

## Step 3 — Cost estimate BEFORE submission

Look up current GPT-5.2 batch pricing on OpenAI's pricing page.
Estimate per call:
- Input: ~2500 tokens (system 400 + user with 50 short answers ~2100)
- Output: ~500 tokens

Total: 21630 calls × ~3000 tokens combined.

Print per-provider estimate and grand total. **If estimated cost
exceeds $200, STOP and ask the user.**

## Step 4 — Submit batches in parallel

Submit all 5 OpenAI batches in parallel. Save batch IDs immediately to
`phase1/clustering/batch_ids.json`:

```json
{
  "openai":   {"batch_id": "...", "n_rows": 4326, "submitted_at": "..."},
  "anthropic":{"batch_id": "...", "n_rows": 4326, "submitted_at": "..."},
  "gemini":   {"batch_id": "...", "n_rows": 4326, "submitted_at": "..."},
  "xai":      {"batch_id": "...", "n_rows": 4326, "submitted_at": "..."},
  "mistral":  {"batch_id": "...", "n_rows": 4326, "submitted_at": "..."}
}
```

Poll each batch status every 5–10 minutes. Print progress per provider
as `provider: completed/total`. Wait until all 5 are in a terminal
state (completed, failed, or expired).

If any batch fails partway, save what was retrieved and report the
failure (provider name, error, completed count). Do not silently retry.

## Step 5 — Parse and unify outputs into canonical schema

For each provider, parse raw batch output and produce a unified
clustering JSONL file. Each row has this exact schema:

```json
{
  "id": "simpleqa_00042",
  "question": "Who wrote War and Peace?",
  "gold_answer": "Leo Tolstoy",
  "model": "openai",
  "n_samples": 50,
  "results": ["Answer: Leo Tolstoy\nConfidence: 95%", "...", "..."],
  "answer_extracted": ["Leo Tolstoy", "Tolstoy", "Leo Tolstoy", "..."],
  "extraction_method": ["answer_line", "answer_line", "fallback_tail", "..."],
  "confidence_verbalized": [0.95, 0.90, null, 0.85, "..."],
  "confidence_extraction_method": ["regex_match", "regex_match", "missing", "regex_match", "..."],
  "verbalized_mean": 0.872,
  "verbalized_n_extracted": 47,
  "n_clusters": 4,
  "cluster_labels": ["Leo Tolstoy", "Fyodor Dostoevsky", "Anton Chekhov", "I don't know"],
  "cluster_assignments": [0, 0, 1, 0, 2, 0, 3, 0, "..."],
  "cluster_sizes": [18, 8, 8, 16],
  "modal_cluster": 0,
  "modal_cluster_label": "Leo Tolstoy",
  "raw_judge_response": "..."
}
```

Field semantics:
- `n_samples`: always 50
- `results`: the original 50 raw generations from Step 1 (preserved lossless)
- `answer_extracted[i]`: the extracted short answer for sample i
- `extraction_method[i]`: `"answer_line"` or `"fallback_tail"` for sample i
- `confidence_verbalized[i]`: float in [0, 1] or `null` if regex did not match
- `confidence_extraction_method[i]`: `"regex_match"` or `"missing"`
- `verbalized_mean`: precomputed mean of non-null `confidence_verbalized` values, over all 50 samples (this is the verbalized baseline confidence for this question — it is the **mean over all 50 samples**, NOT only within the modal cluster, following Lin 2022 / Tian 2023 convention). Use `null` if zero samples have valid confidence.
- `verbalized_n_extracted`: number of samples i where `confidence_verbalized[i] != None` (denominator of `verbalized_mean`)
- `cluster_labels[k]`: the canonical label for cluster k (from judge output)
- `cluster_assignments[i]`: integer in [0, n_clusters-1] giving the cluster of sample i
- `cluster_sizes[k]`: number of samples in cluster k (must equal `len([i for i in cluster_assignments if i == k])`)
- `modal_cluster`: argmax of cluster_sizes
- `modal_cluster_label`: cluster_labels[modal_cluster]
- `raw_judge_response`: the full string returned by GPT-5.2 (for debug)

Parsing the judge response:
1. Parse `raw_judge_response` as JSON (strip code fences and whitespace if needed)
2. Extract the `clusters` array
3. For each cluster, record its label and members
4. Build `cluster_assignments` by inverting: for each sample i, find which cluster's `members` contains i
5. Verify every index 0..49 is assigned exactly once. If not, mark this row as a parse failure (see Step 6).

Save unified files to:
- `phase1/clustering/openai_simpleqa_clustered.jsonl`
- `phase1/clustering/anthropic_simpleqa_clustered.jsonl`
- `phase1/clustering/gemini_simpleqa_clustered.jsonl`
- `phase1/clustering/xai_simpleqa_clustered.jsonl`
- `phase1/clustering/mistral_simpleqa_clustered.jsonl`

## Step 6 — Sanity checks (REQUIRED)

For each provider, run and print:

1. **Row count**: must equal generation file row count (~4326). List
   any missing IDs.

2. **Parse failure rate**: rows where the judge response failed JSON
   parsing OR did not produce a valid partition of {0..49}. **Anything
   above 2% is a problem — report explicitly with example failures.**

3. **K (n_clusters) distribution**: histogram bucketed
   `K=1, 2, 3, 4, 5, 6-10, 11-20, >20`. Report mean and median K.
   Expected on SimpleQA: median K in [2, 5], mean in [2, 6]. Flag if
   mean K > 10 (prompt too strict) or mean K < 1.5 (model too
   confident — possible reasoning leakage from generation step).

4. **Singleton rate**: fraction of questions with K=1 (model gave
   semantically identical answer 50 times). Expected ~20-50% on
   SimpleQA.

5. **Coverage check**: for every row, verify
   `sum(cluster_sizes) == 50` and every index 0..49 appears in
   exactly one `members` list. List any violations.

6. **Modal mass distribution**: histogram of `max(cluster_sizes) / 50`
   (the naive confidence value). Should be roughly bimodal: peak
   near 1.0 (easy questions, K=1) and a spread in [0.2, 0.8]
   (hard questions).

7. **Print 5 random clusterings per provider** for visual inspection:
   the question, gold_answer, all 50 raw `results`, the
   `cluster_assignments`, and the `cluster_labels`. (25 examples
   total across providers.)

8. **Extraction method summary**: report fraction of `answer_line`
   vs `fallback_tail` per provider. Carry over from Step 1 for
   convenience.

9. **Verbalized confidence extraction rate**: per provider, fraction
   of samples where `confidence_verbalized != null`. Should be ≥ 90%
   for any provider whose verbalized baseline is going to appear in
   the main table. If < 90%, flag explicitly — that provider's
   verbalized baseline will be unreliable and downstream paper text
   needs to acknowledge "verbalized confidence baseline computed on
   N% of samples where extraction succeeded."

10. **Verbalized confidence distribution**: per provider, report mean
    and median of `verbalized_mean` across all questions, and a
    histogram with buckets [0.0–0.2, 0.2–0.4, 0.4–0.6, 0.6–0.8,
    0.8–1.0]. Sanity check: on SimpleQA most frontier models should
    have mean verbalized confidence > 0.5. If a provider's mean is
    < 0.3, the model is severely under-confident or extraction is
    buggy — flag explicitly. If > 0.95 across the board, the model is
    overconfident (which is informative — that's exactly the
    miscalibration our paper is about, but it should be reported).

## Step 7 — Cost report

Save `phase1/clustering_cost_report.md` with per-provider breakdown
(input tokens, output tokens, USD spent, batch discount applied) and
grand total. Print to stdout.

## Output structure

```
phase1/clustering/
├── batch_ids.json
├── batch_inputs/
│   ├── openai_clustering.jsonl
│   ├── anthropic_clustering.jsonl
│   ├── gemini_clustering.jsonl
│   ├── xai_clustering.jsonl
│   └── mistral_clustering.jsonl
├── openai_simpleqa_clustered.jsonl
├── anthropic_simpleqa_clustered.jsonl
├── gemini_simpleqa_clustered.jsonl
├── xai_simpleqa_clustered.jsonl
└── mistral_simpleqa_clustered.jsonl
phase1/clustering_cost_report.md
```

## Hard constraints

- API key from `phase1/.env` only. Never echo, never log, never write
  elsewhere.
- All randomness uses `seed=0`.
- Use OpenAI Batch API for ~50% discount.
- **Hard cost cap $200.** STOP and ask if estimate exceeds.
- Do NOT silently skip any provider — every failure must be reported.
- Do NOT modify the system or user prompt text from Step 2.
- Do NOT retry failed batches automatically.

## DO NOT in this step

- Do NOT compute SSSC, ECE, Brier, or any metric (next step)
- Do NOT do gold-answer grading (separate step)
- Do NOT propose follow-up steps or write Phase 2 plans
- Do NOT modify the clustering prompt or extraction logic

## After running, report

1. The 5 clustered JSONL files exist with correct schema (~4326 rows
   each, all required fields present)
2. Sanity check table per provider:
   - row count
   - parse failure rate
   - mean / median K
   - K distribution histogram
   - singleton rate
   - coverage violations (should be 0)
   - extraction method breakdown (answer_line / fallback_tail)
   - verbalized confidence extraction rate
   - verbalized confidence mean / median + distribution histogram
3. 5 random clustering examples per provider (25 total) for visual
   inspection
4. Cost report with per-provider breakdown and grand total
5. Status line: "READY FOR STEP 3 (grading)" only if all 5 providers
   passed:
   - parse failure rate ≤ 2%
   - 100% coverage
   - mean K in [1.5, 10]
   - no missing rows
   Otherwise list which providers failed which check and why.