# Phase 1 Step 3b — Grading: match modal cluster vs gold answer

## Context
Step 2 (clustering) produced 5 unified JSONL files at
`phase1/clustering/{provider}_simpleqa_clustered.jsonl`. Each row has a
`modal_cluster_label` (the model's most-frequent semantic answer) and a
`gold_answer`. We now use GPT-5.2 as a judge to grade whether the
modal cluster's answer matches the gold answer, producing Y ∈ {0, 1}
for downstream ECE / Brier computation.

This step does ONLY grading. Confidence values are already computed in
Step 3a; ECE / Brier metrics are computed in a later step that joins
this output with Step 3a's output.

## Configuration (FIXED)

| Parameter | Value |
|---|---|
| Judge model | `gpt-5.2` (non-reasoning mode) |
| Reasoning | DISABLED |
| API | OpenAI Batch API (24h window, ~50% discount) |
| Granularity | One grading call per (question, model) |
| Total calls | ~4297 questions × 5 models ≈ 21,500 |
| Hard cost cap | $30 |

## Step 0 — Verify GPT-5.2 access

Load `OPENAI_API_KEY` from `phase1/.env`. Send ONE non-batch test
request to `gpt-5.2` with reasoning disabled. Test prompt:

```
Question: Who wrote War and Peace?
Gold answer: Leo Tolstoy
Predicted answer: Tolstoy

Reply with one of: CORRECT, INCORRECT, NOT_ATTEMPTED
```

Expected response: `CORRECT`. Verify reasoning_tokens == 0.

If reasoning cannot be disabled, STOP and report.

## Step 1 — Build batch input files

For each provider, load `phase1/clustering/{provider}_simpleqa_clustered.jsonl`
and build one grading batch row per question.

**System message** (use VERBATIM):

```
You are a grader for a question-answering benchmark. You will receive a question, the official gold answer, and a predicted answer. Your job is to decide whether the predicted answer is semantically equivalent to the gold answer.

Rules:
- Accept different surface forms of the same answer: "Leo Tolstoy" matches "Tolstoy"; "1869" matches "1869 AD"; "Paris, France" matches "Paris"; "1.5 mM" matches "0.0015 M".
- Reject answers that are related but specifically different: "Tolstoy" vs "Dostoevsky" is INCORRECT; "1869" vs "1865" is INCORRECT; "Paris" vs "France" is INCORRECT.
- If the predicted answer is a refusal (e.g., "I don't know", "I cannot answer", "Unknown", "N/A", empty), output NOT_ATTEMPTED.
- Be lenient on capitalization, punctuation, articles (a/the), and minor formatting.

Output format: a single JSON object with one field, "grade", whose value is exactly one of the strings "CORRECT", "INCORRECT", or "NOT_ATTEMPTED".

Example outputs:
{"grade": "CORRECT"}
{"grade": "INCORRECT"}
{"grade": "NOT_ATTEMPTED"}

Output ONLY the JSON object, no preamble, no commentary.
```

**User message template**:

```
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {modal_cluster_label}
```

For each question, build a batch row:
- `custom_id`: `{provider}__{question_id}`
- `method`: `POST`
- `url`: `/v1/chat/completions`
- `body`:
  - `model`: `gpt-5.2`
  - `messages`: `[{"role": "system", "content": <SYSTEM>}, {"role": "user", "content": <USER>}]`
  - reasoning disabled
  - `temperature`: 0.0
  - `max_tokens`: 50
  - `response_format`: `{"type": "json_object"}` (drop if not supported)

Save batch input files to:
- `phase1/grading/batch_inputs/{provider}_grading.jsonl` (5 files)

## Step 2 — Cost estimate BEFORE submission

~21500 calls × ~200 input tokens + ~10 output tokens ≈ 4.5M total
tokens. GPT-5.2 batch pricing: estimate per provider and total.

**Hard cost cap $30.** STOP and ask if estimate exceeds.

## Step 3 — Submit batches in parallel

Submit all 5 OpenAI batches in parallel. Save batch IDs to
`phase1/grading/batch_ids.json`. Poll every 5–10 minutes until all
terminal. Report progress per provider.

## Step 4 — Parse and unify outputs

For each provider, parse raw batch output. Each row in unified output:

```json
{
  "id": "simpleqa_00042",
  "model": "openai",
  "question": "Who wrote War and Peace?",
  "gold_answer": "Leo Tolstoy",
  "modal_cluster_label": "Leo Tolstoy",
  "grade": "CORRECT",
  "Y": 1,
  "raw_judge_response": "{\"grade\": \"CORRECT\"}"
}
```

Field semantics:
- `grade`: parsed from judge JSON, one of `"CORRECT"`, `"INCORRECT"`, `"NOT_ATTEMPTED"`
- `Y`: 1 if `grade == "CORRECT"` else 0 (NOT_ATTEMPTED treated as Y=0 for calibration purposes — model that doesn't attempt should not be confident)
- If parse fails, mark `grade = "PARSE_ERROR"` and `Y = null`; do NOT skip the row

Save to:
- `phase1/grading/{provider}_simpleqa_graded.jsonl` (5 files)

## Step 5 — Sanity checks

For each provider, print:

1. **Row count**: must equal clustering file row count. List missing.

2. **Parse failure rate**: rows with `grade == "PARSE_ERROR"`. Should
   be < 1%. Flag if higher.

3. **Grade distribution**: fraction of CORRECT / INCORRECT /
   NOT_ATTEMPTED per provider. Sanity:
   - Frontier models on SimpleQA: expect CORRECT in [0.30, 0.60]
   - NOT_ATTEMPTED expected [0.05, 0.30]
   - INCORRECT = the rest
   - If CORRECT < 0.20 or > 0.80, flag (suggests grading bug or
     unexpected model behavior)

4. **Per-provider accuracy**: mean(Y). Print as a table:
   ```
   provider     n     correct   incorrect   not_attempted   accuracy
   openai       4297  XXXX      XXXX        XXXX            0.XXX
   ...
   ```

5. **Print 10 random graded examples per provider** (50 total) for
   visual inspection: question, gold, predicted, grade. This catches
   systematic grading bugs (e.g., judge being too strict or too
   lenient).

## Step 6 — Build unified flat table

Concatenate all 5 providers into a single table and save as
`phase1/grading/all_grades.parquet`. Columns: id, model, gold_answer,
modal_cluster_label, grade, Y.

This will be joined with `phase1/confidence/all_confidence.parquet` on
(id, model) in the next step (ECE / Brier computation).

## Step 7 — Cost report

Save `phase1/grading_cost_report.md` with per-provider breakdown and
grand total. Print to stdout.

## Output structure

```
phase1/grading/
├── batch_ids.json
├── batch_inputs/
│   └── {provider}_grading.jsonl  (5 files)
├── {provider}_simpleqa_graded.jsonl  (5 files)
└── all_grades.parquet
phase1/grading_cost_report.md
```

## Hard constraints

- API key from `phase1/.env` only
- All randomness uses `seed=0`
- Use OpenAI Batch API
- **Hard cost cap $30**
- Do NOT silently skip any provider
- Do NOT modify the grading prompt
- Do NOT retry failed batches automatically

## DO NOT

- Do NOT compute ECE, Brier, or any final metric (separate step)
- Do NOT modify the prompt or the Y mapping (CORRECT→1, else→0)
- Do NOT propose follow-up steps

## After running, report

1. The 5 graded JSONL files + the unified parquet exist
2. Sanity check table per provider:
   - row count
   - parse failure rate
   - grade distribution (CORRECT / INCORRECT / NOT_ATTEMPTED)
   - accuracy (= mean Y)
3. 10 random graded examples per provider (50 total)
4. Cost report
5. Status line: "READY FOR ECE COMPUTATION" only if all 5 providers
   passed (parse rate ≤ 1%, accuracy in [0.20, 0.80], no missing rows).
   Otherwise list which providers failed which check.
