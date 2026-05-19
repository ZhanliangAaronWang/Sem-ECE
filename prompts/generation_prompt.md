# Phase 1 — Generation across 5 frontier LLMs on SimpleQA

## Context

This is Step 1 of the Phase 1 main experiment for the Sample-Split
Semantic Consensus (SSSC) calibration paper. We need stochastic
generations from 5 frontier LLMs on the **full SimpleQA benchmark**,
which downstream steps will cluster, grade, and feed into the SSSC vs
naive comparison.

This prompt covers ONLY generation. Clustering, grading, and metric
computation are separate prompts that will follow once generation is
complete and validated.

The existing notebook `clustering_calibration.ipynb` already implements
OpenAI Batch API generation for HLE on a single model. Reuse that as
the template for the OpenAI part. The other 4 providers need separate
batch implementations following the per-provider notes below.

---

## Top-level configuration (FIXED)

| Parameter | Value |
|---|---|
| Dataset | SimpleQA (full set, all questions) |
| Sample budget | n = 50 generations per question |
| Temperature | 0.7 |
| top_p | 1.0 |
| max output tokens | 256 |
| Reasoning | **DISABLED on all models** |
| Hard cost cap | **$2000 total across 5 providers** |

If estimated cost exceeds $2000, STOP and ask the user before
submitting any batches.

---

## Step 0 — Set up `phase1/.env` and verify

**Do not proceed past this step until done.**

The user will provide API keys via a local file. Do NOT accept keys
pasted in chat. Create a template `phase1/.env.example` with these
fields and instruct the user to copy it to `phase1/.env` and fill in
real keys:

```
OPENAI_API_KEY=<your-openai-key>
ANTHROPIC_API_KEY=<your-anthropic-key>
GOOGLE_API_KEY=<your-google-key>
XAI_API_KEY=<your-xai-key>
MISTRAL_API_KEY=<your-mistral-key>
```

Add `phase1/.env` to `.gitignore` immediately. Load all keys via
`python-dotenv` or `os.environ`. **Never echo a key to stdout, never
write a key to any file other than `.env`, never log a key.**

If `phase1/.env` does not exist, STOP and tell the user to create it.

---

## Step 1 — Verify model names AND verify reasoning is OFF

For each provider, send ONE non-batch test request with the prompt
`What is 2+2? Answer with just the number.` (max output 10 tokens).

Use these candidate model name strings — these are the user's intended
models. If any model name resolves incorrectly, STOP and report the
exact API error; do not guess at alternative names. The user will
correct any wrong strings.

| Provider | Candidate model name | How to disable reasoning | Verification |
|---|---|---|---|
| OpenAI | `gpt-5.4` | Look up current OpenAI docs for the GPT-5 series reasoning control parameter (likely `reasoning_effort="minimal"` or similar). Use whatever the current API requires. | Inspect `response.usage.completion_tokens_details.reasoning_tokens` — must be 0. |
| Anthropic | `claude-opus-4-6` | Do NOT pass any `thinking` parameter. | Response must contain no content block of type `thinking`. |
| Google | `gemini-3-pro` | In `generation_config`, set `thinking_config={"thinking_budget": 0}` (verify exact field name in current Gemini API docs). | No thinking metadata in response. |
| xAI | `grok-4.20-0309-non-reasoning` | This is a dedicated non-reasoning model variant. No additional parameter needed. | Answer is "4" or contains "4". |
| Mistral | `mistral-large-latest` | Mistral Large is not a reasoning model by default. No action needed. | Answer is "4" or contains "4". |

Print each test response in full (truncating if very long) and pause
for user confirmation before proceeding. If any provider fails the
reasoning-off verification, STOP and report exactly what was tried.

---

## Step 1.5 — Verify `n` parameter support per provider

This is critical. Provider support for native multi-sample generation
varies; below is the verified state as of late 2025/early 2026:

| Provider | Native `n` support | Implication |
|---|---|---|
| OpenAI | ✅ Yes (chat completions) | One batch row per question with `n=50` |
| Anthropic | ❌ No | Must replicate: 50 batch rows per question |
| Google Gemini | ⚠️ Has `candidateCount` but it is empirically unreliable (returns fewer candidates than requested per known github bug) | Treat as no native support; replicate |
| xAI Grok | ⚠️ Not documented; OpenAI-compatible client | **Test:** send a request with `n=2` and confirm the response has 2 distinct choices. If yes, use native; if no, replicate. |
| Mistral | ⚠️ `n` exists on the API but Mistral docs explicitly state the latest Mistral Large model does NOT support `N completions` | Replicate |

For each of the providers requiring replication, build batch input
files with **(num_questions × 50)** total rows, each row having a
unique `custom_id` of the form `<question_id>__sample<sample_idx>`.

For native-`n` providers, build batch files with **(num_questions)**
rows and `n=50` in the body.

Save the verification results (which providers support native `n`) to
`phase1/n_support.json` so downstream steps know how to interpret
batch output structure.

---

## Step 2 — Load SimpleQA (full benchmark)

Load SimpleQA from HuggingFace. The standard mirror is
`basicv8vc/SimpleQA`; verify this and confirm column names. Typical
columns are `problem` and `answer` but verify against the actual
dataset.

Load the full test split with no subsampling. Print the total question
count for confirmation. Save canonical version to
`phase1/data/simpleqa_full.jsonl` with schema:

```json
{"id": "simpleqa_<5-digit zero-padded index>",
 "question": "<text>",
 "gold_answer": "<text>"}
```

Print total row count and first 3 records for visual inspection.

---

## Step 3 — Prompts (use EXACTLY this text for ALL 5 providers)

**System message** (do not modify, do not paraphrase):

```
You are answering questions for a research benchmark. Be concise and precise.

Your response MUST end with EXACTLY these two lines, with nothing after them:
Answer: <your final answer>
Confidence: <integer between 0 and 100>%

Keep <your final answer> as short as possible — ideally a single name, number, date, or short phrase. Do not put explanations or reasoning on the Answer line itself.

If you genuinely do not know, write:
Answer: I don't know
Confidence: 0%
```

**User message**: just the question text, nothing else (no prefix like
"Question:", no extra instructions).

How each provider takes a system message:
- OpenAI / Mistral / xAI: `messages=[{"role":"system",...},{"role":"user",...}]`
- Anthropic: separate top-level `system="..."` parameter
- Gemini: `system_instruction="..."` parameter (verify exact field name)

The prompt text is identical across all 5 providers; only the API
plumbing differs.

---

## Step 4 — Build batch input files (per provider)

For each provider, construct a batch input file appropriate to that
provider's batch API spec.

**Provider-specific notes:**

**OpenAI** — `/v1/batches` endpoint with `/v1/chat/completions` body.
Native `n=50`. One row per question. ~4326 rows total (assuming
SimpleQA test set size; verify after load).

**Anthropic** — Message Batches API
(`client.messages.batches.create(...)`, beta namespace as of 2025).
Replicate: 50 rows per question. Each row a separate message create
request. ~216k rows total.

**Google Gemini** — Gemini Batch API. Replicate: 50 rows per question.
~216k rows total. Check current Gemini batch API docs for the exact
batch input file format.

**xAI Grok** — Grok Batch API at
https://docs.x.ai/developers/advanced-api-usage/batch-api. The
endpoint is OpenAI-compatible; the OpenAI Python client pointed at
`https://api.x.ai/v1` should work. Use native `n=50` ONLY if Step 1.5
verified support; otherwise replicate.

**Mistral** — Check whether Mistral has a batch API in the current
docs. If yes, use it. If no, fall back to async parallel requests
with rate limiting and exponential backoff (do NOT submit serial
blocking requests for 200k+ calls). Either way, replicate.

Save batch input files to:
- `phase1/batch_inputs/openai_simpleqa.jsonl`
- `phase1/batch_inputs/anthropic_simpleqa.jsonl`
- `phase1/batch_inputs/gemini_simpleqa.jsonl`
- `phase1/batch_inputs/xai_simpleqa.jsonl`
- `phase1/batch_inputs/mistral_simpleqa.jsonl`

---

## Step 5 — Cost estimate BEFORE submitting any batch

For each provider, look up current published per-token pricing (find
each provider's pricing page). Print a table of the form:

```
provider          model                              input tok    output tok    est cost (batch)
OpenAI            gpt-5.4                            ~Xk          ~Yk           $A.AA
Anthropic         claude-opus-4-6                    ~Xk          ~Yk           $B.BB
Google            gemini-3-pro                       ~Xk          ~Yk           $C.CC
xAI               grok-4.20-0309-non-reasoning       ~Xk          ~Yk           $D.DD
Mistral           mistral-large-latest               ~Xk          ~Yk           $E.EE
                                                                                ──────
                                                                                $TOTAL
```

For replicate-mode providers, remember that input tokens are paid 50×
per question (the system + user prompt is sent in each of the 50
replicated requests). For OpenAI native `n=50`, input tokens are paid
once per question.

Apply the batch API discount where applicable (typically 50%).

**HARD COST CAP: $2000 total.** If the estimate exceeds, STOP and
ask the user before submitting any batch. Do not silently truncate or
sample to fit budget.

---

## Step 6 — Submit all batches in parallel and poll

Submit all 5 batches in parallel (do not block between submissions).
Save batch IDs and metadata to `phase1/batch_ids.json` immediately so
they can be recovered if the script crashes:

```json
{
  "openai":   {"batch_id": "...", "submitted_at": "...", "n_rows": ...},
  "anthropic":{"batch_id": "...", "submitted_at": "...", "n_rows": ...},
  ...
}
```

Then poll each provider's batch status periodically (every 5-10
minutes is fine). Print progress as `provider: completed/total` for
each. Wait until all 5 batches are in a terminal state (completed,
failed, or expired).

If any batch fails or expires partway through, save what was
retrieved and report the failure with the provider name, error
message, and number of completed rows. Do NOT silently retry.

---

## Step 7 — Parse and unify outputs into canonical schema

For each provider, convert the raw batch output into a unified JSONL
file. **All 5 unified files MUST have identical schema** so downstream
clustering and grading can treat them uniformly:

```json
{
  "id": "simpleqa_<5-digit>",
  "question": "<text>",
  "gold_answer": "<text>",
  "results": ["<gen 1>", "<gen 2>", "...", "<gen 50>"]
}
```

For replicate-mode providers, group rows by their question id portion
of `custom_id`, sort by sample_idx, and concatenate generations into
the `results` list.

For native-`n` providers (OpenAI, possibly xAI), extract all 50
choices from the single response object.

Save to:
- `phase1/generation/openai_simpleqa.jsonl`
- `phase1/generation/anthropic_simpleqa.jsonl`
- `phase1/generation/gemini_simpleqa.jsonl`
- `phase1/generation/xai_simpleqa.jsonl`
- `phase1/generation/mistral_simpleqa.jsonl`

---

## Step 8 — Sanity checks (REQUIRED, run for all 5 files)

For each provider's unified output file, run all of these checks and
print a clear summary table:

1. **Row count**: must equal SimpleQA full size (~4326). If not, list
   missing question IDs.

2. **Generations per row**: must be exactly 50 for all rows. List any
   rows with fewer generations (indicates failed samples).

3. **Visual inspection**: print 5 random generations from each
   provider (so 25 total). Each should end with `Answer: ...` and
   `Confidence: ...%`.

4. **Verbalized confidence extraction rate**: run regex
   `Confidence:\s*([0-9]+(?:\.[0-9]+)?)\s*%` on all generations per
   provider. Report success rate as `M / N`. **Anything below 90%
   means that provider's prompt format compliance is poor** — report
   it explicitly so the user can decide whether to re-prompt that
   model with stronger format instructions or accept that verbalized
   confidence baseline will be weaker for that model.

5. **Answer extraction rate**: run regex
   `Answer:\s*(.+?)(?=\nConfidence|\Z)` (with DOTALL flag) and
   report success rate per provider.

6. **Average generation length** in characters per provider — sanity
   check that 256 max_tokens isn't truncating most generations (if
   most generations hit exactly 256 chars at the cap, we have a
   truncation problem).

7. **Reasoning leakage check**: scan a random sample of 20
   generations per provider for any text suggesting visible
   reasoning ("Let me think...", "Step 1:", "First, I need to..."
   before the final Answer line). Report count.

---

## Step 9 — Cost report

Save `phase1/generation_cost_report.md` with per-provider breakdown:
prompt tokens, completion tokens, USD spent, batch discount applied
(yes/no), and grand total. Print to stdout as well.

---

## Output structure

```
phase1/
├── .env                       (user-provided, gitignored)
├── .env.example
├── n_support.json
├── batch_ids.json
├── data/
│   └── simpleqa_full.jsonl
├── batch_inputs/
│   ├── openai_simpleqa.jsonl
│   ├── anthropic_simpleqa.jsonl
│   ├── gemini_simpleqa.jsonl
│   ├── xai_simpleqa.jsonl
│   └── mistral_simpleqa.jsonl
├── generation/
│   ├── openai_simpleqa.jsonl
│   ├── anthropic_simpleqa.jsonl
│   ├── gemini_simpleqa.jsonl
│   ├── xai_simpleqa.jsonl
│   └── mistral_simpleqa.jsonl
└── generation_cost_report.md
```

---

## Hard constraints

- **API keys via `phase1/.env` only.** Do not hardcode, do not echo,
  do not log.
- **All randomness uses `seed=0`.**
- **Use Batch API where available** for ~50% discount.
- **Hard cost cap $2000.** Stop and ask if estimate exceeds.
- **Do not silently skip any provider.** Every failure must be
  reported.
- **Do not retry failed batches automatically** — report and let the
  user decide.

---

## DO NOT in this step

- Do NOT cluster the generations (that is the next step's job)
- Do NOT compute SSSC, ECE, Brier, or any metric
- Do NOT propose follow-up steps or write Phase 2 plans
- Do NOT silently subsample SimpleQA to fit budget — if it doesn't
  fit, ask the user
- Do NOT modify the generation prompt text from Step 3

---

## After running, report the following

1. **The 5 unified JSONL files** exist with correct schema (~4326
   rows × 50 generations each)
2. **Sanity check table** per provider (row count, gens per row,
   format compliance, extraction rates, length, reasoning leakage)
3. **20 random generations total** (4 per provider) for visual
   inspection
4. **Cost report** with per-provider breakdown and grand total
5. **Status line**: "READY FOR STEP 2 (clustering)" only if all 5
   providers passed sanity checks (row count exact, ≥90% format
   compliance, no reasoning leakage). Otherwise list which providers
   failed and why.
