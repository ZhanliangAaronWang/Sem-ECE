# Manifest

274 source files, 72,039 lines, grouped by pipeline stage. Descriptions are the files' own module docstrings.


## 0. Utilities  (23 files, 6,282 lines)

| file | lines | what it does |
|---|---:|---|
| `auto_monitor.py` | 118 | Unified auto-monitor. |
| `auto_monitor.sbatch` | 17 | SBATCH --job-name=auto_mon |
| `build_xai_rerun.py` | 91 | Build xAI rerun batch inputs for truncated samples only. |
| `cluster_retry.py` | 136 | Retry clustering for questions that exited Step-2 with parse / partition |
| `cluster_retry_chain.py` | 403 | Cluster-retry chain runner. |
| `cluster_retry_chain.sbatch` | 14 | SBATCH --job-name=cluster_retry |
| `experiment_bootstrap_ci.py` | 226 | experiment_bootstrap_ci.py — paired bootstrap CIs for the Sem_1 vs Sem_2 gap. |
| `experiment_n_sweep.py` | 466 | experiment_n_sweep.py — n-sweep stress test of the local-law scaling. |
| `figure1.py` | 242 | figure1.py — Figure 1 of the Sem-ECE paper, two separate sub-figures. |
| `figure1a.py` | 117 | figure1a.py — Predicted gap E[ĉ_1] − E[ĉ_2] vs sample size n. |
| `figure3.py` | 369 | figure3.py — Sem-ECE stratified by per-question margin Δ_q. |
| `make_ece_figure.py` | 944 | make_ece_figure.py — sample-level calibration figures for the SCOPE paper. |
| `make_ece_vs_margin_figs.py` | 414 | make_ece_vs_margin_figs.py — per-(dataset, model) versions of figure3. |
| `make_scope_figures.py` | 1304 | make_scope_figures.py — SCOPE NeurIPS figures + tables generator. |
| `master_monitor.py` | 267 | Master monitor: tracks Phase 2 Gemini HLE + Phase 3 all providers. |
| `master_monitor.sbatch` | 16 | SBATCH --job-name=master_mon |
| `scan_truncation.py` | 110 | Scan all raw outputs for truncation indicators across phases × providers. |
| `scripts/layer2/compute_per_question.py` | 155 | Layer 2 Step 1 — compute per-question metrics (SCOPE-1/2, stability, margin, correctness). |
| `scripts/layer2/grade_unstable.py` | 137 | Grade z_hat_S cluster labels for questions where z_hat_S != z_hat_n. |
| `scripts/layer2/run_when_all_done.sbatch` | 23 | SBATCH --job-name=l2_final |
| `scripts/layer2/run_when_p4_done.sbatch` | 24 | SBATCH --job-name=l2_p4_rerun |
| `scripts/layer2/stratify_and_plot.py` | 223 | Layer 2 Steps 2-5: stratify, per-bin ECE, bootstrap CI, Figure 2. |
| `workflow_figure.py` | 466 | workflow_figure.py — pedagogical workflow illustration with a real HLE example. |

## 1. Generation — HLE  (50 files, 6,472 lines)

| file | lines | what it does |
|---|---:|---|
| `phase2/batch_ids_anthropic.json` | 24 |  |
| `phase2/batch_state.json` | 100 |  |
| `phase2/build_anthropic_hle.py` | 58 | Build Phase 2 (HLE) generation batch input for Anthropic claude-opus-4-6. |
| `phase2/build_anthropic_hle_retry.py` | 79 | Build Anthropic HLE retry batch input. |
| `phase2/build_batch_inputs.py` | 193 | Build Phase 2 (HLE) generation batch input files for 4 providers. |
| `phase2/build_xai_hle_retry.py` | 73 | Identify xAI HLE samples that failed or truncated (no Confidence: line) |
| `phase2/clustering/collect_anthropic.py` | 179 | Phase 2 (HLE) — Collect Anthropic clustering batch results, parse, validate. |
| `phase2/clustering/collect_gemini.py` | 241 | Phase 2 (HLE) Step 4 — Collect xAI clustering batch results, parse, validate. |
| `phase2/clustering/collect_mistral.py` | 247 | Phase 2 (HLE) Step 4 — Collect Mistral clustering batch results, parse, validate. |
| `phase2/clustering/collect_openai.py` | 247 | Phase 2 (HLE) Step 4 — Collect clustering batch results, parse, validate. |
| `phase2/clustering/collect_xai.py` | 247 | Phase 2 (HLE) Step 4 — Collect xAI clustering batch results, parse, validate. |
| `phase2/clustering/prep_anthropic.py` | 201 | Phase 2 (HLE) prep_anthropic — extract answers from anthropic_hle_raw.jsonl |
| `phase2/clustering/prep_gemini.py` | 213 | Phase 2 (HLE) Step 1 — Extract answers from Gemini batch/worker outputs and build clustering batch. |
| `phase2/clustering/prep_mistral.py` | 239 | Phase 2 (HLE) Step 1 — Extract answers from Mistral batch outputs and build clustering batch. |
| `phase2/clustering/prep_openai.py` | 215 | Phase 2 (HLE) Step 1 — Extract answers from OpenAI batch outputs and build clustering batch. |
| `phase2/clustering/prep_xai.py` | 239 | Phase 2 (HLE) Step 1 — Extract answers from xAI batch outputs and build clustering batch. |
| `phase2/clustering/retry_parse_failures.py` | 271 | Retry Anthropic clustering parse failures with seed=99. |
| `phase2/clustering/submit_anthropic_chunked.py` | 118 | Phase 2 (HLE) — Submit Anthropic clustering batch via OpenAI batch API, chunked. |
| `phase2/clustering/submit_gemini_chunked.py` | 119 | Phase 2 (HLE) Step 2 — Submit xAI clustering batch via OpenAI batch API, chunked. |
| `phase2/clustering/submit_mistral_chunked.py` | 164 | Phase 2 (HLE) Step 2-3 — Submit clustering batch for Mistral to OpenAI, chunked + poll. |
| `phase2/clustering/submit_openai_chunked.py` | 164 | Phase 2 (HLE) Step 2 — Submit clustering batch to OpenAI, chunked. |
| `phase2/clustering/submit_xai_chunked.py` | 164 | Phase 2 (HLE) Step 2 — Submit xAI clustering batch to OpenAI, chunked. |
| `phase2/gemini_batch_loop.py` | 175 | Sequential Gemini batch submission loop. |
| `phase2/gemini_batch_loop.sbatch` | 16 | SBATCH --job-name=gem_batch |
| `phase2/gemini_worker.py` | 173 | Async Gemini worker for HLE Phase 2 generation. |
| `phase2/gemini_worker_resume.py` | 191 | Async Gemini resume worker for HLE Phase 2 generation. |
| `phase2/gemini_workers.sbatch` | 20 | SBATCH --job-name=gemini_p2 |
| `phase2/gemini_workers_resume.sbatch` | 20 | SBATCH --job-name=gem_p2_r |
| `phase2/grading/batch_ids.json` | 32 |  |
| `phase2/grading/collect_and_check.py` | 328 | Phase 2 (HLE) Step 7 -- Collect grading batch results, parse, compute ECE. |
| `phase2/grading/prep_and_submit.py` | 134 | Phase 2 (HLE) Step 6 -- Grading: build batch input, estimate cost, submit. |
| `phase2/merge_anthropic_hle_retry.py` | 77 | Merge Anthropic HLE retry results into the main raw file. |
| `phase2/monitor.sbatch` | 16 | SBATCH --job-name=p2_monitor |
| `phase2/monitor_all.py` | 186 | Monitor all Phase 2 HLE generation batches and send email when all complete. |
| `phase2/submit_all.py` | 241 | Submit Phase 2 (HLE) generation batches for all 4 providers. |
| `phase2/submit_anthropic_hle.py` | 120 | Submit Phase 2 HLE Anthropic batch (107,900 rows -> 2 sub-batches). |
| `phase2/submit_anthropic_hle.sbatch` | 14 | SBATCH --job-name=p2_anth_hle |
| `phase2/submit_anthropic_hle_retry.py` | 118 | Submit Anthropic HLE retry batch (71,599 calls, max_tokens=1500). |
| `phase2/watch_chain_anthropic_fix.py` | 129 | Phase 2 HLE Anthropic fix-up chain. |
| `phase2/watch_chain_anthropic_fix.sbatch` | 15 | SBATCH --job-name=p2_chain_anth_fix |
| `phase2/watch_chain_anthropic_hle.py` | 122 | Phase 2 HLE Anthropic retry full chain. |
| `phase2/watch_chain_anthropic_hle.sbatch` | 15 | SBATCH --job-name=p2_chain_anth_hle |
| `phase2/watch_chain_p2.py` | 97 | Phase 2 (HLE) watcher: anthropic + xai downstream pipeline. |
| `phase2/watch_chain_p2.sbatch` | 15 | SBATCH --job-name=p2_chain |
| `phase2/watch_gemini_chain.py` | 58 | P2 HLE Gemini pipeline chain: poll clustering -> collect -> grade -> ECE. |
| `phase2/watch_gemini_chain.sbatch` | 16 | SBATCH --job-name=p2_gem_chain |
| `phase2/xai_worker.py` | 159 | Async xAI worker for HLE Phase 2 generation. |
| `phase2/xai_worker_retry3.py` | 157 | xAI HLE retry worker — reads from xai_hle_retry3.jsonl, writes to |
| `phase2/xai_workers.sbatch` | 17 | SBATCH --job-name=xai_p2 |
| `phase2/xai_workers_retry3.sbatch` | 16 | SBATCH --job-name=xai_p2_r3 |

## 1. Generation — MATH-500 / AIME  (34 files, 4,671 lines)

| file | lines | what it does |
|---|---:|---|
| `phase3/batch_state.json` | 95 |  |
| `phase3/build_batch_inputs.py` | 170 | Build Phase 3 (MATH-500 + AIME) generation batch input files for 4 providers. |
| `phase3/clustering/collect_gemini.py` | 241 | Phase 3 (MATH+AIME) Step 4 — Collect xAI clustering batch results, parse, validate. |
| `phase3/clustering/collect_mistral.py` | 241 | Phase 3 (MATH+AIME) Step 4 — Collect Mistral clustering batch results, parse, validate. |
| `phase3/clustering/collect_openai.py` | 242 | Phase 3 (MATH+AIME) Step 4 — Collect clustering batch results, parse, validate. |
| `phase3/clustering/collect_xai.py` | 241 | Phase 3 (MATH+AIME) Step 4 — Collect xAI clustering batch results, parse, validate. |
| `phase3/clustering/prep_gemini.py` | 213 | Phase 3 (MATH+AIME) Step 1 — Extract answers from Gemini batch/worker outputs and build clustering batch. |
| `phase3/clustering/prep_mistral.py` | 233 | Phase 3 (MATH+AIME) Step 1 — Extract answers from Mistral batch outputs and build clustering batch. |
| `phase3/clustering/prep_openai.py` | 220 | Phase 3 (MATH+AIME) Step 1 — Extract answers from OpenAI batch outputs and build clustering batch. |
| `phase3/clustering/prep_xai.py` | 222 | Phase 3 (MATH+AIME) Step 1 — Extract answers from xAI batch outputs and build clustering batch. |
| `phase3/clustering/submit_gemini_chunked.py` | 119 | Phase 3 (MATH+AIME) Step 2 — Submit xAI clustering batch via OpenAI batch API, chunked. |
| `phase3/clustering/submit_mistral_chunked.py` | 119 | Phase 3 (MATH+AIME) Step 2 — Submit Mistral clustering batch via OpenAI batch API, chunked. |
| `phase3/clustering/submit_openai_chunked.py` | 122 | Phase 3 (MATH+AIME) Step 2 — Submit clustering batch to OpenAI, chunked. |
| `phase3/clustering/submit_xai_chunked.py` | 119 | Phase 3 (MATH+AIME) Step 2 — Submit xAI clustering batch via OpenAI batch API, chunked. |
| `phase3/download_generation.py` | 169 | Phase 3 — Download completed generation batch outputs from OpenAI, xAI, and Mistral. |
| `phase3/gemini_batch_loop.py` | 169 | Sequential Gemini batch submission loop for Phase 3 (MATH+AIME). |
| `phase3/gemini_batch_loop.sbatch` | 16 | SBATCH --job-name=p3_gem |
| `phase3/gemini_worker.py` | 173 | Async Gemini worker for HLE Phase 2 generation. |
| `phase3/gemini_worker_resume.py` | 189 | Async Gemini resume worker for MATH+AIME Phase 3 generation. |
| `phase3/gemini_workers.sbatch` | 17 | SBATCH --job-name=gem_p3_w |
| `phase3/gemini_workers_resume.sbatch` | 20 | SBATCH --job-name=gem_p3_r |
| `phase3/grading/batch_ids.json` | 26 |  |
| `phase3/grading/collect_and_check.py` | 231 | Phase 3 (MATH-500 + AIME) Step 7 -- Collect grading batch results, parse, compute ECE. |
| `phase3/grading/prep_and_submit.py` | 139 | Phase 3 (MATH-500 + AIME 2024/2025) Step 6 -- Grading: build batch input, estimate cost, submit. |
| `phase3/monitor.sbatch` | 16 | SBATCH --job-name=p3_mon |
| `phase3/monitor_all.py` | 173 | Monitor all Phase 3 MATH+AIME generation batches. Sends email when done. |
| `phase3/preprocess.py` | 86 | Phase 3 preprocessing: MATH-500 + AIME 2024 + AIME 2025. |
| `phase3/submit_all.py` | 221 | Submit Phase 3 (MATH-500 + AIME) generation batches for all 4 providers. |
| `phase3/watch_clustering.py` | 155 | Phase 3 clustering watcher. |
| `phase3/watch_clustering.sbatch` | 19 | SBATCH --job-name=p3_clstw |
| `phase3/watch_gemini_chain.py` | 91 | Phase 3 Gemini one-shot pipeline watcher. |
| `phase3/watch_gemini_chain.sbatch` | 15 | SBATCH --job-name=gem_p3_chain |
| `phase3/watch_grading.py` | 130 | Phase 3 grading watcher. |
| `phase3/watch_grading.sbatch` | 19 | SBATCH --job-name=p3_gradw |

## 1. Generation — MedXpertQA  (15 files, 1,539 lines)

| file | lines | what it does |
|---|---:|---|
| `phase5/batch_state.json` | 58 |  |
| `phase5/build_batch_inputs.py` | 119 | Build MedXpertQA Text batch inputs for OpenAI — TWO protocols. |
| `phase5/clustering/collect_openai.py` | 242 | Phase 4 (PopQA) Step 4 — Collect clustering batch results, parse, validate. |
| `phase5/clustering/prep_openai.py` | 220 | Phase 4 (PopQA) Step 1 — Extract answers from OpenAI batch outputs and build clustering batch. |
| `phase5/clustering/submit_openai_chunked.py` | 122 | Phase 4 (PopQA) Step 2 — Submit clustering batch to OpenAI, chunked. |
| `phase5/finish_p5.py` | 35 | Wait for logit_v2 batch, download, run parse_logit, write .pipeline_done. |
| `phase5/finish_p5.sbatch` | 14 | SBATCH --job-name=p5_finish |
| `phase5/grading/batch_ids.json` | 8 |  |
| `phase5/grading/collect_and_check.py` | 231 | Phase 3 (MATH-500 + AIME) Step 7 -- Collect grading batch results, parse, compute ECE. |
| `phase5/grading/prep_and_submit.py` | 139 | Phase 3 (MATH-500 + AIME 2024/2025) Step 6 -- Grading: build batch input, estimate cost, submit. |
| `phase5/parse_logit.py` | 102 | Parse logit-based MCQA confidence from OpenAI batch output with top_logprobs. |
| `phase5/preprocess.py` | 54 | Phase 5 — MedXpertQA Text preprocessing. |
| `phase5/submit_all.py` | 70 | Submit Phase 5 (MedXpertQA Text) OpenAI batches — SCOPE + Logit protocols. |
| `phase5/watch_chain.py` | 109 | Phase 5 MedXpertQA full pipeline watcher. |
| `phase5/watch_chain.sbatch` | 16 | SBATCH --job-name=p5_chain |

## 1. Generation — PopQA  (43 files, 5,367 lines)

| file | lines | what it does |
|---|---:|---|
| `phase4/batch_ids_anthropic.json` | 17 |  |
| `phase4/build_anthropic_popqa.py` | 82 | Build Phase 4 (PopQA) generation batch input for Anthropic claude-opus-4-6. |
| `phase4/build_batch_inputs.py` | 128 | Phase 4 — PopQA generation batch inputs. |
| `phase4/build_batch_inputs_expand.py` | 113 | Phase 4 expansion — build batch inputs for 1500 NEW PopQA questions only. |
| `phase4/build_xai_popqa.py` | 56 | Build xAI PopQA generation batch input — was previously skipped, adding now. |
| `phase4/clustering/collect_anthropic.py` | 167 | Phase 4 (PopQA) — Collect Anthropic clustering batch results. |
| `phase4/clustering/collect_gemini.py` | 241 | Phase 4 (PopQA) Step 4 — Collect xAI clustering batch results, parse, validate. |
| `phase4/clustering/collect_mistral.py` | 241 | Phase 4 (PopQA) Step 4 — Collect Mistral clustering batch results, parse, validate. |
| `phase4/clustering/collect_openai.py` | 242 | Phase 4 (PopQA) Step 4 — Collect clustering batch results, parse, validate. |
| `phase4/clustering/collect_xai.py` | 241 | Phase 4 (PopQA) Step 4 — Collect xAI clustering batch results, parse, validate. |
| `phase4/clustering/prep_anthropic.py` | 198 | Phase 4 (PopQA) prep_anthropic — extract answers from anthropic_popqa_raw.jsonl |
| `phase4/clustering/prep_gemini.py` | 218 | Phase 4 (PopQA) Step 1 — Extract answers from Gemini batch/worker outputs and build clustering batch. |
| `phase4/clustering/prep_mistral.py` | 236 | Phase 4 (PopQA) Step 1 — Extract answers from Mistral batch outputs and build clustering batch. |
| `phase4/clustering/prep_openai.py` | 223 | Phase 4 (PopQA) Step 1 — Extract answers from OpenAI batch outputs and build clustering batch. |
| `phase4/clustering/prep_xai.py` | 228 | Phase 4 (PopQA) Step 1 — Extract answers from xAI batch outputs and build clustering batch. |
| `phase4/clustering/submit_anthropic_chunked.py` | 111 | Phase 4 (PopQA) — Submit Anthropic clustering batch via OpenAI batch API. |
| `phase4/clustering/submit_gemini_chunked.py` | 119 | Phase 4 (PopQA) Step 2 — Submit xAI clustering batch via OpenAI batch API, chunked. |
| `phase4/clustering/submit_mistral_chunked.py` | 119 | Phase 4 (PopQA) Step 2 — Submit Mistral clustering batch via OpenAI batch API, chunked. |
| `phase4/clustering/submit_openai_chunked.py` | 122 | Phase 4 (PopQA) Step 2 — Submit clustering batch to OpenAI, chunked. |
| `phase4/clustering/submit_xai_chunked.py` | 119 | Phase 4 (PopQA) Step 2 — Submit xAI clustering batch via OpenAI batch API, chunked. |
| `phase4/download_expand.py` | 167 | Phase 4 expansion — download generation outputs for OpenAI / Mistral / Gemini. |
| `phase4/finish_gemini.py` | 48 | Finish Gemini P4: poll clustering -> collect -> grade -> collect grade -> update marker. |
| `phase4/finish_gemini.sbatch` | 14 | SBATCH --job-name=p4_gem_fix |
| `phase4/grading/batch_ids.json` | 32 |  |
| `phase4/grading/collect_and_check.py` | 231 | Phase 3 (MATH-500 + AIME) Step 7 -- Collect grading batch results, parse, compute ECE. |
| `phase4/grading/prep_and_submit.py` | 139 | Phase 3 (MATH-500 + AIME 2024/2025) Step 6 -- Grading: build batch input, estimate cost, submit. |
| `phase4/preprocess.py` | 59 | Phase 4 — PopQA preprocessing. |
| `phase4/preprocess_expand.py` | 67 | Phase 4 expansion — sample 1500 additional PopQA questions (no overlap with existing 500). |
| `phase4/run_expand_chain.py` | 215 | Chain runner for the PopQA expansion pipeline. |
| `phase4/run_expand_chain.sbatch` | 14 | SBATCH --job-name=p4_expand_chain |
| `phase4/submit_all.py` | 220 | Submit Phase 4 (PopQA) generation batches for OpenAI / Gemini / Mistral (xAI blocked). |
| `phase4/submit_anthropic_popqa.py` | 120 | Submit Phase 4 PopQA Anthropic batch (25,000 rows on the 500-Q subset |
| `phase4/submit_expand.py` | 175 | Submit Phase 4 expansion batches (1500 NEW PopQA questions × 3 providers). |
| `phase4/watch_chain.py` | 83 | Phase 4 PopQA unified pipeline watcher. |
| `phase4/watch_chain.sbatch` | 15 | SBATCH --job-name=p4_chain |
| `phase4/watch_chain_anthropic.py` | 100 | Phase 4 PopQA Anthropic full chain. |
| `phase4/watch_chain_anthropic.sbatch` | 15 | SBATCH --job-name=p4_chain_anth |
| `phase4/watch_chain_xai.py` | 89 | Phase 4 PopQA xAI-only pipeline watcher. |
| `phase4/watch_chain_xai.sbatch` | 15 | SBATCH --job-name=p4_chain_xai |
| `phase4/watch_expand_chain.py` | 141 | P4 expansion watcher: |
| `phase4/watch_expand_chain.sbatch` | 14 | SBATCH --job-name=p4_expand |
| `phase4/xai_worker_popqa.py` | 157 | Async xAI worker for PopQA Phase 4 generation. |
| `phase4/xai_workers_popqa.sbatch` | 16 | SBATCH --job-name=xai_p4 |

## 1. Generation — SimpleQA  (51 files, 6,052 lines)

| file | lines | what it does |
|---|---:|---|
| `phase1/batch_ids.json` | 14 |  |
| `phase1/batch_state_mistral.json` | 34 |  |
| `phase1/batch_state_openai.json` | 45 |  |
| `phase1/build_anthropic_retry.py` | 55 | Build a retry batch input containing ONLY the custom_ids whose original |
| `phase1/build_batch_inputs.py` | 175 | Build batch input JSONL files for all 5 providers. |
| `phase1/clustering/batch_ids.json` | 149 |  |
| `phase1/clustering/collect_openai.py` | 308 | Steps 5+6+7 (OpenAI only): poll the clustering batch, parse output, run sanity checks. |
| `phase1/clustering/collect_provider.py` | 181 | Collect clustering batch results for any provider, parse into clustered.jsonl. |
| `phase1/clustering/prep_anthropic.py` | 197 | Step 1+2+3 for Anthropic clustering: extract answers from anthropic_raw.jsonl |
| `phase1/clustering/prep_gemini.py` | 192 | Steps 1+2+3 for Gemini clustering: extract answers from gemini_raw.jsonl, |
| `phase1/clustering/prep_mistral.py` | 191 | Steps 1+2+3 for Mistral clustering: extract answers from mistral_raw.jsonl, |
| `phase1/clustering/prep_openai.py` | 226 | Steps 1+2+3 for OpenAI clustering: extract answers, build batch file, print cost estimate. |
| `phase1/clustering/prep_xai.py` | 192 | Steps 1+2+3 for xAI clustering: extract answers from xai_raw.jsonl, |
| `phase1/clustering/step0_test_gpt52.py` | 68 | Step 0: verify gpt-5.2 access with reasoning DISABLED. |
| `phase1/clustering/submit_anthropic_chunked.py` | 171 | Submit Anthropic clustering as multiple smaller GPT-5.2 batches, sequentially. |
| `phase1/clustering/submit_gemini_chunked.py` | 171 | Submit Anthropic clustering as multiple smaller GPT-5.2 batches, sequentially. |
| `phase1/clustering/submit_openai.py` | 53 | Step 4 (OpenAI only): upload openai_clustering.jsonl and start the batch. |
| `phase1/clustering/submit_openai_chunked.py` | 161 | Submit OpenAI clustering as multiple smaller batches, sequentially. |
| `phase1/gemini_async.py` | 197 | Async parallel Gemini requests for SimpleQA generation. |
| `phase1/gemini_async.sbatch` | 20 | SBATCH --job-name=gemini_async |
| `phase1/gemini_batch_collect.py` | 149 | Poll all jobs in gemini_batch_jobs.json and append results to gemini_raw.jsonl. |
| `phase1/gemini_batch_collect_v2.py` | 134 | Collect completed Gemini batch results and append to gemini_raw.jsonl. |
| `phase1/gemini_batch_submit.py` | 136 | Submit remaining Gemini SimpleQA rows via the Gemini Batch API. |
| `phase1/gemini_batch_test.py` | 79 | Tiny test: submit 5 inlined requests via Gemini Batch API and poll until done. |
| `phase1/gemini_worker.py` | 200 | Parallel-worker variant of gemini_async.py. |
| `phase1/gemini_workers.sbatch` | 34 | SBATCH --job-name=gemini_w |
| `phase1/grading/batch_ids.json` | 31 |  |
| `phase1/grading/collect_and_check.py` | 217 | Phase 1 Step 3b — Collect grading batch results, parse, sanity check. |
| `phase1/grading/prep_and_submit.py` | 140 | Phase 1 Step 3b — Grading: build batch input, estimate cost, submit. |
| `phase1/merge_anthropic_retry.py` | 79 | Merge retry results into anthropic_raw.jsonl, replacing the errored |
| `phase1/merge_gemini.py` | 69 | Merge all gemini_raw_w*.jsonl worker files into the master gemini_raw.jsonl. |
| `phase1/monitor_all.py` | 184 | Monitor all 5 provider jobs and send email when all are done. |
| `phase1/monitor_all.sbatch` | 16 | SBATCH --job-name=monitor_p1 |
| `phase1/n_support.json` | 37 |  |
| `phase1/recover_mistral.py` | 132 | Recover Mistral results. |
| `phase1/retry_openai.py` | 88 | Retry the 112 OpenAI rows that sub-batch 7 silently dropped. |
| `phase1/run_anthropic_pipeline.py` | 84 | Sequential downstream pipeline for SimpleQA Anthropic (claude-opus-4-6). |
| `phase1/run_anthropic_pipeline.sbatch` | 14 | SBATCH --job-name=p1_anth_pipe |
| `phase1/run_simpleqa_gemini_chain.py` | 276 | SimpleQA Gemini backfill chain. |
| `phase1/run_simpleqa_gemini_chain.sbatch` | 14 | SBATCH --job-name=p1_gem_chain |
| `phase1/step1_results.json` | 53 |  |
| `phase1/step1_verify_models.py` | 156 | Step 1: verify each of the 5 provider model names and that reasoning is OFF. |
| `phase1/submit_anthropic.py` | 96 | Submit Anthropic Message Batch and poll until complete. |
| `phase1/submit_anthropic_retry.py` | 110 | Submit retry batch for the 72,060 SimpleQA samples that errored with |
| `phase1/submit_anthropic_retry.sbatch` | 14 | SBATCH --job-name=p1_anth_retry |
| `phase1/submit_gemini.py` | 150 | Submit Gemini batch job(s) and poll until complete. |
| `phase1/submit_mistral.py` | 105 | Submit Mistral batch job(s) from split files and poll until complete. |
| `phase1/submit_openai.py` | 135 | Submit OpenAI batch in sequential sub-batches. |
| `phase1/submit_xai.py` | 124 | Submit xAI batch and poll until complete. |
| `phase1/xai_async.py` | 176 | Async parallel xAI requests for SimpleQA generation. |
| `phase1/xai_async.sbatch` | 20 | SBATCH --job-name=xai_async |

## 1. Generation — xAI rerun  (3 files, 236 lines)

| file | lines | what it does |
|---|---:|---|
| `xai_rerun/submit.py` | 112 | Submit xAI rerun batches for P2 and P3 truncated samples. |
| `xai_rerun/watch.py` | 107 | Poll xAI rerun batches; download outputs as they complete. |
| `xai_rerun/watch.sbatch` | 17 | SBATCH --job-name=xai_watch |

## 2. Clustering — retry  (1 files, 7 lines)

| file | lines | what it does |
|---|---:|---|
| `cluster_retry/batch_state.json` | 7 |  |

## 2. Extended budget (n=100)  (16 files, 1,908 lines)

| file | lines | what it does |
|---|---:|---|
| `n100/analyze_convergence.py` | 158 | Exp 3 preview: n-sweep convergence + new/old drift check on n=100 clusters. |
| `n100/build_gen_credit.py` | 67 | Build n=100 extension generation inputs for anthropic + xai (credit-gated). |
| `n100/build_gen_ds.py` | 107 | Build +50-sample generation inputs for HLE (phase2) and PopQA (phase4). |
| `n100/build_gen_inputs.py` | 93 | Exp 3 pilot — build +50-sample generation inputs for SimpleQA (n: 50→100). |
| `n100/collect_anthropic_gen.py` | 67 | Poll anthropic n=100 generation batches -> {ds}_anthropic_raw.jsonl ({custom_id,text}). |
| `n100/collect_cluster.py` | 128 | Poll n=100 clustering batches (gpt-5.2), parse into clustered100 jsonl. |
| `n100/collect_gen.py` | 174 | Poll the n=100 generation batches (openai/gemini/mistral), collect raw samples. |
| `n100/figure_convergence.py` | 61 | Exp 3 figure: 3-panel n-sweep convergence (Sem1 & Sem2 vs n) + gap log-log slope. |
| `n100/finish_simpleqa_anthropic.py` | 123 | Finish the last n=100 cell: SimpleQA × Anthropic. |
| `n100/full_table.py` | 404 | Exp 3 — complete n=10…100 results, every benchmark x provider cell. |
| `n100/merge_and_build_cluster_inputs.py` | 126 | Merge old 50 + new 50 samples → n=100; build GPT-5.2 clustering batch inputs. |
| `n100/orchestrate.py` | 83 | Auto-advance the n=100 pipeline as generation completes, per dataset. |
| `n100/submit_all.py` | 118 | Submit n=100 extension generation batches for openai / gemini / mistral. |
| `n100/submit_anthropic_gen.py` | 44 | Submit anthropic n=100 generation via Message Batches (all datasets). |
| `n100/submit_cluster.py` | 51 | Submit n=100 clustering batches (gpt-5.2), one per provider, dataset-aware. |
| `n100/xai_async_gen.py` | 104 | Async xAI n=100 generation (grok-4.20-0309-non-reasoning), per dataset. |

## 3. Decoding sensitivity  (5 files, 1,528 lines)

| file | lines | what it does |
|---|---:|---|
| `temp_sweep/analyze.py` | 233 | Exp 8 — analysis: margin distribution, ECE with CIs, gap below/above the JDR boundary. |
| `temp_sweep/build.py` | 82 | Exp 8 (R6b) — temperature / sampling sensitivity. Build subset + generation inputs. |
| `temp_sweep/run_cluster_grade.py` | 301 | Exp 8 — cluster + grade the temperature arms with the ORIGINAL judge. |
| `temp_sweep/run_gen.py` | 105 | Exp 8 — submit / collect the three temperature-arm generation batches (OpenAI batch API). |
| `temp_sweep/subset_ids.json` | 807 |  |

## 4. Cross-judge — Claude  (2 files, 228 lines)

| file | lines | what it does |
|---|---:|---|
| `cross_judge_claude/collect.py` | 159 | Poll Claude judge batches, parse into clustered jsonl (dataset-cell files). |
| `cross_judge_claude/submit.py` | 69 | Submit cross-judge clustering to claude-opus-4-6 via Anthropic Message Batches. |

## 4. Cross-judge — Gemini  (6 files, 878 lines)

| file | lines | what it does |
|---|---:|---|
| `cross_judge/build_inputs.py` | 99 | Build Gemini cross-judge batch inputs from the paper's 15 clustered cells. |
| `cross_judge/collect.py` | 265 | Collect Gemini cross-judge batch results and parse into clustered jsonl. |
| `cross_judge/compare.py` | 211 | Compare Gemini-3.1-Pro judge vs GPT-5.2 judge (rebuttal robustness study). |
| `cross_judge/retry_failures.py` | 99 | Retry residual parse failures with live calls (seed=SEED, larger output cap). |
| `cross_judge/smoke_test.py` | 90 | Live smoke test of gemini-3.1-pro-preview as clustering judge. |
| `cross_judge/submit.py` | 114 | Submit cross-judge clustering rows as Gemini inline batch jobs. |

## 4. Cross-judge — Kimi  (2 files, 198 lines)

| file | lines | what it does |
|---|---:|---|
| `cross_judge_kimi/parse.py` | 85 | Parse Kimi-K3 judge raw.jsonl -> clustering/{phase}_{provider}_kimik3_clustered.jsonl. |
| `cross_judge_kimi/run_judge.py` | 113 | Cross-judge clustering with Kimi-K3 via QUBRID (OpenAI-compatible, async). |

## 4. Cross-judge — Mistral (dropped)  (5 files, 417 lines)

| file | lines | what it does |
|---|---:|---|
| `cross_judge_mistral/build_inputs.py` | 45 | Convert cross_judge/batch_inputs (Gemini format) into Mistral batch chunks. |
| `cross_judge_mistral/collect.py` | 172 | Poll Mistral batch jobs, download outputs, parse into clustered jsonl. |
| `cross_judge_mistral/retry_failures.py` | 76 | Retry residual Mistral-judge parse failures with live calls (new random_seed). |
| `cross_judge_mistral/smoke_test.py` | 68 | Live smoke test of mistral-large-latest (Large 3) as clustering judge. |
| `cross_judge_mistral/submit.py` | 56 | Upload chunks and create Mistral batch jobs (mistral-large-latest = Large 3). |

## 5. Clustering-prompt sensitivity  (5 files, 32,461 lines)

| file | lines | what it does |
|---|---:|---|
| `prompt_sens/build.py` | 130 | Exp 10 — clustering-prompt sensitivity (R-comment: not just temperature). |
| `prompt_sens/case_studies.py` | 123 | Find concrete questions where the MERGE-leaning and SPLIT-leaning clustering prompts |
| `prompt_sens/collect_analyze.py` | 210 | Exp 10 — collect the prompt-variant partitions and compare them to the paper's. |
| `prompt_sens/design.json` | 31820 |  |
| `prompt_sens/strict_loose_diff.py` | 178 | Every question in the 600-question prompt-sensitivity subset where the STRICT |

## 6. Analysis — baselines & budget  (3 files, 644 lines)

| file | lines | what it does |
|---|---:|---|
| `baselines/budget_sweep.py` | 225 | Exp 7 — total-budget sweep B ∈ {5,10,20,30,50}. Pure local, no API calls. |
| `baselines/figure_budget_sweep.py` | 119 | Exp 7 figure — budget sweep B ∈ {5,10,20,30,50}. |
| `baselines/semantic_entropy.py` | 300 | R4 semantic-entropy baseline — pure local analysis of the paper's n=50 samples. |

## 6. Analysis — cluster audit  (2 files, 366 lines)

| file | lines | what it does |
|---|---:|---|
| `audit_cluster/build.py` | 158 | Cluster-level audit — is each cluster produced by the judge internally coherent? |
| `audit_cluster/score.py` | 208 | Score the cluster-level audit. |

## 6. Analysis — pair audit  (2 files, 815 lines)

| file | lines | what it does |
|---|---:|---|
| `audit/build_audit_sheets.py` | 342 | Phase 3 / R2 — human-audit sheet generator (build now, run after the two-judge comparison). |
| `audit/score_audit.py` | 473 | Phase 3 / R2 — scoring for the human audit sheets. |

## 6. Analysis — structural ambiguity  (3 files, 565 lines)

| file | lines | what it does |
|---|---:|---|
| `hedged/impact.py` | 160 | Bound the effect of hedged / approximate answers on the evaluation. |
| `hedged/metric_stability.py` | 226 | Is Sem-ECE itself — the metric, not the Sem1-Sem2 gap — robust to the partition |
| `hedged/prevalence.py` | 179 | How common are hedged / approximate answers, and how does the judge treat them? |

## 7. Rebuttal deliverables  (3 files, 1,405 lines)

| file | lines | what it does |
|---|---:|---|
| `rebuttal/exp3_n100_convergence/full_table.py` | 404 | Exp 3 — complete n=10…100 results, every benchmark x provider cell. |
| `rebuttal/exp8_temperature_sensitivity/subset_ids.json` | 807 |  |
| `rebuttal/make_judge_summary.py` | 194 | Generate the consolidated CROSS-JUDGE results document (judges only). |
