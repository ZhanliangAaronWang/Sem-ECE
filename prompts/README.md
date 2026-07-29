# Prompts

Every prompt used anywhere in the pipeline, extracted from the Python string constants it is defined in and written out verbatim. Prompts are deduplicated by content: one file per distinct prompt, listing every site that uses it.

This matters for the paper's robustness claims — the cross-judge experiments assert a *byte-identical* clustering prompt across judges, and the table below is what makes that checkable.

## What the deduplication verifies

- **One clustering prompt, 20 sites.** The reference clustering prompt (`system_prompt_31cffa792e.txt`) hashes identically across all five providers in phases 1-5 *and* all four cross-judge directories. The claim that Gemini, Claude and Kimi were given the same instruction as gpt-5.2 is verified by construction here, not asserted.

- **The `original` and `shuffled` prompt-sensitivity arms hash to that same prompt.** `shuffled` permutes the order in which answers are presented, not the instruction, so an identical prompt hash is the expected result and confirms the arm was built as described.

- **The parse-failure retry used a different prompt** (`system_prompt_35284bfc49.txt`, 1,895 chars against the reference's 1,950). It was applied only to questions whose first partition failed to parse. We note it because it is a genuine deviation from the single-prompt story.


## clustering  (6 distinct)

| file | chars | defined as | used in |
|---|---:|---|---|
| [`clustering/arm_strict_cbda25e8ca.txt`](clustering/arm_strict_cbda25e8ca.txt) | 1,992 | `arm:strict` | `prompt_sens/design.json`:arm:strict |
| [`clustering/arm_loose_c983e440cd.txt`](clustering/arm_loose_c983e440cd.txt) | 1,972 | `arm:loose` | `prompt_sens/design.json`:arm:loose |
| [`clustering/system_prompt_31cffa792e.txt`](clustering/system_prompt_31cffa792e.txt) | 1,950 | `SYSTEM_PROMPT` | `phase1/clustering/prep_anthropic.py`:SYSTEM_PROMPT; `phase1/clustering/prep_gemini.py`:SYSTEM_PROMPT; `phase1/clustering/prep_mistral.py`:SYSTEM_PROMPT; `phase1/clustering/prep_openai.py`:SYSTEM_PROMPT; `phase1/clustering/prep_xai.py`:SYSTEM_PROMPT; `phase2/clustering/prep_anthropic.py`:SYSTEM_PROMPT; +16 more |
| [`clustering/system_prompt_35284bfc49.txt`](clustering/system_prompt_35284bfc49.txt) | 1,895 | `SYSTEM_PROMPT` | `phase2/clustering/retry_parse_failures.py`:SYSTEM_PROMPT |
| [`clustering/arm_no_example_16add8bbbe.txt`](clustering/arm_no_example_16add8bbbe.txt) | 1,554 | `arm:no_example` | `prompt_sens/design.json`:arm:no_example |
| [`clustering/arm_minimal_2baf3753dc.txt`](clustering/arm_minimal_2baf3753dc.txt) | 393 | `arm:minimal` | `prompt_sens/design.json`:arm:minimal |

## generation  (3 distinct)

| file | chars | defined as | used in |
|---|---:|---|---|
| [`generation/system_fddf0526bd.txt`](generation/system_fddf0526bd.txt) | 466 | `SYSTEM` | `phase1/build_batch_inputs.py`:SYSTEM; `phase2/build_anthropic_hle.py`:SYSTEM_PROMPT; `phase2/build_batch_inputs.py`:SYSTEM_PROMPT; `phase3/build_batch_inputs.py`:SYSTEM_PROMPT; `phase4/build_anthropic_popqa.py`:SYSTEM_PROMPT; `phase4/build_batch_inputs.py`:SYSTEM_PROMPT; +2 more |
| [`generation/scope_system_14bcae21c5.txt`](generation/scope_system_14bcae21c5.txt) | 346 | `SCOPE_SYSTEM` | `phase5/build_batch_inputs.py`:SCOPE_SYSTEM |
| [`generation/logit_system_7a86ee9a8e.txt`](generation/logit_system_7a86ee9a8e.txt) | 199 | `LOGIT_SYSTEM` | `phase5/build_batch_inputs.py`:LOGIT_SYSTEM |

## grading  (3 distinct)

| file | chars | defined as | used in |
|---|---:|---|---|
| [`grading/system_prompt_d559474b2b.txt`](grading/system_prompt_d559474b2b.txt) | 1,087 | `SYSTEM_PROMPT` | `phase1/grading/prep_and_submit.py`:SYSTEM_PROMPT; `phase2/grading/prep_and_submit.py`:SYSTEM_PROMPT; `phase3/grading/prep_and_submit.py`:SYSTEM_PROMPT; `phase4/grading/prep_and_submit.py`:SYSTEM_PROMPT; `phase5/grading/prep_and_submit.py`:SYSTEM_PROMPT |
| [`grading/grade_system_prompt_97aedd4109.txt`](grading/grade_system_prompt_97aedd4109.txt) | 998 | `GRADE_SYSTEM_PROMPT` | `cluster_retry_chain.py`:GRADE_SYSTEM_PROMPT |
| [`grading/system_prompt_8eb237931f.txt`](grading/system_prompt_8eb237931f.txt) | 833 | `SYSTEM_PROMPT` | `scripts/layer2/grade_unstable.py`:SYSTEM_PROMPT |
