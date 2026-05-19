# `data/` directory

Place per-phase artifacts here:

```
data/
├── phase1/   # SimpleQA
│   ├── clustering/<provider>_simpleqa_clustered.jsonl
│   ├── grading/<provider>_simpleqa_graded.jsonl
│   └── confidence/<provider>_confidence.jsonl   (produced by compute_confidence.py)
├── phase2/   # HLE
│   ├── clustering/<provider>_simpleqa_clustered.jsonl   (legacy filename suffix)
│   ├── grading/<provider>_hle_graded.jsonl
│   └── confidence/<provider>_confidence.jsonl
└── phase4/   # PopQA
    ├── clustering/<provider>_popqa_clustered.jsonl
    ├── grading/<provider>_popqa_graded.jsonl
    └── confidence/<provider>_confidence.jsonl
```

`<provider>` is one of `openai`, `anthropic`, `gemini`, `xai`, `mistral`.

See top-level `README.md` §3 for the JSONL field schemas and links to the
public benchmarks (SimpleQA / HLE / PopQA).
