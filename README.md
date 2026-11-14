# text-tsfm-eval

Tests whether an LLM forecasts better with text than without it, on the windows built by
`text-tsfm-benchmarks`. It follows the Direct Prompt method from the Context is Key benchmark.

Status: the prompt builder, reply parser and scorer are done and tested. The model runner is not built yet.

## Layout

This folder expects the dataset repo beside it:

    text-tsfm-benchmarks/   builds the dataset
    text-tsfm-eval/         this repo

## Pieces

- `prompt_builder.py`: makes two prompts per window, `with_context` and `no_context`. They differ only in the context block, and neither contains the hidden answers.
- `reply_parser.py`: turns a model reply into the forecast values, or gives a reason it can't be used.
- `scorer.py`: scores forecasts against the hidden answers and compares the two conditions.
- `examples/`: one full prompt of each kind.

## Run

    python3 test_reply_parser.py
    python3 test_scorer.py
    python3 prompt_builder.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl --out prompts/AECI_validation_prompts.jsonl
    python3 scorer.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl FORECASTS.jsonl --out results/validation

Needs Python 3 and numpy. `test_scorer.py` checks the scorer against the dataset's own baselines if the dataset has been built.

## Forecasts file

The model runner writes one JSON line per attempt:

    {"name": "CarbonCastAECI", "seed": 20211001, "condition": "with_context", "sample": 0, "ok": true, "values": [72 numbers]}
    {"name": "CarbonCastAECI", "seed": 20211001, "condition": "no_context", "sample": 0, "ok": false, "reason": "missing 2 of 72 timestamps"}

## Credit

The prompt wording, the reply-reading rule and the CRPS estimator follow the Context is Key benchmark code,
https://github.com/ServiceNow/context-is-key-forecasting (Apache-2.0). Differences in the parser are marked `DIFF` in `reply_parser.py`.
