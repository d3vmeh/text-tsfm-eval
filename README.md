# text-tsfm-eval

Tests whether an LLM forecasts better with text than without it, on the windows built by
`text-tsfm-benchmarks`. It follows the Direct Prompt method from the Context is Key benchmark.

Status: the prompt builder, reply parser, scorer and model runner are done and tested.

## Layout

This folder expects the dataset repo beside it:

    text-tsfm-benchmarks/   builds the dataset
    text-tsfm-eval/         this repo

## Pieces

- `prompt_builder.py`: makes two prompts per window, `with_context` and `no_context`. They differ only in the context block, and neither contains the hidden answers.
- `reply_parser.py`: turns a model reply into the forecast values, or gives a reason it can't be used.
- `model_runner.py`: sends each prompt to a local llama.cpp server, reads the reply with the parser and writes the forecasts file. It can be stopped and resumed, saves every raw reply, and records the settings used.
- `api_runner.py`: the same job as `model_runner.py`, for a hosted API that speaks the OpenAI-style chat format, such as OpenRouter. Same output layout, so the scorer works on both. It has a cost limit and reads the key from `OPENROUTER_API_KEY` or a `.env` file.
- `scorer.py`: scores forecasts against the hidden answers and compares the two conditions.
- `chronos_baseline.py`: forecasts the same windows with Chronos-2, a forecasting model that reads numbers only. A baseline, not part of the text comparison.
- `compare_forecasts.py`: puts several forecast sets side by side on exactly the same windows, for example the baseline next to an LLM run.
- `examples/`: one full prompt of each kind.
- `RUN_LOG.md`: how each run was made (server launch flags, settings), the runs so far, and what was learned. Read it before a new run.

## Run

    python3 test_prompt_builder.py
    python3 test_reply_parser.py
    python3 test_scorer.py
    python3 prompt_builder.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl --out prompts/AECI_validation_prompts.jsonl
    python3 test_model_runner.py
    python3 test_api_runner.py
    python3 model_runner.py prompts/AECI_validation_prompts.jsonl --run validation --temperature 1.0
    python3 scorer.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl results/validation/forecasts.jsonl --out results/validation/score

Needs Python 3 and numpy. `api_runner.py` also needs `requests`. `test_scorer.py` checks the scorer against the dataset's own baselines if the dataset has been built.

## Model runner

The runner needs a llama.cpp server that is already running. It does not start or change the server.

    export LLAMA_API_KEY=...     (only if the server was started with --api-key)
    python3 model_runner.py PROMPTS.jsonl --run NAME --temperature 1.0 [--base-url http://127.0.0.1:9999]

It writes `results/NAME/forecasts.jsonl` (one line per attempt), `results/NAME/settings.json` (model file, quantization,
context size, temperature, seed, max output tokens) and every raw reply in `replies/NAME/`.

- Stop it with Ctrl-C and run the same command to continue. Finished prompts are skipped. Resuming with different settings is refused.
- Before each request it counts the prompt's tokens and stops if prompt plus output does not fit the server's context window.
- A rejected reply is retried with a new sampling seed, up to `--max-attempts` (default 3). At temperature 0 a retry would give the same reply, so there is no retry.
- Both conditions get the same settings. Repetition penalties are switched off, because a forecast is mostly repeated digits.
- `--samples N` asks for N forecasts per prompt. With two or more, the scorer also reports CRPS. N can be raised later in the same run.
- `--server-note "..."` records how the server was started (its launch flags), since the server does not report them.
- `--workers N` keeps N requests in flight on one server. The server's slots share one context, so the runner checks that N requests fit together.
- `--shard 0/2` and `--shard 1/2` split the windows over two servers. Both prompts of a window always go to the same server. Use a different `--run` name per server, then join the two forecasts files with `cat` before scoring.
- Don't use the server for anything else during a run.

## Chronos-2 baseline

Chronos-2 needs PyTorch, which is kept in a virtual environment so the other scripts still need only numpy:

    python3 -m venv .venv
    .venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    .venv/bin/python -m pip install "chronos-forecasting>=2.0"
    .venv/bin/python test_baseline.py
    .venv/bin/python chronos_baseline.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl --run validation_chronos2
    python3 compare_forecasts.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl \
        chronos2=results/validation_chronos2/forecasts.jsonl:no_context \
        llm_no_text=results/validation/forecasts.jsonl:no_context \
        llm_with_text=results/validation/forecasts.jsonl:with_context

It runs on the CPU in a few seconds. The model gets the same past values the LLM prompt shows and nothing else, one window at a time.
`python3 test_baseline.py` works without PyTorch too; it then skips the one test that needs the real model.

## Forecasts file

The model runner writes one JSON line per attempt:

    {"name": "CarbonCastAECI", "seed": 20211001, "condition": "with_context", "sample": 0, "ok": true, "values": [72 numbers]}
    {"name": "CarbonCastAECI", "seed": 20211001, "condition": "no_context", "sample": 0, "ok": false, "reason": "missing 2 of 72 timestamps"}

## Credit

The prompt wording, the reply-reading rule and the CRPS estimator follow the Context is Key benchmark code,
https://github.com/ServiceNow/context-is-key-forecasting (Apache-2.0). Differences in the parser are marked `DIFF` in `reply_parser.py`.
