# Run log

How the model runner was built, how it works, how each run was made, and what was learned.
Run outputs live in `results/` and `replies/`, which git ignores, so this file is the lasting record.

Last updated: 2026-11-14, while the two validation runs were still going. Section 6 says which numbers are final.

## 1. What the runner does

`model_runner.py` takes a prompts file from `prompt_builder.py` and, for each prompt:

1. Counts the prompt's tokens on the server (`/tokenize`).
2. Stops the run if prompt + 256 (chat template) + max output tokens does not fit the server's context. Nothing is sent.
3. Sends the messages to `/v1/chat/completions` with the run's settings.
4. Saves the raw reply in `replies/RUN/`: a `.json` (the server's full answer), a `.txt` (the text given to the parser) and a `.thinking.txt` (the model's thinking, if any).
5. Checks that the server read the whole prompt (its reported prompt tokens must not be less than the count from step 1).
6. Gives the reply text to `parse_reply()` and writes one line to `results/RUN/forecasts.jsonl`.
7. If the reply was rejected, asks again with a new sampling seed, up to `--max-attempts` (default 3).

Things worth knowing:

- **One line per attempt.** A prompt that needed three attempts has three lines. `scorer.py` counts every rejected line as a failed reply and uses the accepted one.
- **Two different seeds.** `seed` in each line is the window's id from the records (for example 20211001). The sampling seed sent to the server is `request_seed` = `--seed` + 1000 x sample + attempt.
- **Same settings for both conditions.** Every request carries the same settings. Only `messages` differs. A test checks this.
- **Penalties are off.** The runner sends `repeat_penalty 1.0`, `presence_penalty 0`, `frequency_penalty 0`, because a forecast is mostly repeated digits. It also sends `cache_prompt false` so a request does not depend on the one before it.
- **`top_p` and `top_k` are left to the server** unless given. The server's values are saved in `settings.json` under `server_default_sampling` (0.95 and 20 in the runs below).
- **Server trouble is not a model failure.** If the server is down, loading, out of memory or times out, the runner waits 10, 30, 60, 120 and 300 seconds, tries again each time, then stops. Nothing is written to the forecasts file for it.
- **Stop and resume.** Each attempt is written and flushed to disk at once. Run the same command again and finished prompts are skipped. A half-written last line is moved to `forecasts.jsonl.partial`.
- **Resuming with different settings is refused.** Model file, context size, server build, chat template, temperature, seed, max output tokens, max attempts, top_p, top_k, the penalties, the prompts file and the server note must all match. `--samples` may be raised later.
- **At temperature 0 a rejected reply is not retried**, because the retry would be identical.
- **`--samples N`** asks for N forecasts per prompt. Sample 0 is finished for every prompt before sample 1 starts.
- **`--workers N`** keeps N requests in flight on one server. **`--shard I/N`** gives one server every N-th window; both prompts of a window always stay together.
- **`--server-note`** records how the server was started. The server does not report its launch flags (thinking cap, GPU), so they only reach `settings.json` through this note.
- **The API key** is read from the `LLAMA_API_KEY` environment variable and is never written to any file.

`settings.json` holds: model file, quantization (from the file name, and separately what the server reports), context size, slots, server build, temperature, seed, max output tokens, max attempts, samples, penalties, the prompts file's checksum, the server note, and one entry per session.

## 2. How it was built and tested

`test_model_runner.py` starts a small fake server inside the test, so no model is needed. 20 tests, covering:

- a good reply, a rejected-then-accepted reply, and three rejections in a row
- stop and resume, a half-written last line, resuming with other settings
- identical settings for both conditions
- a context that is too small (stops before sending), and a prompt the server cut short
- a server that keeps failing, and one that fails once and recovers
- settings recorded and the API key kept out of every file
- thinking kept separate, inline `<think>` text, a reply cut off at the cap
- temperature 0, several samples, several workers, shards, quantization names
- `scorer.py` reading the runner's output

Run the test files before a run: 8, 19, 11 and 20 tests, plus `test_baseline.py` (5 tests, or 6 with `.venv/bin/python`, which adds one that uses the real Chronos-2 model).

Note: `test_reply_parser.py` prints "19 tests" but runs only 18 unless `prompts/AECI_pilot_prompts.jsonl` exists. Build the pilot prompts first.

`prompt_builder.py`, `reply_parser.py` and `scorer.py` were not changed.

## 3. The machine and the servers

- GPU 1: NVIDIA RTX 4090, 24 GB. In llama.cpp's Vulkan build it is `Vulkan0`.
- GPU 2: AMD Radeon 8060S (Strix Halo), which shares system memory, about 110 GB. It is `Vulkan1`.
- llama.cpp build used: `~/apps/llama-b10964-vulkan` (build b10964-b29c606e2). It sees both GPUs. There is no CUDA build.
- Model: `~/models/qwen3.8/Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.9 GB). The server reports its type as "Q4_K - Small".
- Draft model for faster writing: `~/models/qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf`.

The launch commands are the one in `~/apps/start_llama_server_amd_mtp.sh` with a few flags changed. The script files were not edited. Changes common to every launch: `--host 127.0.0.1` (this machine only), no `--api-key`, and `--device` to pin one GPU. Without `--device` the script does not choose a GPU.

Base command (run from `~/apps/llama-b10964-vulkan`):

    ./llama-server -m ~/models/qwen3.8/Qwen3.8-27B-UD-Q4_K_XL.gguf --no-mmproj \
      -ngl 999 --flash-attn on --load-mode mmap -b 4096 -ub 2048 \
      --host 127.0.0.1 --jinja --reasoning-effort low \
      -md ~/models/qwen3.8/mtp-Qwen3.8-27B-Q4_0.gguf --spec-type draft-mtp --spec-draft-n-max 3 -ngld 999 \
      --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0

Added per server:

| Server | Added flags |
|---|---|
| 4090, thinking capped (validation) | `--port 9998 --ctx-size 32768 --device Vulkan0 --reasoning-budget 6000` |
| Radeon, thinking uncapped (validation) | `--port 9999 --ctx-size 262144 --device Vulkan1 --timeout 7200` |
| 4090, thinking off (pilot only) | `--port 9998 --ctx-size 32768 --device Vulkan0 --reasoning off` |
| Radeon, first pilot | `--port 9999 --ctx-size 131072 --device Vulkan1` |

Server logs are in `../server_logs/` (outside both repos).

## 4. How to do a run

1. Run the four test files.
2. Start a server (section 3) and wait until `curl http://127.0.0.1:PORT/health` says ok.
3. Build prompts, for example:
   `python3 prompt_builder.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl --out prompts/AECI_validation_prompts.jsonl`
4. Run, for example (the 4090 validation run):
   `python3 model_runner.py prompts/AECI_validation_prompts.jsonl --run validation_4090_think6000 --temperature 1.0 --max-tokens 15000 --base-url http://127.0.0.1:9998 --server-note "..."`
5. Score, at any time, also while it runs:
   `python3 scorer.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl results/validation_4090_think6000/forecasts.jsonl --out results/validation_4090_think6000/score`
6. To stop: Ctrl-C the runner. To continue: the same command.

Use a new `--run` name whenever anything about the server or settings changes. Use the validation windows only. The test split has not been run or opened.

## 5. What was found

Sizes:

1. One prompt is about 17,000 tokens with this model's tokenizer (16,758 to 17,012 across the 178 validation prompts, plus about 30 for the chat template). That is almost one token per character, because digits are tokenized one by one.
2. A perfect 72-line reply is about 2,170 tokens.
3. So a 16,384 context would silently cut the history. 32,768 is the smallest that works.

Thinking (the main difficulty):

4. This is a thinking model. Its chat template has three levels: `low`, `medium` and `xhigh`. `low` is the lowest, and it is what the script uses.
5. Even at `low`, total output was 6,600 to 19,400 tokens in ten uncapped replies. Thinking alone is roughly 4,000 to 17,000.
6. With a 12,000-token output cap, 6 of 16 pilot attempts were cut off. The cut-off replies were not stuck in a loop. The model keeps re-copying data and trying another angle.
7. Retrying until a reply is short enough keeps only the short-thinking replies. That can bias the comparison, so a cap that cuts replies off is not acceptable.
8. `--reasoning-budget 6000` ends thinking after 6,000 tokens and the model then answers. Output stays under about 8,300 tokens. This is the only setting that bounds the output.
9. `--reasoning off` does not remove the reasoning. The model writes it in the visible answer instead, and output ranged from 2,168 to 14,127 tokens. The parser accepts text before the `<forecast>` block, so these parse, but the length is not bounded.
10. The server returns thinking in `reasoning_content`, apart from the answer in `content`, so the parser never sees it.
11. Uncapped, the model sometimes ends its turn while still thinking and writes no forecast at all (seen once so far: 24,344 tokens of thinking, empty answer, normal stop).

The two GPUs:

12. Speed with a 17,000-token prompt: the 4090 reads about 2,000 tokens/s and writes 80 to 90. The Radeon reads about 214 and writes about 27.
13. The very first request on the 4090 took about 60 s longer (a one-time warm-up). Later server restarts did not show it.
14. The 4090 holds the model with a 32,768 context (23 of 24.5 GB used). A 49,152 context loads but fails with "ErrorOutOfDeviceMemory" on the first real prompt. So the 4090's output cap cannot go above about 15,000 tokens, and uncapped thinking cannot run on it.
15. Parallel requests do not help on the Radeon. Four at once gave about 30 tokens/s in total, against 27 for one.
16. The server's 4 slots share one context. A second request arriving while one is running broke the running one ("Context size has been exceeded"). Nothing else may use the server during a run.
17. Same seed on the same GPU gives the identical reply. The same seed on the other GPU gives a different reply (6,944 output tokens on the 4090, 4,800 on the Radeon, for the same prompt). So the two conditions must never be split across GPUs.

Model behaviour:

18. A repeated failure: the model sometimes forecasts only 48 of the 72 hours. A retry fixed it each time.
19. The model uses the text. Its thinking quotes the fuel mix and the weather forecast.
20. It guessed that AECI is in Alberta. That is believed to be wrong (Associated Electric Cooperative, US Midwest). The background text does not say where the grid is.
21. In one reply it counted 480 past hours where 504 were given. The server had read the whole prompt, so this was the model's own miscount.

Settings:

22. `start_llama_server_amd_nomtp.sh` sets `--repeat-penalty 1.05`. That would distort a reply made of repeated digits. The runner switches penalties off in every request.
23. The file name says `UD-Q4_K_XL` and the server reports "Q4_K - Small". Both are recorded.
24. `aeci_pilot.py` prints the test split's baseline MAEs in its build summary. That is the script's own output. No test file was opened.

## 6. Runs

Pilot runs (5 windows, 10 prompts, `prompts/AECI_pilot_prompts.jsonl`, temperature 1.0). These are final.

| Run | GPU | Thinking | Output cap | Prompts with a forecast | Cut off | MAE with text | MAE without | Text effect |
|---|---|---|---|---|---|---|---|---|
| `pilot_4090` | 4090 | low | 12,000 | 9 of 10 | 6 of 16 attempts | 50.45 | 44.24 | -14.0% (4 windows) |
| `pilot_radeon_cap60k` (seed 1) | Radeon | low, uncapped | 60,000 | 10 of 10 | 0 | 40.98 | 48.56 | +15.6% |
| `pilot_4090_thinking_off_cap15k` | 4090 | off | 15,000 | 10 of 10 | 0 | 46.60 | 42.96 | -8.5% |
| `pilot_4090_budget6000` | 4090 | capped at 6,000 | 15,000 | 10 of 10 | 0 | 49.36 | 44.06 | -12.0% |

Positive means text helped. With five windows and one sample each, these swing from -14% to +16% and none is evidence either way. They were checks that everything works.

Other pilot folders are small tests: `pilot_radeon` (2 prompts), `pilot_radeon_budget6000` (4 prompts), `pilot_4090_thinking_off` (cap of 4,000 was too small), `pilot_4090_ctx49k_cap30k` (out of memory, no replies), `pilot_radeon_par4` (parallel speed test, stopped).

Validation runs (89 windows, 178 prompts, temperature 1.0, seed 0, one sample, up to 3 attempts). **In progress when this was written.**

| Run | GPU | Thinking | Output cap | Started | Status |
|---|---|---|---|---|---|
| `validation_4090_think6000` | 4090 | capped at 6,000 | 15,000 | 2026-11-14 10:05 PST | 20 of 178 after 35 minutes, 0 rejected, about 104 s per prompt |
| `validation_radeon_uncapped` | Radeon | low, uncapped | 60,000 | 2026-11-14 10:14 PST | 1 of 178 after 25 minutes, 1 rejected (finding 11) |

Each run does both conditions for every window, so within a run the prompt is the only difference. The two runs differ in thinking cap and in GPU, so compare with-text against without-text inside each run, not error numbers across runs.

### Chronos-2 baseline (final)

Chronos-2 is a forecasting model from Amazon that reads numbers, not text. It was added on 2026-11-14 as a baseline: how well does a real forecasting model do on the same windows? It is not part of the with-text against without-text comparison.

- Script: `chronos_baseline.py`. Run name: `validation_chronos2`. Scored with `compare_forecasts.py`.
- Model: `amazon/chronos-2`, snapshot `29ec3766d36d6f73f0696f85560a422f50e8498c`, 119.5 million parameters. Packages: chronos-forecasting 2.3.2, torch 2.15.0 (CPU only), in `.venv`.
- Input: the 504 past values only, rounded to 6 significant digits exactly as the LLM prompt prints them. No text, no weather.
- One window per call, with cross-learning off. The windows overlap in time, so letting the model look across windows would leak one window's future into another.
- The forecast is the median (0.5 quantile). The 10% to 90% quantiles are saved in `quantiles.jsonl`.
- No randomness: the same input gives the same forecast. All 89 windows took 4.4 s on the CPU.

| On all 89 validation windows | MAE |
|---|---|
| Chronos-2 | 50.55 |
| Last-value baseline | 60.59 |
| Daily-repeat baseline | 62.36 |

Not known: whether Chronos-2's training data included CarbonCast or the EIA grid data it comes from. If it did, "zero-shot" would not be accurate.

Not done: giving Chronos-2 the weather forecast as numeric covariates. That would show whether the information in the text helps a model when it arrives as numbers. It needs hourly covariates built in the dataset repo, with the same leak checks.

## 7. Mistakes made, so they are not repeated

- A question to Dev waited about eight hours, and the first overnight window was lost. Start long runs before asking anything that can wait.
- Twice a test run that was meant to be stopped kept running, because the shell wrapper was stopped and not the Python process inside it. One of them re-sent a request five minutes later and collided with another run (finding 16). The runner re-asked and nothing wrong was recorded. Before reusing a server, confirm with `ps` that no other `model_runner.py` is alive.
- A cap of 4,000 tokens was assumed to be enough with thinking off. It was not (finding 9). Measure output length before choosing a cap.

## 8. Open questions

- Is capping thinking at 6,000 tokens the right choice? 6,000 was picked to bound the output, not tuned. The uncapped Radeon run is the check.
- How many samples per prompt? The scorer supports several (median for MAE, and CRPS). One sample is noisy. The Context is Key code is believed to use temperature 1.0 and 25 samples; that was recalled, not checked against the code.
- Should the background text say where AECI is (finding 20)? That would be a dataset change.
