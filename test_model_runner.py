"""Tests for model_runner.py. Run: python3 test_model_runner.py

They use a small fake server inside this script, so no model or llama.cpp server is needed.
"""
import contextlib, io, json, tempfile, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import model_runner
from model_runner import main, pick_shard, quantization_from_name
from scorer import score

model_runner.WAITS = (0.01, 0.01)          # don't really wait in tests
TS = [f"2021-10-01 {h:02d}:00:00" for h in range(6)]
MODEL = "/models/fake/Fake-7B-UD-Q4_K_XL.gguf"


def good(base=500.0):
    return "<forecast>\n" + "\n".join(f"({t}, {base + k})" for k, t in enumerate(TS)) + "\n</forecast>"


def n_tokens(text):
    return len(text) // 2


class Fake:
    """A stand-in for the llama.cpp server. `reply` decides what each chat request gets back."""
    def __init__(self, reply=lambda body, n: {"content": good()}, n_ctx=65536, api_key=None, model=MODEL, slots=1):
        self.reply, self.n_ctx, self.api_key, self.model, self.slots = reply, n_ctx, api_key, model, slots
        self.chats = []                     # every chat request body received
        self.in_flight = 0; self.most_in_flight = 0; self.count_lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

            def allowed(self):
                if fake.api_key and self.headers.get("Authorization") != f"Bearer {fake.api_key}":
                    self.send(401, {"error": "bad key"}); return False
                return True

            def do_GET(self):
                if not self.allowed(): return
                if self.path == "/props":
                    self.send(200, {"default_generation_settings": {"n_ctx": fake.n_ctx, "params": {"top_k": 20, "top_p": 0.95}},
                                    "total_slots": fake.slots, "model_path": fake.model, "build_info": "b1-fake", "chat_template": "T"})
                elif self.path == "/v1/models":
                    self.send(200, {"data": [{"id": "fake", "meta": {"n_params": 7}}]})
                else:
                    self.send(404, {"error": "no such path"})

            def do_POST(self):
                if not self.allowed(): return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    self.send(200, {"tokens": list(range(n_tokens(body["content"])))}); return
                with fake.count_lock:
                    fake.chats.append(body); nth = len(fake.chats) - 1
                    fake.in_flight += 1; fake.most_in_flight = max(fake.most_in_flight, fake.in_flight)
                try:
                    r = fake.reply(body, nth)
                finally:
                    with fake.count_lock:
                        fake.in_flight -= 1
                if "status" in r:
                    self.send(r["status"], {"error": "boom"}); return
                counted = sum(n_tokens(m["content"]) for m in body["messages"])
                msg = {"role": "assistant", "content": r.get("content", "")}
                if "thinking" in r:
                    msg["reasoning_content"] = r["thinking"]
                self.send(200, {"choices": [{"message": msg, "finish_reason": r.get("finish", "stop")}],
                                "usage": {"prompt_tokens": r.get("prompt_tokens", counted + 12), "completion_tokens": 40}})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown(); self.httpd.server_close()


def write_prompts(folder, windows=2):
    path = Path(folder) / "prompts.jsonl"
    with open(path, "w") as f:
        for w in range(windows):
            for cond, extra in (("with_context", "Background: wind is rising.\n"), ("no_context", "")):
                f.write(json.dumps({"name": "Fake", "seed": 20211001 + w, "condition": cond, "future_timestamps": TS,
                                    "messages": [{"role": "system", "content": "You are a useful forecasting assistant."},
                                                 {"role": "user", "content": f"\n<context>\n{extra}</context>\nhistory {w}\n"}]}) + "\n")
    return path


def go(folder, fake, *extra, temperature="1.0", run="t", prompts=None):
    """Run the runner against the fake server. Returns (exit code, printed text, forecast lines)."""
    prompts = prompts or write_prompts(folder)
    argv = [str(prompts), "--run", run, "--temperature", temperature, "--base-url", fake.url,
            "--results-dir", f"{folder}/results", "--replies-dir", f"{folder}/replies", "--timeout", "5", *extra]
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(argv)
    path = Path(folder) / "results" / run / "forecasts.jsonl"
    lines = [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    return code, out.getvalue(), lines


def t_good_reply(d):
    fake = Fake(lambda body, n: {"content": good(500 + n)})
    code, _, lines = go(d, fake); fake.close()
    first = lines[0]
    files = sorted(p.name for p in (Path(d) / "replies/t").iterdir())
    return (code == 0 and len(lines) == 4 and all(l["ok"] for l in lines) and first["values"] == [500.0 + k for k in range(6)]
            and [l["condition"] for l in lines] == ["with_context", "no_context"] * 2
            and "Fake_20211001_with_context_s0_a0.json" in files and "Fake_20211001_with_context_s0_a0.txt" in files
            and (Path(d) / "replies/t/Fake_20211001_with_context_s0_a0.txt").read_text() == good(500))


def t_retry_then_accept(d):
    fake = Fake(lambda body, n: {"content": "I cannot forecast this."} if n == 0 else {"content": good()})
    code, _, lines = go(d, fake); fake.close()
    a, b = lines[0], lines[1]
    return (code == 0 and len(lines) == 5 and not a["ok"] and a["reason"] == "no forecast block" and a["attempt"] == 0
            and b["ok"] and b["attempt"] == 1 and (a["name"], a["seed"], a["condition"]) == (b["name"], b["seed"], b["condition"])
            and fake.chats[0]["seed"] != fake.chats[1]["seed"] and a["request_seed"] == fake.chats[0]["seed"])


def t_gives_up_and_resume_skips(d):
    fake = Fake(lambda body, n: {"content": "<forecast>\n(2021-10-01 00:00:00, 5)\n</forecast>"})
    prompts = write_prompts(d, windows=1)
    code, _, lines = go(d, fake, prompts=prompts)
    sent = len(fake.chats)
    code2, text2, lines2 = go(d, fake, prompts=prompts); fake.close()
    return (code == 0 and len(lines) == 6 and not any(l["ok"] for l in lines) and lines[0]["reason"].startswith("missing 5 of 6")
            and sent == 6 and code2 == 0 and len(fake.chats) == 6 and len(lines2) == 6 and "2 already finished, 0 to do" in text2)


def t_stop_and_resume(d):
    fake = Fake()
    prompts = write_prompts(d, windows=3)
    _, _, part = go(d, fake, "--limit", "3", prompts=prompts)
    _, text, lines = go(d, fake, prompts=prompts); fake.close()
    keys = [(l["seed"], l["condition"]) for l in lines]
    return len(part) == 3 and len(lines) == 6 and len(set(keys)) == 6 and len(fake.chats) == 6 and "3 already finished, 3 to do" in text


def t_same_settings_both_conditions(d):
    fake = Fake()
    go(d, fake, "--top-p", "0.95"); fake.close()
    strip = lambda body: {k: v for k, v in body.items() if k != "messages"}
    a, b = fake.chats[0], fake.chats[1]
    return (strip(a) == strip(b) and a["messages"] != b["messages"] and a["repeat_penalty"] == 1.0 and a["presence_penalty"] == 0.0
            and a["temperature"] == 1.0 and a["max_tokens"] == 12000 and a["top_p"] == 0.95 and "top_k" not in a and a["cache_prompt"] is False)


def t_context_too_small(d):
    fake = Fake(n_ctx=8000)                 # the default 12000 output tokens alone do not fit
    code, text, lines = go(d, fake); fake.close()
    return code == 2 and not fake.chats and not lines and "server's context is 8000" in text


def t_cut_prompt_detected(d):
    fake = Fake(lambda body, n: {"content": good(), "prompt_tokens": 5})
    code, text, lines = go(d, fake); fake.close()
    saved = list((Path(d) / "replies/t").glob("*.json"))
    return code == 2 and not lines and len(fake.chats) == 1 and "may have been cut" in text and len(saved) == 1


def t_server_trouble_not_a_model_failure(d):
    fake = Fake(lambda body, n: {"status": 500})
    code, text, lines = go(d, fake); fake.close()
    return code == 2 and not lines and len(fake.chats) == 3 and "not answering" in text


def t_server_recovers(d):
    fake = Fake(lambda body, n: {"status": 503} if n == 0 else {"content": good()})
    code, _, lines = go(d, fake); fake.close()
    return code == 0 and len(lines) == 4 and all(l["ok"] and l["attempt"] == 0 for l in lines)


def t_settings_recorded_and_key_kept_out(d):
    import os
    fake = Fake(api_key="s3cret-key")
    os.environ["TEST_RUNNER_KEY"] = "s3cret-key"
    code, _, lines = go(d, fake, "--api-key-env", "TEST_RUNNER_KEY", "--seed", "7", "--max-tokens", "900", "--server-note", "--reasoning off")
    os.environ["TEST_RUNNER_KEY"] = "wrong"
    code_bad, text_bad, _ = go(d, fake, "--api-key-env", "TEST_RUNNER_KEY", "--seed", "7", "--max-tokens", "900", run="bad")
    del os.environ["TEST_RUNNER_KEY"]; fake.close()
    s = json.loads((Path(d) / "results/t/settings.json").read_text())
    everything = "".join(p.read_text() for p in Path(d).rglob("*") if p.is_file())
    return (code == 0 and s["model_file"] == MODEL and s["quantization"] == "UD-Q4_K_XL" and s["context_size"] == 65536
            and s["temperature"] == 1.0 and s["seed"] == 7 and s["max_output_tokens"] == 900 and s["server_note"] == "--reasoning off" and s["server_default_sampling"]["top_k"] == 20
            and s["sessions"][0]["api_key_used"] is True and "s3cret-key" not in everything
            and code_bad == 2 and "refused the API key" in text_bad and lines[0]["request_seed"] == 7)


def t_resume_with_other_settings_refused(d):
    fake = Fake()
    prompts = write_prompts(d)
    go(d, fake, "--limit", "1", prompts=prompts)
    code, text, lines = go(d, fake, prompts=prompts, temperature="0.5")
    other = Fake(model="/models/fake/Other-7B-Q8_0.gguf")
    code2, text2, _ = go(d, other, prompts=prompts)
    other.close()
    code3, text3, _ = go(d, fake, "--server-note", "thinking off", prompts=prompts)
    fake.close()
    return (code == 2 and "temperature: was 1.0, now 0.5" in text and len(lines) == 1 and len(fake.chats) == 1
            and code2 == 2 and "model_file" in text2 and not other.chats and code3 == 2 and "server_note" in text3)


def t_thinking_handled(d):
    inline = "<think>Maybe <forecast>\n(x, 1)\n</forecast>?</think>\n" + good(600)
    fake = Fake(lambda body, n: {"content": good(), "thinking": "Wind rises, so..."} if n == 0 else {"content": inline})
    code, _, lines = go(d, fake); fake.close()
    folder = Path(d) / "replies/t"
    return (code == 0 and lines[0]["ok"] and lines[0]["thinking_chars"] == 17 and lines[1]["ok"] and lines[1]["values"][0] == 600.0
            and (folder / "Fake_20211001_with_context_s0_a0.thinking.txt").read_text() == "Wind rises, so..."
            and (folder / "Fake_20211001_no_context_s0_a0.txt").read_text() == inline)


def t_cut_off_reply_flagged(d):
    cut = "<forecast>\n(2021-10-01 00:00:00, 500.0)\n(2021-10-01 01:00"
    fake = Fake(lambda body, n: {"content": cut, "finish": "length"})
    code, text, lines = go(d, fake, "--max-attempts", "1", prompts=write_prompts(d, windows=1)); fake.close()
    return (code == 0 and not lines[0]["ok"] and "cut off" in lines[0]["reason"] and lines[0]["finish_reason"] == "length"
            and "2 replies cut off" in text)


def t_temperature_zero_no_retry(d):
    fake = Fake(lambda body, n: {"content": "no"})
    code, text, lines = go(d, fake, temperature="0", prompts=write_prompts(d, windows=1)); fake.close()
    return code == 0 and len(lines) == 2 and len(fake.chats) == 2 and "not retried" in text


def t_samples(d):
    fake = Fake(lambda body, n: {"content": good(500 + n)})
    code, _, lines = go(d, fake, "--samples", "3", prompts=write_prompts(d, windows=1)); fake.close()
    seeds = [c["seed"] for c in fake.chats]
    more = Fake(lambda body, n: {"content": good(900 + n)})          # later: ask for a 4th sample in the same run
    code2, _, lines2 = go(d, more, "--samples", "4", prompts=Path(d) / "prompts.jsonl"); more.close()
    return (code == 0 and [(l["condition"], l["sample"]) for l in lines] == [(c, s) for s in range(3) for c in ("with_context", "no_context")]
            and seeds == [0, 0, 1000, 1000, 2000, 2000]
            and code2 == 0 and len(lines2) == 8 and [c["seed"] for c in more.chats] == [3000, 3000])


def t_workers(d):
    import time
    def slow(body, n):
        time.sleep(0.15); return {"content": "no"} if n == 1 else {"content": good(500 + n)}
    fake = Fake(slow, slots=4)
    prompts = write_prompts(d, windows=4)
    code, text, lines = go(d, fake, "--workers", "3", prompts=prompts)
    ok_keys = [(l["seed"], l["condition"]) for l in lines if l["ok"]]
    files = list((Path(d) / "replies/t").glob("*.json"))
    code2, _, lines2 = go(d, fake, "--workers", "3", prompts=prompts); sent = len(fake.chats); fake.close()
    small = Fake(slots=4, n_ctx=30000)          # one request fits (about 12,300 tokens), three at once do not
    code3, text3, lines3 = go(d, small, "--workers", "3", prompts=prompts, run="small"); small.close()
    one_slot = Fake(slots=1)
    code4, text4, _ = go(d, one_slot, "--workers", "2", prompts=prompts, run="oneslot"); one_slot.close()
    return (code == 0 and len(lines) == 9 and sorted(ok_keys) == sorted(set(ok_keys)) and len(ok_keys) == 8 and len(files) == 9
            and fake.most_in_flight >= 2 and fake.most_in_flight <= 3 and "3 request(s) at once" in text
            and code2 == 0 and sent == 9 and len(lines2) == 9
            and code3 == 2 and not small.chats and not lines3 and "3 request(s) at once need" in text3
            and code4 == 2 and not one_slot.chats and "more than the server's 1 slots" in text4)


def t_half_written_line(d):
    fake = Fake()
    prompts = write_prompts(d)
    go(d, fake, "--limit", "2", prompts=prompts)
    path = Path(d) / "results/t/forecasts.jsonl"
    with open(path, "a") as f:
        f.write('{"name": "Fake", "seed": 20211002, "condi')
    code, text, lines = go(d, fake, prompts=prompts); fake.close()
    return code == 0 and len(lines) == 4 and all(l["ok"] for l in lines) and Path(str(path) + ".partial").exists() and "half-written" in text


def t_shards(d):
    entries = [json.loads(l) for l in write_prompts(d, windows=5).read_text().splitlines()]
    a, b = pick_shard(entries, "0/2"), pick_shard(entries, "1/2")
    windows = lambda part: {e["seed"] for e in part}
    both = lambda part: all(sum(1 for e in part if e["seed"] == w) == 2 for w in windows(part))
    return len(a) == 6 and len(b) == 4 and not windows(a) & windows(b) and len(windows(a) | windows(b)) == 5 and both(a) and both(b)


def t_quantization_names(d):
    cases = {"Qwen3.8-27B-UD-Q4_K_XL.gguf": "UD-Q4_K_XL", "Qwen3.8-27B-UD-Q6_K_M.gguf": "UD-Q6_K_M",
             "Qwen3.6-35B-A3B-MTP-Q8_0.gguf": "Q8_0", "gpt-oss-120b-F16.gguf": "F16",
             "Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf": "UD-Q3_K_XL", "model.gguf": ""}
    return all(quantization_from_name("/m/" + name) == want for name, want in cases.items())


def t_scorer_reads_the_output(d):
    fake = Fake(lambda body, n: {"content": "nope"} if n == 0 else {"content": good(500 if n % 2 else 510)})
    _, _, lines = go(d, fake); fake.close()
    truth = [500.0 + k for k in range(6)]
    records = {("Fake", 20211001 + w): {"truth": __import__("numpy").array(truth), "past": __import__("numpy").array(truth * 4), "scaling": 1.0}
               for w in range(2)}
    rep, _ = score(records, lines)
    return (rep["forecast_lines_for_unknown_windows"] == 0 and rep["paired_windows"] == 2
            and rep["per_condition"]["with_context"]["failed_replies"] == 1
            and rep["per_condition"]["with_context"]["failure_reasons"] == {"no forecast block": 1})


TESTS = [
    ("a good reply becomes one ok line with the values written, and the raw reply is saved", t_good_reply),
    ("a rejected reply is retried with a new seed; both attempts are recorded", t_retry_then_accept),
    ("after 3 rejected attempts the failure stays recorded and a second run does not redo it", t_gives_up_and_resume_skips),
    ("a stopped run resumes without redoing finished prompts", t_stop_and_resume),
    ("both conditions get identical settings; only the messages differ; penalties are off", t_same_settings_both_conditions),
    ("a context too small for prompt + output stops the run before anything is sent", t_context_too_small),
    ("a prompt cut by the server is detected and not recorded as a forecast", t_cut_prompt_detected),
    ("a server that keeps failing stops the run without recording model failures", t_server_trouble_not_a_model_failure),
    ("a server that fails once and recovers costs nothing", t_server_recovers),
    ("settings.json records model file, quantization, context, temperature, seed, max tokens; the API key is never saved", t_settings_recorded_and_key_kept_out),
    ("resuming with a different temperature or model is refused", t_resume_with_other_settings_refused),
    ("thinking is saved separately, and inline <think> text does not break parsing", t_thinking_handled),
    ("a reply cut off at max output tokens is recorded as a failure and flagged", t_cut_off_reply_flagged),
    ("at temperature 0 a rejected reply is not retried", t_temperature_zero_no_retry),
    ("several samples per prompt: sample 0 for all prompts first, each with its own seed; more can be added later", t_samples),
    ("several workers: requests overlap, each prompt is recorded once, and all of them must fit the context together", t_workers),
    ("a half-written last line is set aside and the run continues", t_half_written_line),
    ("shards keep both prompts of a window together and cover every window once", t_shards),
    ("quantization is read from the model file name", t_quantization_names),
    ("scorer.py reads the forecasts file and counts the failed attempt", t_scorer_reads_the_output),
]


def main_tests():
    fails = 0
    for name, test in TESTS:
        with tempfile.TemporaryDirectory() as d:
            try:
                ok = bool(test(d))
            except Exception as e:                      # a crash is a failure, with its cause shown
                ok = False; name += f"  [{type(e).__name__}: {e}]"
        fails += not ok
        print(f"{'pass' if ok else 'FAIL'} | {name}")
    print(f"\n{len(TESTS)} tests, {fails} failed")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main_tests()
