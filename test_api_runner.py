"""Offline tests for api_runner.py, using a fake model server on this machine. Nothing leaves the computer.
Run:  python3 test_api_runner.py"""
import json, os, re, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from prompt_builder import build_entries
from scorer import score

HERE = Path(__file__).resolve().parent
KEY = "TESTKEY-not-a-real-key-123"
RUN = f"selftest_api_{os.getpid()}"
state = {"calls": 0, "auth": set(), "bodies": [], "per_prompt": {}}

def toy_record(seed):
    past = {f"2019-01-21T{h:02d}:00:00.000": 600.0 + h + seed for h in range(24)}
    fut = {f"2019-01-22T{h:02d}:00:00.000": 620.0 + h for h in (6, 7, 8)}
    return {"name": "Toy", "seed": seed, "background": "Background text.", "scenario": "Scenario text.", "constraints": "",
            "past_time": json.dumps({"x": past}), "future_time": json.dumps({"x": fut}), "metric_scaling": 0.01}

class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state["calls"] += 1; state["auth"].add(self.headers.get("Authorization")); state["bodies"].append({k: v for k, v in body.items() if k != "messages"})
        prompt = body["messages"][1]["content"]
        n = state["per_prompt"][prompt] = state["per_prompt"].get(prompt, 0) + 1
        hist = re.findall(r"<history>\n(.*?)\n</history>", prompt, re.S)[0].split("\n")
        last = hist[-1].rsplit(",", 1)[1].rstrip(")").strip()
        stamps = re.findall(r"'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)'", prompt.split("following timestamps:")[1])
        good = "<think>thinking</think>\n<forecast>\n" + "\n".join(f"({t}, {last})" for t in stamps) + "\n</forecast>"
        first_prompt = min(state["per_prompt"], key=lambda p: list(state["per_prompt"]).index(p)) == prompt
        if first_prompt and n == 1:                       # first prompt, first try: server error
            self.send_response(500); self.end_headers(); self.wfile.write(b"boom"); return
        content = "Sorry, no forecast." if (not first_prompt and n == 1 and "Scenario text." in prompt and "seed" not in prompt and len(state["per_prompt"]) == 2) else good
        out = {"model": body["model"], "provider": "FakeProvider", "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
               "usage": {"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01}}
        data = json.dumps(out).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

results = []
def check(name, cond):
    results.append(bool(cond)); print(f"{'pass' if cond else 'FAIL'} | {name}")

def run_runner(prompts, port, extra=()):
    env = {**os.environ, "RUNNER_TEST_KEY": KEY}
    cmd = [sys.executable, str(HERE / "api_runner.py"), str(prompts), "--run", RUN, "--model", "fake-model", "--base-url", f"http://127.0.0.1:{port}/v1",
           "--api-key-env", "RUNNER_TEST_KEY", "--workers", "1", "--attempts", "3", *extra]
    return subprocess.run(cmd, cwd=HERE, env=env, capture_output=True, text=True)

def main():
    server = HTTPServer(("127.0.0.1", 0), Fake); port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    recs = [toy_record(1), toy_record(2)]
    forecasts = HERE / "results" / RUN / "forecasts.jsonl"; settings = HERE / "results" / RUN / "settings.json"; replies = HERE / "replies" / RUN
    try:
        with tempfile.TemporaryDirectory() as d:
            prompts = Path(d) / "p.jsonl"
            prompts.write_text("".join(json.dumps(e) + "\n" for r in recs for e in build_entries(r)))
            # 1. Stop at a cost limit: each fake call "costs" $0.01, so a $0.015 limit stops after two paid requests.
            p = run_runner(prompts, port, ("--max-cost", "0.015"))
            lines = [json.loads(l) for l in forecasts.read_text().splitlines()]
            check("stops when the reported cost reaches the limit", p.returncode == 0 and "reached the limit" in p.stdout and sum(l["ok"] for l in lines) == 2)
            # 2. Run again without the limit: it must pick up where it stopped and finish all four prompts.
            before = state["calls"]; p = run_runner(prompts, port)
            lines = [json.loads(l) for l in forecasts.read_text().splitlines()]
            ok = {(l["seed"], l["condition"]) for l in lines if l["ok"]}
            check("a second run resumes and finishes every prompt", p.returncode == 0 and len(ok) == 4 and "2 already finished" in p.stdout and state["calls"] - before == 2)
            check("a server error is recorded and retried", any((not l["ok"]) and l["reason"].startswith("request error: HTTP 500") for l in lines))
            # 3. A third run has nothing left to do and makes no requests.
            before = state["calls"]; p = run_runner(prompts, port)
            check("a finished run makes no further requests", state["calls"] == before and "0 to run" in p.stdout)
            # 4. The key: sent to the server, written nowhere.
            written = forecasts.read_text() + settings.read_text() + "".join(f.read_text() for f in replies.glob("*.json")) + p.stdout + p.stderr
            check("the key is sent as a bearer token and appears in no output file or printout", state["auth"] == {f"Bearer {KEY}"} and KEY not in written)
            # 5. OpenRouter-only fields are not sent to another server, and no identifying headers or fields are added.
            check("no OpenRouter-only fields go to a non-OpenRouter server", all(set(b) <= {"model", "temperature", "max_tokens", "seed"} for b in state["bodies"]))
            # 5b. A retry uses a new sampling seed, and every line records it.
            first = sorted((l for l in lines if (l["seed"], l["condition"]) == (lines[0]["seed"], lines[0]["condition"])), key=lambda l: l["attempt"])
            check("each attempt gets its own sampling seed, recorded in the forecasts file", [l["request_seed"] for l in first] == list(range(len(first))) and len(first) >= 2
                  and {b["seed"] for b in state["bodies"]} >= {0, 1})
            # 6. Raw replies are saved with token counts, and the settings file keeps totals.
            s = json.loads(settings.read_text()); one = json.loads(next(replies.glob("*.json")).read_text())
            # Totals must cover all three invocations: 4 parsed forecasts and 4 paid requests at $0.01, plus the one server error.
            t = s["totals"]
            check("raw replies are saved, and totals carry across resumed runs", s["model"] == "fake-model" and t["parsed"] == 4 and t["requests"] == 4 and abs(t["cost_usd"] - 0.04) < 1e-9
                  and t["request_errors"] == 1 and one["usage"]["prompt_tokens"] == 100 and one["provider"] == "FakeProvider" and "content" in one)
            # 7. The scorer can read the runner's output.
            records = {("Toy", r["seed"]): {"truth": np.array(list(json.loads(r["future_time"])["x"].values())), "past": np.array(list(json.loads(r["past_time"])["x"].values())), "scaling": 0.01} for r in recs}
            rep, _ = score(records, lines)
            check("scorer.py reads the forecasts file (2 windows compared)", rep["paired_windows"] == 2 and rep["mae"]["mean_difference"] == 0)
            # 8. No key configured: a clear message, no request.
            before = state["calls"]
            q = subprocess.run([sys.executable, str(HERE / "api_runner.py"), str(prompts), "--run", RUN, "--model", "m", "--base-url", f"http://127.0.0.1:{port}/v1", "--api-key-env", "RUNNER_TEST_KEY_MISSING"],
                               cwd=d, env={k: v for k, v in os.environ.items() if k != "RUNNER_TEST_KEY_MISSING"}, capture_output=True, text=True)
            check("with no key it stops with a clear message and sends nothing", q.returncode != 0 and "No API key" in q.stderr and state["calls"] == before)
    finally:
        server.shutdown()
        for f in [forecasts, settings, *replies.glob("*.json")]:
            if f.exists(): f.unlink()
        if replies.exists(): replies.rmdir()
        if forecasts.parent.exists() and not any(forecasts.parent.iterdir()): forecasts.parent.rmdir()
    print(f"\n{len(results)} tests, {results.count(False)} failed")
    raise SystemExit(1 if not all(results) else 0)

if __name__ == "__main__":
    main()
