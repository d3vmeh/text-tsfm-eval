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
state = {"calls": 0, "auth": set(), "bodies": [], "per_prompt": {}, "order": {}}

def toy_record(seed):
    past = {f"2019-01-21T{h:02d}:00:00.000": 600.0 + h + seed for h in range(24)}
    fut = {f"2019-01-22T{h:02d}:00:00.000": 620.0 + h for h in (6, 7, 8)}
    return {"name": "Toy", "seed": seed, "background": "Background text.", "scenario": "Scenario text.", "constraints": "",
            "past_time": json.dumps({"x": past}), "future_time": json.dumps({"x": fut}), "metric_scaling": 0.01}

class Fake(BaseHTTPRequestHandler):
    """Streams replies the way a hosted API does. By prompt, in the order first seen:
       prompt 0: first call answers HTTP 500, then a good reply       (provider trouble)
       prompt 1: first call a reply with no forecast, then a good one (a rejected reply)
       prompt 2: first call a reply that is cut off part way, then good (dropped connection)
       prompt 3: a good reply"""
    def log_message(self, *a): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        state["calls"] += 1; state["auth"].add(self.headers.get("Authorization")); state["bodies"].append({k: v for k, v in body.items() if k != "messages"})
        prompt = body["messages"][1]["content"]
        idx = state["order"].setdefault(prompt, len(state["order"]))
        n = state["per_prompt"][prompt] = state["per_prompt"].get(prompt, 0) + 1
        hist = re.findall(r"<history>\n(.*?)\n</history>", prompt, re.S)[0].split("\n")
        last = hist[-1].rsplit(",", 1)[1].rstrip(")").strip()
        stamps = re.findall(r"'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)'", prompt.split("following timestamps:")[1])
        good = "<forecast>\n" + "\n".join(f"({t}, {last})" for t in stamps) + "\n</forecast>"
        if idx == 0 and n == 1:
            self.send_response(500); self.end_headers(); self.wfile.write(b"boom"); return
        content = "Sorry, no forecast." if (idx == 1 and n == 1) else good
        def chunk(delta=None, finish=None, usage=None):
            d = {"model": body["model"], "provider": "FakeProvider", "choices": [{"delta": delta or {}, "finish_reason": finish}]}
            if usage: d["usage"] = usage
            self.wfile.write(f"data: {json.dumps(d)}\n\n".encode()); self.wfile.flush()
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        self.wfile.write(b": keep-alive comment\n\n")
        chunk({"reasoning": "thinking about it"})
        half = len(content) // 2
        chunk({"content": content[:half]})
        if idx == 2 and n == 1:
            return                                           # the connection closes with the reply unfinished
        chunk({"content": content[half:]})
        chunk(finish="stop", usage={"prompt_tokens": 100, "completion_tokens": 50, "cost": 0.01})
        self.wfile.write(b"data: [DONE]\n\n")

results = []
def check(name, cond):
    results.append(bool(cond)); print(f"{'pass' if cond else 'FAIL'} | {name}")

def run_runner(prompts, port, extra=()):
    env = {**os.environ, "RUNNER_TEST_KEY": KEY}
    cmd = [sys.executable, str(HERE / "api_runner.py"), str(prompts), "--run", RUN, "--model", "fake-model", "--base-url", f"http://127.0.0.1:{port}/v1",
           "--api-key-env", "RUNNER_TEST_KEY", "--workers", "1", "--attempts", "3", "--retry-waits", "0.05,0.05", *extra]
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
            # 1. Stop at a cost limit: each finished reply "costs" $0.01, so a $0.015 limit stops after two of them
            #    (prompt 0's good reply, then prompt 1's reply with no forecast).
            p = run_runner(prompts, port, ("--max-cost", "0.015"))
            lines = [json.loads(l) for l in forecasts.read_text().splitlines()]
            check("stops when the reported cost reaches the limit", p.returncode == 0 and "reached the limit" in p.stdout
                  and [l["ok"] for l in lines] == [True, False] and lines[1]["reason"] == "no forecast block")
            # 2. Run again without the limit: it must pick up where it stopped and finish all four prompts.
            before = state["calls"]; p = run_runner(prompts, port)
            lines = [json.loads(l) for l in forecasts.read_text().splitlines()]
            ok = {(l["seed"], l["condition"]) for l in lines if l["ok"]}
            check("a second run resumes and finishes every prompt", p.returncode == 0 and len(ok) == 4 and "1 already finished" in p.stdout and state["calls"] - before == 4)
            # 3. Provider trouble (the HTTP 500 and the cut-off reply) was retried and is NOT recorded as a rejected reply.
            s_ = json.loads(settings.read_text())
            check("network and provider trouble is retried without counting against the model",
                  s_["totals"]["transport_errors"] == 2 and sum(not l["ok"] for l in lines) == 1 and not any("request error" in (l.get("reason") or "") for l in lines))
            # 3b. A third run has nothing left to do and makes no requests.
            before = state["calls"]; p = run_runner(prompts, port)
            check("a finished run makes no further requests", state["calls"] == before and "0 to run" in p.stdout)
            # 4. The key: sent to the server, written nowhere.
            written = forecasts.read_text() + settings.read_text() + "".join(f.read_text() for f in replies.glob("*.json")) + p.stdout + p.stderr
            check("the key is sent as a bearer token and appears in no output file or printout", state["auth"] == {f"Bearer {KEY}"} and KEY not in written)
            # 5. OpenRouter-only fields are not sent to another server, and no identifying headers or fields are added.
            check("no OpenRouter-only fields go to a non-OpenRouter server", all(set(b) <= {"model", "temperature", "max_tokens", "seed", "stream", "stream_options"} and b["stream"] is True for b in state["bodies"]))
            # 5b. The rejected prompt was asked again in the resumed run with attempt 1 and a new sampling seed, not attempt 0 again.
            again = sorted((l for l in lines if (l["seed"], l["condition"]) == (lines[1]["seed"], lines[1]["condition"])), key=lambda l: l["attempt"])
            check("a retry continues the attempt count and uses a new sampling seed, also after a resume",
                  [(l["attempt"], l["request_seed"], l["ok"]) for l in again] == [(0, 0, False), (1, 1, True)])
            # 6. Raw replies are saved with token counts, and the settings file keeps totals.
            s = json.loads(settings.read_text()); one = json.loads(next(replies.glob("*.json")).read_text())
            # Totals must cover all three invocations: 4 parsed forecasts, 5 finished replies at $0.01, 1 rejected, 2 transport errors.
            t = s["totals"]
            check("raw replies and thinking are saved, and totals carry across resumed runs", s["model"] == "fake-model" and t["parsed"] == 4 and t["rejected"] == 1
                  and t["requests"] == 5 and abs(t["cost_usd"] - 0.05) < 1e-9 and t["transport_errors"] == 2
                  and one["usage"]["prompt_tokens"] == 100 and one["provider"] == "FakeProvider" and one["reasoning"] == "thinking about it" and one["finish_reason"] == "stop")
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
