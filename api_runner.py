"""Send prompts to a hosted model API and save its forecasts.

For hosted services that speak the OpenAI-style chat API, such as OpenRouter. For a local llama.cpp
server use model_runner.py instead: it checks the server's context size and token counts, which a hosted
API does not expose.

For each prompt it asks the model, reads the reply with reply_parser.parse_reply, retries a rejected reply
with a new sampling seed, and writes the outcome in the format scorer.py reads. Replies are read as a stream,
so a reply that takes many minutes does not drop. Network or provider trouble (timeouts, dropped
connections, "busy" responses) is retried after a wait and never counts as a rejected reply; if it keeps
failing, the prompt is left unfinished and a later run asks again. Output layout, line format
and seed rule match model_runner.py, so scorer.py and compare_forecasts.py work on both.

Usage:
  python3 api_runner.py PROMPTS.jsonl --run NAME --model MODEL_ID [options]

Example:
  python3 api_runner.py prompts/AECI_pilot_prompts.jsonl --run pilot_or_qwen38_27b --model qwen/qwen3.8-27b \\
      --reasoning-max-tokens 6000 --max-cost 1.00

Outputs, all under results/ and replies/ (both ignored by git):
  results/NAME/forecasts.jsonl   one line per attempt, for scorer.py
  results/NAME/settings.json     the settings used, and totals for the whole run, including resumed
                                 parts. --max-cost applies to that whole-run total. Never contains the key.
  replies/NAME/*.json            every raw reply, with its thinking, token counts, provider and timing

Thinking models write a lot before the forecast (see RUN_LOG.md). --max-tokens counts thinking too, so keep
it high, and use --reasoning-max-tokens to bound the thinking where the provider supports it.

The API key is read from an environment variable (default OPENROUTER_API_KEY). If it isn't set, a .env file
in this folder or a parent folder is read. The key is never printed or written anywhere.
A run can be stopped and started again: finished prompts are skipped.
"""
import argparse, json, os, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from reply_parser import parse_reply

HERE = Path(__file__).resolve().parent
OPENROUTER = "https://openrouter.ai/api/v1"


def load_env_file(start: Path):
    """Read KEY=VALUE lines from the nearest .env at or above `start`. Existing variables are not replaced."""
    for folder in [start, *start.parents]:
        f = folder / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                m = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
                if m and m.group(1) not in os.environ:
                    os.environ[m.group(1)] = m.group(2).strip("'\"")
            return f
    return None


class TransportError(Exception):
    """Network or provider trouble, as opposed to a reply the parser rejected."""
    def __init__(self, message, retry=True):
        super().__init__(message); self.retry = retry


def safe(text):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text)).lstrip(".") or "x"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class Runner:
    def __init__(self, args, api_key):
        self.a, self.key = args, api_key
        self.forecasts = HERE / "results" / safe(args.run) / "forecasts.jsonl"
        self.settings = HERE / "results" / safe(args.run) / "settings.json"
        self.replies = HERE / "replies" / safe(args.run)
        self.forecasts.parent.mkdir(parents=True, exist_ok=True); self.replies.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.totals = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0, "seconds": 0.0,
                       "parsed": 0, "rejected": 0, "transport_errors": 0, "prompts_left_unfinished": 0}
        self.transport_log = []
        if self.settings.exists():                                  # a resumed run keeps counting from where it stopped
            try:
                old = json.loads(self.settings.read_text()).get("totals", {})
                self.totals.update({k: old[k] for k in self.totals if isinstance(old.get(k), (int, float))})
            except (ValueError, OSError):
                pass
        self.stop = False

    def done_keys(self):
        """(name, seed, condition, sample) that already have a usable forecast or have used up their attempts.
        Also remembers how many attempts each one has used, so a resumed run carries on from there."""
        self.used_attempts = {}
        if not self.forecasts.exists():
            return set()
        ok, fails = set(), {}
        for r in read_jsonl(self.forecasts):
            k = (r["name"], r["seed"], r["condition"], r["sample"])
            if r.get("ok"):
                ok.add(k)
            else:
                fails[k] = fails.get(k, 0) + 1
        self.used_attempts = dict(fails)
        return ok | {k for k, n in fails.items() if n >= self.a.attempts}

    def call(self, messages, request_seed):
        """Send one request and read the reply as a stream, so a long reply keeps the connection alive.
        Returns (reply dict, seconds). Raises TransportError for network or provider trouble."""
        body = {"model": self.a.model, "messages": messages, "temperature": self.a.temperature, "max_tokens": self.a.max_tokens,
                "seed": request_seed, "stream": True}
        reasoning = {}
        if self.a.reasoning_max_tokens is not None:
            reasoning["max_tokens"] = self.a.reasoning_max_tokens
        if self.a.reasoning_effort:
            reasoning["effort"] = self.a.reasoning_effort
        if reasoning:
            body["reasoning"] = reasoning
        if self.a.base_url.rstrip("/") == OPENROUTER:
            body["usage"] = {"include": True}                       # ask OpenRouter to report the cost
            if self.a.provider:
                body["provider"] = {"order": [self.a.provider], "allow_fallbacks": False}
        else:
            body["stream_options"] = {"include_usage": True}
        body.update(self.a.extra)
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        t0 = time.time()
        out = {"content": "", "reasoning": "", "finish_reason": None, "usage": {}, "model": None, "provider": None}
        try:
            with requests.post(self.a.base_url.rstrip("/") + "/chat/completions", headers=headers, json=body, stream=True,
                               timeout=(30, self.a.idle_timeout)) as resp:
                if resp.status_code != 200:
                    kind = "busy" if resp.status_code in (408, 429, 500, 502, 503, 504, 529) else "refused"
                    raise TransportError(f"HTTP {resp.status_code}: {resp.text[:200]}", retry=(kind == "busy"))
                for raw in resp.iter_lines(decode_unicode=True):
                    if time.time() - t0 > self.a.timeout:
                        raise TransportError(f"no complete reply after {self.a.timeout} s")
                    if not raw or not raw.startswith("data:"):       # blank lines and ": keep-alive" comments
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise TransportError(f"provider error mid-reply: {json.dumps(chunk['error'])[:200]}")
                    out["model"] = chunk.get("model") or out["model"]; out["provider"] = chunk.get("provider") or out["provider"]
                    if chunk.get("usage"):
                        out["usage"] = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        out["content"] += delta.get("content") or ""
                        out["reasoning"] += delta.get("reasoning") or delta.get("reasoning_content") or ""
                        out["finish_reason"] = choice.get("finish_reason") or out["finish_reason"]
        except TransportError:
            raise
        except (requests.RequestException, ValueError) as e:        # dropped connection, idle timeout, broken chunk
            raise TransportError(f"{type(e).__name__}: {str(e)[:160]}")
        if out["finish_reason"] is None:
            raise TransportError("the reply stopped arriving before it was finished")
        if out["finish_reason"] == "error":
            raise TransportError("the provider ended the reply with an error")
        return out, time.time() - t0

    def record(self, entry, sample, attempt, ok, values=None, reason=None, request_seed=None, finish=None):
        line = {"name": entry["name"], "seed": entry["seed"], "condition": entry["condition"], "sample": sample, "ok": ok}
        line.update({"values": values} if ok else {"reason": reason})
        line.update({"attempt": attempt, "request_seed": request_seed, "finish_reason": finish})
        with self.lock:
            with open(self.forecasts, "a") as f:
                f.write(json.dumps(line) + "\n")
            self.totals["parsed" if ok else "rejected"] += 1
            self.write_settings()

    def write_settings(self):
        a = self.a
        self.settings.write_text(json.dumps({
            "run": a.run, "prompts_file": a.prompts, "base_url": a.base_url, "model": a.model, "provider_pinned": a.provider,
            "temperature": a.temperature, "max_tokens": a.max_tokens, "seed": a.seed, "samples": a.samples, "attempts": a.attempts,
            "seed_rule": "request_seed = seed + 1000 x sample + attempt", "reasoning_max_tokens": a.reasoning_max_tokens, "reasoning_effort": a.reasoning_effort,
            "workers": a.workers, "extra_request_fields": a.extra, "max_cost_usd": a.max_cost, "idle_timeout": a.idle_timeout, "timeout": a.timeout,
            "totals": self.totals, "recent_transport_errors": self.transport_log[-20:]}, indent=2) + "\n")

    def one(self, job):
        entry, sample = job
        for attempt in range(self.used_attempts.get((entry["name"], entry["seed"], entry["condition"], sample), 0), self.a.attempts):
            stem = f"{safe(entry['name'])}_{safe(entry['seed'])}_{safe(entry['condition'])}_s{sample}_a{attempt}"
            request_seed = self.a.seed + 1000 * sample + attempt      # a retry must not repeat the same draw
            # Network and provider trouble is not the model's fault: wait and ask again without using up an attempt.
            out = None
            for wait in [*self.a.retry_waits, None]:
                if self.stop:
                    return
                try:
                    out, seconds = self.call(entry["messages"], request_seed)
                    break
                except TransportError as e:
                    with self.lock:
                        self.totals["transport_errors"] += 1
                        self.transport_log.append(f"{stem}: {e}")
                        self.write_settings()
                    if wait is None or not e.retry:
                        with self.lock:
                            self.totals["prompts_left_unfinished"] += 1
                        return                                         # nothing is written, so a resume asks again
                    time.sleep(wait)
            text = out["content"]; usage = out["usage"] or {}; finish = out["finish_reason"]
            (self.replies / f"{stem}.json").write_text(json.dumps({
                "name": entry["name"], "seed": entry["seed"], "condition": entry["condition"], "sample": sample, "attempt": attempt,
                "model": out["model"], "provider": out["provider"], "finish_reason": finish,
                "request_seed": request_seed, "seconds": round(seconds, 2), "usage": usage,
                "reasoning": out["reasoning"] or None, "content": text}, indent=1) + "\n")
            with self.lock:
                self.totals["requests"] += 1; self.totals["seconds"] += seconds
                self.totals["prompt_tokens"] += usage.get("prompt_tokens") or 0
                self.totals["completion_tokens"] += usage.get("completion_tokens") or 0
                self.totals["cost_usd"] += float(usage.get("cost") or 0)
                if self.a.max_cost is not None and self.totals["cost_usd"] >= self.a.max_cost:
                    self.stop = True
            res = parse_reply(text, entry["future_timestamps"])
            if res.ok:
                self.record(entry, sample, attempt, True, values=res.values, request_seed=request_seed, finish=finish)
                return
            reason = res.reason + (" [reply hit the output limit]" if finish == "length" else "")
            self.record(entry, sample, attempt, False, reason=reason, request_seed=request_seed, finish=finish)

    def run(self):
        entries = read_jsonl(self.a.prompts)
        if self.a.limit:
            entries = entries[: self.a.limit]
        done = self.done_keys()
        jobs = [(e, s) for e in entries for s in range(self.a.samples) if (e["name"], e["seed"], e["condition"], s) not in done]
        print(f"{len(entries)} prompts x {self.a.samples} sample(s): {len(jobs)} to run, {len(entries) * self.a.samples - len(jobs)} already finished")
        self.write_settings()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=self.a.workers) as pool:
            for i, _ in enumerate(pool.map(self.one, jobs), 1):
                if i % 10 == 0 or i == len(jobs):
                    t = self.totals
                    print(f"  {i}/{len(jobs)} done | parsed {t['parsed']}, rejected {t['rejected']} | tokens in {t['prompt_tokens']:,} out {t['completion_tokens']:,} | cost ${t['cost_usd']:.4f}", flush=True)
        t = self.totals
        if self.stop:
            print(f"Stopped: reported cost ${t['cost_usd']:.4f} reached the limit of ${self.a.max_cost}. Run the same command again to continue.")
        print(f"Finished in {time.time() - t0:.0f}s. Replies {t['requests']}, parsed {t['parsed']}, rejected {t['rejected']}, network or provider errors {t['transport_errors']}, reported cost ${t['cost_usd']:.4f}")
        finished = self.done_keys()
        left = sum(1 for e, s_ in jobs if (e["name"], e["seed"], e["condition"], s_) not in finished)
        if left:
            print(f"{left} prompt(s) have no final outcome yet. Run the same command again to continue them.")
        print(f"Forecasts: {self.forecasts}\nSettings:  {self.settings}\nReplies:   {self.replies}/")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prompts"); ap.add_argument("--run", required=True, help="a name for this run, used in output file names")
    ap.add_argument("--model", required=True); ap.add_argument("--base-url", default=OPENROUTER)
    ap.add_argument("--api-key-env", default="OPENROUTER_API_KEY", help="environment variable holding the key, or NONE if the server needs none")
    ap.add_argument("--temperature", type=float, default=1.0, help="Context is Key's Direct Prompt default is 1.0")
    ap.add_argument("--max-tokens", type=int, default=40000, help="max output tokens, thinking included (default 40000)")
    ap.add_argument("--seed", type=int, default=0, help="base sampling seed (default 0)")
    ap.add_argument("--reasoning-max-tokens", type=int, help="cap on thinking tokens, where the provider supports it")
    ap.add_argument("--reasoning-effort", help="thinking effort, for example low, where the provider supports it")
    ap.add_argument("--samples", type=int, default=1, help="forecasts per prompt"); ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4, help="requests in flight at once (default 4)")
    ap.add_argument("--timeout", type=int, default=3600, help="longest one reply may take in total, seconds (default 3600)")
    ap.add_argument("--idle-timeout", type=int, default=300, help="give up on a reply if nothing arrives for this many seconds (default 300)")
    ap.add_argument("--retry-waits", type=lambda v: [float(x) for x in v.split(",") if x], default=[10, 30, 60, 120, 300],
                    help="seconds to wait before each retry after network or provider trouble (default 10,30,60,120,300)")
    ap.add_argument("--provider", help="OpenRouter only: use just this provider"); ap.add_argument("--limit", type=int, help="only the first N prompts")
    ap.add_argument("--max-cost", type=float, help="stop once the reported cost in dollars reaches this")
    ap.add_argument("--extra", type=json.loads, default={}, help="extra request fields as JSON, e.g. '{\"reasoning\": {\"enabled\": false}}'")
    args = ap.parse_args()
    key = None
    if args.api_key_env != "NONE":
        if args.api_key_env not in os.environ:
            load_env_file(Path.cwd())
        key = os.environ.get(args.api_key_env)
        if not key:
            raise SystemExit(f"No API key: set {args.api_key_env} in the environment or in a .env file. If the server needs none, pass --api-key-env NONE.")
    Runner(args, key).run()


if __name__ == "__main__":
    main()
