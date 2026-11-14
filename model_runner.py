"""Send prompts to a local llama.cpp server and record the forecasts.

For each prompt made by prompt_builder.py it asks the server for a reply, reads the reply with
parse_reply() and writes one JSON line per attempt, in the format scorer.py expects.

Outputs, for --run NAME
  results/NAME/forecasts.jsonl   one line per attempt. Give this file to scorer.py.
  results/NAME/settings.json     model file, quantization, context size, temperature, seed,
                                 max output tokens and the other settings used.
  replies/NAME/                  every raw reply: a .json with the server's full answer, a .txt with
                                 the text given to the parser, and a .thinking.txt if the model thought.

What it does to protect the results
  - Context size. Before each request it counts the prompt's tokens on the server and stops if
    prompt + max output tokens does not fit the server's context window. After each reply it checks
    that the server really read the whole prompt.
  - Same settings for both conditions. Every request gets the same settings; only the messages
    differ. Repetition penalties are switched off, because a forecast is mostly repeated digits.
  - Retries. A rejected reply is asked for again with a new sampling seed, up to --max-attempts.
    At temperature 0 the reply would be identical, so there it is asked for once.
  - Stop and resume. Each attempt is saved as soon as it finishes. Run the same command again and
    finished prompts are skipped. Resuming with different settings is refused.
  - Server trouble (not running, crashed, timed out) is never recorded as a model failure. The runner
    waits and tries again, then stops so the run can be resumed later.
  - Several requests at once (--workers). The server's slots share one context, so the runner checks
    that all requests in flight fit together. Results from every worker go to the same files.

The window's "seed" in each line is the window's id from the records. The sampling seed sent to the
server is "request_seed" = --seed + 1000 x sample + attempt.

Usage:
  export LLAMA_API_KEY=...     (only if the server was started with --api-key)
  python3 model_runner.py prompts/AECI_pilot_prompts.jsonl --run pilot --temperature 1.0
"""
import argparse, hashlib, http.client, json, os, queue, re, sys, threading, time, urllib.error, urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from reply_parser import parse_reply

TEMPLATE_MARGIN = 256      # tokens allowed for the chat template that wraps the messages
TRIM_SLACK = 16            # chat templates may trim blank lines at the ends of a message
WAITS = (10, 30, 60, 120, 300)   # seconds to wait between tries when the server is in trouble
REQUIRED = ("name", "seed", "condition", "messages", "future_timestamps")
# Settings that must not change between the first run and a resumed run. (--samples may be raised later.)
FINGERPRINT = ("prompts_sha256", "model_file", "context_size", "server_build", "chat_template_sha256", "temperature",
               "seed", "max_output_tokens", "max_attempts", "top_p", "top_k", "repeat_penalty",
               "presence_penalty", "frequency_penalty", "cache_prompt", "server_note")


class StopRun(Exception):
    """A problem that makes more requests pointless or unsafe. The run stops and can be resumed."""


class ServerTrouble(Exception):
    """The server did not answer properly. Worth waiting and trying again."""


def call(url, body=None, api_key=None, timeout=60):
    """One HTTP request. GET without a body, POST with one. Returns the parsed JSON."""
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="GET" if body is None else "POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        if e.code in (401, 403):
            raise StopRun(f"the server refused the API key (HTTP {e.code}). Set it with: export LLAMA_API_KEY=...")
        if e.code >= 500:                                     # includes 503 while the model is loading
            raise ServerTrouble(f"HTTP {e.code}: {detail}")
        raise StopRun(f"the server rejected the request (HTTP {e.code}): {detail}")
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:      # refused, reset, timed out
        raise ServerTrouble(f"{type(e).__name__}: {e}")
    except ValueError:
        raise ServerTrouble("the server's answer was not JSON")


def patiently(what, fn):
    """Run fn. If the server is in trouble, wait and try again a few times, then stop the run."""
    for wait in (*WAITS, None):
        try:
            return fn()
        except ServerTrouble as e:
            if wait is None:
                raise StopRun(f"the server is not answering ({what}): {e}")
            print(f"  server trouble during {what}: {e}. Waiting {wait} s, then trying again.", flush=True)
            time.sleep(wait)


def quantization_from_name(model_file):
    """Read the quantization from the file name: 'Qwen3.8-27B-UD-Q4_K_XL.gguf' -> 'UD-Q4_K_XL'. '' if not found."""
    found = re.findall(r"(?<![A-Za-z0-9])((?:UD-)?(?:I?Q\d+(?:_[A-Za-z0-9]+)*|BF16|F16|F32))(?![A-Za-z0-9_])",
                       Path(model_file).name, flags=re.IGNORECASE)
    return found[-1] if found else ""


def sha256(data):
    return hashlib.sha256(data if isinstance(data, bytes) else str(data).encode("utf-8")).hexdigest()


def describe_server(base_url, api_key):
    """Ask the server what it is running. These come from the server, not from what was typed."""
    props = call(base_url + "/props", api_key=api_key)
    defaults = props.get("default_generation_settings") or {}
    model_file = props.get("model_path") or ""
    info = {"model_file": model_file, "quantization": quantization_from_name(model_file),
            "quantization_reported_by_server": props.get("model_ftype"),
            "context_size": defaults.get("n_ctx"), "slots": props.get("total_slots"),
            "server_build": props.get("build_info"), "chat_template_sha256": sha256(props.get("chat_template") or ""),
            "server_default_sampling": defaults.get("params")}
    try:                                                      # model size, if the server offers it
        info["model_meta"] = (call(base_url + "/v1/models", api_key=api_key).get("data") or [{}])[0].get("meta")
    except (StopRun, ServerTrouble):
        info["model_meta"] = None
    return info


def count_tokens(base_url, api_key, messages):
    """Tokens in the message texts alone. The real prompt is this plus the chat template."""
    return sum(len(call(base_url + "/tokenize", {"content": m["content"]}, api_key)["tokens"]) for m in messages)


def load_prompts(path):
    entries = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            missing = [k for k in REQUIRED if k not in entry]
            if missing:
                raise StopRun(f"{path} line {n} is missing {missing}. Was it made by prompt_builder.py?")
            entries.append(entry)
    if not entries:
        raise StopRun(f"{path} has no prompts")
    return entries


def pick_shard(entries, shard):
    """'I/N' keeps every N-th window, starting at window I. Both prompts of a window stay together,
    so a window's with-text and without-text replies always come from the same server."""
    if not shard:
        return entries
    m = re.fullmatch(r"(\d+)/(\d+)", shard)
    if not m or not int(m.group(1)) < int(m.group(2)):
        raise StopRun("--shard must look like 0/2 or 1/2")
    index, of = int(m.group(1)), int(m.group(2))
    order = {}
    for e in entries:
        order.setdefault((e["name"], e["seed"]), len(order))
    return [e for e in entries if order[(e["name"], e["seed"])] % of == index]


def load_done(path):
    """What the forecasts file already holds: {(name, seed, condition, sample): {"ok", "attempts"}}.
    A half-written last line (the run was killed mid-write) is moved to FILE.partial."""
    done = {}
    if not path.exists():
        return done
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        with open(str(path) + ".partial", "ab") as f:
            f.write(raw[cut:] + b"\n")
        with open(path, "r+b") as f:
            f.truncate(cut)
        print(f"  note: moved a half-written last line of {path} to {path}.partial")
        raw = raw[:cut]
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        state = done.setdefault((r["name"], r["seed"], r["condition"], r.get("sample", 0)), {"ok": False, "attempts": 0})
        state["attempts"] += 1
        state["ok"] = state["ok"] or bool(r.get("ok"))
    return done


def safe_part(value):
    """Names come from the records file, so they are cleaned before being used in a file name."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))[:80] or "x"


def save_reply(folder, stem, record, content, thinking):
    """Save one raw reply. Returns the .json path."""
    folder = folder.resolve()
    dest = (folder / f"{stem}.json").resolve()
    if dest.parent != folder:
        raise StopRun(f"refusing to write outside {folder}")
    dest.write_text(json.dumps(record, indent=1) + "\n")
    (folder / f"{stem}.txt").write_text(content)
    if thinking:
        (folder / f"{stem}.thinking.txt").write_text(thinking)
    return dest


def append_line(path, line):
    with open(path, "a") as f:
        f.write(json.dumps(line) + "\n")
        f.flush(); os.fsync(f.fileno())


def check_settings(path, settings, session):
    """Write settings.json, or on a resume check that nothing important changed."""
    if path.exists():
        old = json.loads(path.read_text())
        changed = [f"{k}: was {old.get(k)!r}, now {settings.get(k)!r}" for k in FINGERPRINT if old.get(k) != settings.get(k)]
        if changed:
            raise StopRun("this run was started with different settings, so resuming would mix them:\n    "
                          + "\n    ".join(changed) + "\n  Use a new --run name, or restore the old settings.")
        settings = {**old, "samples": max(old.get("samples", 1), settings["samples"]), "sessions": old.get("sessions", []) + [session]}
    else:
        settings = {**settings, "sessions": [session]}
    path.write_text(json.dumps(settings, indent=2) + "\n")


def build_request(messages, args, request_seed):
    """Everything sent to the server. Only `messages` differs between the two conditions."""
    body = {"messages": messages, "temperature": args.temperature, "seed": request_seed, "max_tokens": args.max_tokens,
            "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "cache_prompt": False, "stream": False}
    if args.top_p is not None:
        body["top_p"] = args.top_p
    if args.top_k is not None:
        body["top_k"] = args.top_k
    return body


def run(args):
    base_url = args.base_url.rstrip("/")
    api_key = os.environ.get(args.api_key_env) or None
    prompts_path = Path(args.prompts)
    entries = pick_shard(load_prompts(prompts_path), args.shard)
    if args.limit:
        entries = entries[:args.limit]
    if args.max_attempts < 1 or args.samples < 1 or args.workers < 1:
        raise StopRun("--max-attempts, --samples and --workers must be at least 1")

    server = patiently("reading the server's settings", lambda: describe_server(base_url, api_key))
    if not server["context_size"]:
        raise StopRun("the server did not report its context size, so the prompt cannot be checked against it")
    if not server["model_file"]:
        raise StopRun("the server did not report which model file it loaded")
    n_ctx = int(server["context_size"])
    settings = {**server, "prompts_file": str(prompts_path), "prompts_sha256": sha256(prompts_path.read_bytes()),
                "temperature": args.temperature, "seed": args.seed, "max_output_tokens": args.max_tokens,
                "max_attempts": args.max_attempts, "samples": args.samples, "top_p": args.top_p, "top_k": args.top_k,
                "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "cache_prompt": False,
                "server_note": args.server_note,
                "note": "top_p and top_k of null mean the server's own default was used (see server_default_sampling)"}
    out_dir = Path(args.results_dir) / safe_part(args.run); replies_dir = Path(args.replies_dir) / safe_part(args.run)
    out_dir.mkdir(parents=True, exist_ok=True); replies_dir.mkdir(parents=True, exist_ok=True)
    session = {"started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "base_url": base_url,
               "shard": args.shard, "limit": args.limit, "workers": args.workers, "api_key_used": bool(api_key)}
    check_settings(out_dir / "settings.json", settings, session)
    forecasts_path = out_dir / "forecasts.jsonl"
    done = load_done(forecasts_path)

    attempts_allowed = args.max_attempts if args.temperature > 0 else 1
    print(f"Model file:   {server['model_file']}  (quantization: {server['quantization'] or 'not in the file name'})")
    print(f"Context size: {n_ctx} tokens, {server['slots']} slot(s), {args.workers} request(s) at once. Server build: {server['server_build']}")
    if server["slots"] and args.workers > server["slots"]:
        raise StopRun(f"--workers {args.workers} is more than the server's {server['slots']} slots")
    print(f"Settings:     temperature {args.temperature}, seed {args.seed}, max output tokens {args.max_tokens}, "
          f"up to {attempts_allowed} attempt(s), {args.samples} sample(s) per prompt")
    if args.temperature <= 0 and args.max_attempts > 1:
        print("  note: at temperature 0 a retry gives the same reply, so rejected replies are not retried")

    jobs = [(e, s) for s in range(args.samples) for e in entries]     # finish sample 0 for every prompt first
    finished = lambda key: key in done and (done[key]["ok"] or done[key]["attempts"] >= attempts_allowed)
    todo = [(e, s) for e, s in jobs if not finished((e["name"], e["seed"], e["condition"], s))]
    print(f"Prompts:      {len(jobs)} in this run, {len(jobs) - len(todo)} already finished, {len(todo)} to do\n")

    # Several workers can each have one request in flight. Shared state is only touched while holding `lock`.
    lock = threading.Lock(); stop = threading.Event(); problems = []
    raw_tokens = {}; outcomes = Counter(); tally = {"cut_short": 0, "finished": 0}
    session_start = time.time()
    waiting = queue.Queue()
    for n, (entry, sample) in enumerate(todo, 1):
        waiting.put((n, entry, sample))

    def do_job(n, entry, sample):
        key = (entry["name"], entry["seed"], entry["condition"], sample)
        label = f"[{n}/{len(todo)}] {entry['name']} {entry['seed']} {entry['condition']} sample {sample}"
        tok_key = key[:3]
        with lock:
            counted = raw_tokens.get(tok_key)
        if counted is None:
            counted = patiently("counting tokens", lambda: count_tokens(base_url, api_key, entry["messages"]))
            with lock:
                raw_tokens[tok_key] = counted
        need = (counted + TEMPLATE_MARGIN + args.max_tokens) * args.workers
        if need > n_ctx:
            raise StopRun(f"{label}: the prompt is {counted} tokens. With {args.max_tokens} output tokens and {TEMPLATE_MARGIN} "
                          f"for the chat template, {args.workers} request(s) at once need {need}, but the server's context is "
                          f"{n_ctx}. Nothing was sent. Use a larger server context, fewer workers or fewer output tokens.")
        with lock:
            first_attempt = done.get(key, {"attempts": 0})["attempts"]
        ok = False
        for attempt in range(first_attempt, attempts_allowed):
            if stop.is_set():
                return
            request_seed = args.seed + 1000 * sample + attempt
            body = build_request(entry["messages"], args, request_seed)
            t0 = time.time()
            reply = patiently("a forecast request", lambda: call(base_url + "/v1/chat/completions", body, api_key, args.timeout))
            seconds = round(time.time() - t0, 2)
            try:
                choice = reply["choices"][0]; message = choice.get("message") or {}
            except (KeyError, IndexError, TypeError):
                raise StopRun(f"{label}: the server's answer had no choices: {str(reply)[:300]}")
            content = message.get("content") or ""; thinking = message.get("reasoning_content") or ""
            usage = reply.get("usage") or {}
            prompt_tokens, completion_tokens = usage.get("prompt_tokens"), usage.get("completion_tokens")
            finish = choice.get("finish_reason")
            stem = f"{safe_part(entry['name'])}_{safe_part(entry['seed'])}_{safe_part(entry['condition'])}_s{sample}_a{attempt}"
            result = parse_reply(content, entry["future_timestamps"])
            record = {"name": entry["name"], "seed": entry["seed"], "condition": entry["condition"], "sample": sample,
                      "attempt": attempt, "request": {k: v for k, v in body.items() if k != "messages"},
                      "prompt_tokens_counted_before_sending": counted, "seconds": seconds,
                      "parse": {"ok": result.ok, "reason": result.reason, "notes": result.notes,
                                "extra_timestamps": result.extra_timestamps}, "server_reply": reply}
            reply_file = save_reply(replies_dir, stem, record, content, thinking)
            if prompt_tokens is None or prompt_tokens < counted - TRIM_SLACK:
                raise StopRun(f"{label}: the server says it read {prompt_tokens} prompt tokens, but the prompt is "
                              f"{counted} tokens. The prompt may have been cut. This attempt was not recorded "
                              f"as a forecast. Raw reply: {reply_file}")
            line = {"name": entry["name"], "seed": entry["seed"], "condition": entry["condition"], "sample": sample, "ok": result.ok}
            line.update({"values": result.values} if result.ok else {"reason": result.reason})
            line.update({"attempt": attempt, "request_seed": request_seed, "finish_reason": finish,
                         "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens, "seconds": seconds,
                         "thinking_chars": len(thinking), "reply_file": str(reply_file)})
            with lock:
                append_line(forecasts_path, line)
                state = done.setdefault(key, {"ok": False, "attempts": 0})
                state["attempts"] += 1; state["ok"] = state["ok"] or result.ok
                tally["cut_short"] += finish == "length"
                print(f"{label} attempt {attempt}: {'ok' if result.ok else 'REJECTED: ' + result.reason} | "
                      f"{prompt_tokens} prompt + {completion_tokens} output tokens, {seconds:.0f} s"
                      + (" | CUT OFF at max output tokens" if finish == "length" else ""), flush=True)
            if result.ok:
                ok = True; break
        with lock:
            outcomes["ok" if ok else "failed"] += 1
            tally["finished"] += 1
            left = len(todo) - tally["finished"]
            if left and tally["finished"] % 10 == 0:
                each = (time.time() - session_start) / tally["finished"]
                print(f"  {tally['finished']} of {len(todo)} prompts finished, one every {each:.0f} s, roughly {each * left / 3600:.1f} hours left", flush=True)

    def worker():
        while not stop.is_set():
            try:
                job = waiting.get_nowait()
            except queue.Empty:
                return
            try:
                do_job(*job)
            except StopRun as e:
                problems.append(e); stop.set()
            except Exception as e:                    # a bug must stop the run, not silently lose a worker
                problems.append(StopRun(f"unexpected error in the runner: {type(e).__name__}: {e}")); stop.set()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads) and not stop.is_set():
        for t in threads:
            t.join(timeout=0.2)
    if problems:
        lock.acquire()                                # no worker may be mid-write while the run stops
        raise problems[0]

    spent = time.time() - session_start
    print(f"\nDone. This session: {outcomes['ok']} prompts with a forecast, {outcomes['failed']} without, "
          f"{tally['cut_short']} replies cut off at max output tokens.")
    if tally["finished"]:
        print(f"Time: {spent / 60:.1f} minutes, one prompt every {spent / tally['finished']:.0f} s.")
    print(f"Forecasts: {forecasts_path}\nSettings:  {out_dir / 'settings.json'}\nReplies:   {replies_dir}/")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prompts", help="prompts file made by prompt_builder.py")
    ap.add_argument("--run", required=True, help="a name for this run; outputs go in results/NAME and replies/NAME")
    ap.add_argument("--temperature", type=float, required=True, help="sampling temperature (0 = always the most likely token)")
    ap.add_argument("--seed", type=int, default=0, help="base sampling seed (default 0)")
    ap.add_argument("--max-tokens", type=int, default=12000, help="max output tokens, thinking included (default 12000)")
    ap.add_argument("--max-attempts", type=int, default=3, help="attempts per sample before giving up (default 3)")
    ap.add_argument("--samples", type=int, default=1, help="forecasts wanted per prompt (default 1)")
    ap.add_argument("--top-p", type=float, help="leave out to use the server's default")
    ap.add_argument("--top-k", type=int, help="leave out to use the server's default")
    ap.add_argument("--base-url", default="http://127.0.0.1:9999", help="where the llama.cpp server listens")
    ap.add_argument("--api-key-env", default="LLAMA_API_KEY", help="environment variable holding the API key, if any")
    ap.add_argument("--timeout", type=float, default=3600, help="seconds to wait for one reply (default 3600)")
    ap.add_argument("--server-note", default="", help="how the server was started (its launch flags). The server does not "
                    "report these, so write them here to have them in settings.json")
    ap.add_argument("--limit", type=int, help="only the first N prompts (for a quick trial)")
    ap.add_argument("--shard", help="I/N: only every N-th window starting at I, to split a run over servers")
    ap.add_argument("--workers", type=int, default=1, help="requests to have in flight at once on this server (default 1). "
                    "The server's context must hold all of them together")
    ap.add_argument("--results-dir", default="results"); ap.add_argument("--replies-dir", default="replies")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)        # show progress at once, also when saved to a log file
    try:
        run(args)
    except StopRun as e:
        print(f"\nSTOPPED: {e}\nFinished attempts are saved. Fix the problem and run the same command to continue.")
        return 2
    except KeyboardInterrupt:
        print("\nStopped by you. Finished attempts are saved. Run the same command to continue.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
