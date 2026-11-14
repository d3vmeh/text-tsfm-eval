"""Tests for reply_parser.py. Run:  python3 test_reply_parser.py"""
import json, re
from pathlib import Path
from reply_parser import parse_reply, parse_reply_cik, forecast_regex

HERE = Path(__file__).resolve().parent
TS = [f"2019-01-22 {h:02d}:00:00" for h in range(6, 12)]          # six timestamps for small cases
VALS = [608.14, 615.55, 618.57, 633.65, 642.39, 640.79]
GOOD = "<forecast>\n" + "\n".join(f"({t}, {v})" for t, v in zip(TS, VALS)) + "\n</forecast>"

def cik(text, ts=TS):
    try: return parse_reply_cik(text, ts)
    except Exception: return None

CASES = [  # name, reply, should parse?, does CiK's rule accept it?
    ("well formed", GOOD, True, True),
    ("text before and after the tags", "Here is my forecast.\n" + GOOD + "\nHope this helps.", True, True),
    ("quotes around timestamps", GOOD.replace("(2019", "('2019").replace(":00,", ":00',"), True, True),
    ("scientific notation and a negative", GOOD.replace("608.14", "6.0814e2").replace("615.55", "-615.55"), True, True),
    ("extra timestamps are ignored", GOOD.replace("</forecast>", "(2019-01-22 12:00:00, 1.0)\n</forecast>"), True, True),
    ("think block that quotes the tags", "<think>I must answer in <forecast>(t4, v4)</forecast> form.</think>\n" + GOOD, True, False),
    ("blank line inside the block", GOOD.replace("\n(2019-01-22 08", "\n\n(2019-01-22 08"), True, False),
    ("spaces around the timestamp", GOOD.replace("(2019-01-22 06:00:00,", "( 2019-01-22 06:00:00 ,"), True, False),
    ("no tags at all", "\n".join(f"({t}, {v})" for t, v in zip(TS, VALS)), False, False),
    ("block never closed (cut off)", GOOD.replace("\n</forecast>", ""), False, False),
    ("one timestamp missing", GOOD.replace("(2019-01-22 09:00:00, 633.65)\n", ""), False, False),
    ("a timestamp shifted by a day", GOOD.replace("2019-01-22 09", "2019-01-23 09"), False, False),
    ("commentary inside the block", GOOD.replace("<forecast>\n", "<forecast>\nValues below\n"), False, False),
    ("value is a word", GOOD.replace("633.65", "high"), False, False),
    ("thousands separator", GOOD.replace("608.14", "1,608.14"), False, True),
    ("nan value", GOOD.replace("608.14", "nan"), False, True),
    ("empty reply", "", False, False),
    ("only the prompt's example echoed first", "<forecast>\n(t4, v4)\n(t5, v5)\n</forecast>\n" + GOOD, False, False),
]

def main():
    fails = 0
    for name, reply, want_ok, want_cik in CASES:
        r = parse_reply(reply, TS); c = cik(reply)
        ok = (r.ok == want_ok) and ((c is not None) == want_cik)
        if r.ok and c is not None and "nan" not in reply:
            ok = ok and r.values == c                                   # where both accept, they must agree
        if r.ok and name == "well formed":
            ok = ok and r.values == VALS
        fails += not ok
        print(f"{'pass' if ok else 'FAIL'} | {name:<42} | parsed: {str(r.ok):<5} | CiK rule accepts: {str(c is not None):<5} | {r.reason}")
    # Realistic case: a perfect 72-line reply for a real pilot window must give back exactly the values written.
    prompts = HERE / "prompts/AECI_pilot_prompts.jsonl"
    if prompts.exists():
        entry = json.loads(prompts.read_text().split("\n")[0]); ts = entry["future_timestamps"]
        vals = [500 + 0.25 * k for k in range(len(ts))]
        reply = "<forecast>\n" + "\n".join(f"({t}, {v})" for t, v in zip(ts, vals)) + "\n</forecast>"
        r = parse_reply(reply, ts)
        ok = r.ok and r.values == vals and len(r.values) == 72 and re.fullmatch(forecast_regex(ts), reply) is not None
        ok = ok and re.fullmatch(forecast_regex(ts), reply.replace(ts[5], "2019-01-30 00:00:00")) is None
        fails += not ok
        print(f"{'pass' if ok else 'FAIL'} | real pilot window, 72 timestamps, and the forcing pattern accepts good / rejects bad")
    print(f"\n{len(CASES) + 1} tests, {fails} failed")
    raise SystemExit(1 if fails else 0)

if __name__ == "__main__":
    main()
