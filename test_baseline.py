"""Tests for chronos_baseline.py and compare_forecasts.py. Run: python3 test_baseline.py

The first tests use a stand-in for the model, so they need no PyTorch. The last one uses the real
Chronos-2 model and only runs where it is installed (.venv/bin/python test_baseline.py).
"""
import json, re, tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from chronos_baseline import QUANTILES, forecast_records, history
from compare_forecasts import compare, parse_set, window_scores
from prompt_builder import make_prompt
from scorer import load_records

HERE = Path(__file__).resolve().parent


def record(seed, past, future, name="Fake"):
    start = datetime(2021, 10, 1, 6)
    stamp = lambda k: (start + timedelta(hours=k)).strftime("%Y-%m-%dT%H:%M:%S.000")
    series = lambda values, offset: json.dumps({"carbon_intensity": {stamp(offset + k): v for k, v in enumerate(values)}})
    return {"name": name, "seed": seed, "past_time": series(past, 0), "future_time": series(future, len(past)),
            "background": "Wind is rising.", "constraints": "", "scenario": "", "metric_scaling": 1.0}


PAST = [585.73, 604.0912345, 1608.14159, 0.000123456, 612.5, 599.99, 601.0, 640.25]
FUTURE = [610.0, 620.0, 630.0, 640.0]


def stand_in(calls):
    """A fake model: every quantile of every hour is the last past value plus the quantile level."""
    def predict(values, horizon):
        calls.append((list(values), horizon))
        return [[values[-1] + q for q in QUANTILES] for _ in range(horizon)]
    return predict


def t_same_numbers_as_prompt(d):
    rec = record(1, PAST, FUTURE)
    block = make_prompt(rec, False).split("<history>")[1].split("</history>")[0]
    shown = [float(m) for m in re.findall(r", ([-+0-9.e]+)\)", block)]
    return history(rec) == shown and len(shown) == len(PAST) and history(rec)[1] == 604.091


def t_answers_never_used(d):
    a, b = [], []
    out_a = list(forecast_records([record(1, PAST, FUTURE)], stand_in(a)))
    out_b = list(forecast_records([record(1, PAST, [9999.0, -5.0, 0.0, 123.0])], stand_in(b)))
    return a == b and out_a == out_b and all(v not in json.dumps(out_a) for v in ("610.0", "620.0", "630.0"))


def t_one_call_per_window(d):
    calls = []
    recs = [record(1, PAST, FUTURE), record(2, PAST[::-1], FUTURE[:3])]
    out = list(forecast_records(recs, stand_in(calls)))
    line, quant = out[0]
    return (len(calls) == 2 and calls[0][1] == 4 and calls[1][1] == 3 and len(calls[0][0]) == len(PAST)
            and line == {"name": "Fake", "seed": 1, "condition": "no_context", "sample": 0, "ok": True, "values": [640.25 + 0.5] * 4}
            and quant["quantile_levels"] == QUANTILES and len(quant["quantiles"]) == 4 and len(out[1][0]["values"]) == 3)


def t_wrong_shape_refused(d):
    try:
        list(forecast_records([record(1, PAST, FUTURE)], lambda values, horizon: [[1.0] * len(QUANTILES)] * (horizon - 1)))
    except ValueError:
        return True
    return False


def write(path, lines):
    Path(path).write_text("".join(json.dumps(l) + "\n" for l in lines)); return str(path)


def t_compare_hand_example(d):
    recs = [record(s, PAST * 3, FUTURE) for s in (1, 2, 3)]
    rpath = write(f"{d}/records.jsonl", recs); records = load_records(rpath)
    line = lambda seed, cond, off, ok=True: ({"name": "Fake", "seed": seed, "condition": cond, "sample": 0, "ok": True, "values": [v + off for v in FUTURE]}
                                              if ok else {"name": "Fake", "seed": seed, "condition": cond, "sample": 0, "ok": False, "reason": "no forecast block"})
    base = write(f"{d}/base.jsonl", [line(1, "no_context", 2), line(2, "no_context", 4), line(3, "no_context", 1)])
    llm = write(f"{d}/llm.jsonl", [line(1, "no_context", 10), line(1, "with_context", 6), line(2, "no_context", 8), line(2, "with_context", 12),
                                   line(3, "no_context", 5), line(3, "with_context", 0, ok=False)])
    sets = [("base", window_scores(records, base, "no_context")), ("llm_no", window_scores(records, llm, "no_context")),
            ("llm_with", window_scores(records, llm, "with_context"))]
    rep = compare(records, sets)
    two = compare(records, sets[:2])
    return (rep["windows_per_set"] == {"base": 3, "llm_no": 3, "llm_with": 2} and rep["windows_compared"] == 2      # window 3 dropped for all
            and np.allclose([rep["sets"]["base"], rep["sets"]["llm_no"], rep["sets"]["llm_with"]], [3.0, 9.0, 9.0])
            and np.isclose(rep["against_reference"]["llm_no"]["mean_difference"], -6.0)                               # the baseline was better
            and rep["against_reference"]["llm_no"]["windows_higher_error"] == 2
            and two["windows_compared"] == 3 and np.isclose(two["sets"]["base"], 7 / 3) and np.isclose(two["sets"]["llm_no"], 23 / 3)
            and "baselines" in rep and parse_set("a=results/x:y/forecasts.jsonl:no_context") == ("a", "results/x:y/forecasts.jsonl", "no_context"))


def t_real_model(d):
    """Only where Chronos-2 is installed: the forecast is repeatable and has 72 values for a real pilot window."""
    try:
        from chronos_baseline import load_chronos
        predict, info = load_chronos()
    except ImportError:
        return None
    pilot = HERE.parent / "text-tsfm-benchmarks/outputs/pilot/AECI_pilot_cik_format.jsonl"
    if not pilot.exists():
        return None
    rec = json.loads(pilot.read_text().split("\n")[0])
    (one, _), (two, _) = list(forecast_records([rec, rec], predict))
    past = history(rec)
    return (one["values"] == two["values"] and len(one["values"]) == 72 and all(np.isfinite(one["values"]))
            and min(past) * 0.5 < min(one["values"]) and max(one["values"]) < max(past) * 1.5 and len(info["snapshot"]) == 40)


TESTS = [
    ("the model is given exactly the numbers the prompt shows", t_same_numbers_as_prompt),
    ("the hidden answers are never used: changing them changes nothing", t_answers_never_used),
    ("one call per window, with that window's past values only; the median becomes the forecast", t_one_call_per_window),
    ("a forecast of the wrong shape is refused", t_wrong_shape_refused),
    ("comparison hand example: only shared windows count, and the difference has the right sign", t_compare_hand_example),
    ("real Chronos-2 model: repeatable, 72 finite values in a sensible range", t_real_model),
]


def main():
    fails = 0; skipped = 0
    for name, test in TESTS:
        with tempfile.TemporaryDirectory() as d:
            try:
                ok = test(d)
            except Exception as e:
                ok = False; name += f"  [{type(e).__name__}: {e}]"
        if ok is None:
            skipped += 1; print(f"skip | {name} (Chronos-2 is not installed for this Python)"); continue
        fails += not ok
        print(f"{'pass' if ok else 'FAIL'} | {name}")
    print(f"\n{len(TESTS) - skipped} tests, {fails} failed" + (f", {skipped} skipped" if skipped else ""))
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
