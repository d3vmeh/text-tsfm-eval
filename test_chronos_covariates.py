"""Tests for chronos_covariates.py that need no model. Run:  python3 test_chronos_covariates.py"""
import json
import numpy as np
from chronos_baseline import QUANTILES
from chronos_covariates import forecast_records
from scorer import score

def record(seed, answers=(620.0, 621.0, 622.0)):
    past = {f"2019-01-21T{h:02d}:00:00.000": 600.0 + h + seed for h in range(24)}
    fut = {f"2019-01-22T{h:02d}:00:00.000": v for h, v in zip((6, 7, 8), answers)}
    return {"name": "Toy", "seed": seed, "past_time": json.dumps({"x": past}), "future_time": json.dumps({"x": fut}), "metric_scaling": 0.01}

def cov(seed):
    return {"name": "Toy", "seed": seed, "variables": ["wind", "solar"], "alignment": "one_day_later",
            "past_covariates": {"wind": [float(seed)] * 24, "solar": [1.0] * 24}, "future_covariates": {"wind": [float(seed) + 10] * 3, "solar": [2.0] * 3}}

seen = []
def fake_predict(values, past_cov, future_cov, horizon):
    seen.append((list(values), {k: list(v) for k, v in past_cov.items()}, {k: list(v) for k, v in future_cov.items()}, horizon))
    return [[values[-1] + future_cov["wind"][i] * 0 + q for q in QUANTILES] for i in range(horizon)]

results = []
def check(name, cond):
    results.append(bool(cond)); print(f"{'pass' if cond else 'FAIL'} | {name}")

def main():
    recs = [record(1), record(2)]; covs = [cov(2), cov(1)]                       # covariates deliberately in another order
    out = list(forecast_records(recs, covs, fake_predict, ["wind"], "with_covariates"))
    check("each window gets its own covariates, matched by name and seed", [s[1]["wind"][0] for s in seen] == [1.0, 2.0] and [s[2]["wind"][0] for s in seen] == [11.0, 12.0])
    check("only the chosen variables are passed", all(set(s[1]) == {"wind"} and set(s[2]) == {"wind"} for s in seen))
    first = list(seen); seen.clear()
    list(forecast_records([record(1, (-1.0, -2.0, -3.0)), record(2, (-9.0, -9.0, -9.0))], covs, fake_predict, ["wind"], "with_covariates"))
    check("changing the hidden answers changes nothing the model is given", seen == first)
    line = out[0][0]
    check("the point forecast is the median, in the scorer's format", line["ok"] and line["condition"] == "with_covariates" and len(line["values"]) == 3 and abs(line["values"][0] - (624.0 + 0.5)) < 1e-9)
    try:
        list(forecast_records([record(3)], covs, fake_predict, ["wind"], "c")); missing = False
    except KeyError:
        missing = True
    bad = cov(1); bad["future_covariates"]["wind"] = [1.0]
    try:
        list(forecast_records([record(1)], [bad], fake_predict, ["wind"], "c")); wrong_len = False
    except ValueError:
        wrong_len = True
    check("a window with no covariates, or covariates of the wrong length, is refused", missing and wrong_len)
    records = {("Toy", r["seed"]): {"truth": np.array(list(json.loads(r["future_time"])["x"].values())), "past": np.array(list(json.loads(r["past_time"])["x"].values())), "scaling": 0.01} for r in recs}
    lines = [o[0] for o in out] + [{**o[0], "condition": "no_context"} for o in out]
    rep, _ = score(records, [{**l, "condition": "with_context"} if l["condition"] == "with_covariates" else l for l in lines])
    check("scorer.py reads the output", rep["paired_windows"] == 2)
    print(f"\n{len(results)} tests, {results.count(False)} failed")
    raise SystemExit(1 if not all(results) else 0)

if __name__ == "__main__":
    main()
