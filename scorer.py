"""Score forecasts against the hidden answers, and compare the with-text and without-text runs.

Inputs
  RECORDS.jsonl    Context is Key style records (they hold the hidden answers in future_time).
  FORECASTS.jsonl  One line per forecast attempt, written by the model runner:
                     {"name": ..., "seed": ..., "condition": "with_context" | "no_context", "sample": 0,
                      "ok": true, "values": [72 numbers]}
                   or, for a reply that could not be used:
                     {"name": ..., "seed": ..., "condition": ..., "sample": 0, "ok": false, "reason": "..."}

What it reports
  - MAE per window and condition. With several samples, the point forecast is the per-hour median.
  - A paired comparison on the windows where both conditions produced a forecast.
  - Last-value and daily-repeat baselines on those same windows.
  - How many replies failed, and why.
  - With two or more samples per window: CRPS, and CRPS x metric_scaling. For records with no region
    of interest and no constraints (ours), that product is the Context is Key score, RCRPS.

Usage:  python3 scorer.py RECORDS.jsonl FORECASTS.jsonl [--out results/NAME]
"""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CONDITIONS = ("with_context", "no_context")


def load_records(path):
    """Return {(name, seed): {"truth": array, "past": array, "scaling": float}}."""
    out = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            series = lambda field: np.array(list(list(json.loads(r[field]).values())[-1].values()), dtype=float)
            out[(r["name"], r["seed"])] = {"truth": series("future_time"), "past": series("past_time"), "scaling": r["metric_scaling"]}
    return out


def crps(target, samples):
    """CRPS per hour, the unbiased estimator used by the Context is Key code (needs 2+ samples).
    target: (hours,), samples: (n_samples, hours)."""
    n = samples.shape[0]
    assert n >= 2 and target.shape == samples.shape[1:]
    s = np.sort(samples, axis=0)
    abs_diff = np.abs(target[None, :] - s).sum(axis=0) / n
    beta0 = s.sum(axis=0) / n
    beta1 = (np.arange(n)[:, None] * s).sum(axis=0) / (n * (n - 1))
    return abs_diff + beta0 - 2 * beta1


def mae(forecast, truth):
    return float(np.abs(np.asarray(forecast, dtype=float) - truth).mean())


def baselines(rec):
    past, truth = rec["past"], rec["truth"]
    last = np.full(len(truth), past[-1])
    daily = np.array([past[-24 + (k % 24)] for k in range(len(truth))])
    return mae(last, truth), mae(daily, truth)


def paired_stats(a_without, a_with, n_boot=10000, seed=0):
    """a_without, a_with: scores on the same windows, lower is better. Difference > 0 means text helped."""
    a_without, a_with = np.asarray(a_without, dtype=float), np.asarray(a_with, dtype=float)
    d = a_without - a_with
    out = {"windows": int(len(d)), "mean_with_context": float(a_with.mean()), "mean_no_context": float(a_without.mean()),
           "mean_difference": float(d.mean()), "percent_improvement": float(100 * d.mean() / a_without.mean()) if a_without.mean() else 0.0,
           "windows_improved": int((d > 0).sum()), "windows_worse": int((d < 0).sum()), "windows_tied": int((d == 0).sum())}
    if len(d) >= 2:
        rng = np.random.default_rng(seed)
        means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        out["difference_ci95"] = [float(lo), float(hi)]
        out["ci_excludes_zero"] = bool(lo > 0 or hi < 0)
    return out


def score(records, forecast_lines):
    samples = defaultdict(list); failures = {c: Counter() for c in CONDITIONS}; unknown = 0
    for fc in forecast_lines:
        key = (fc["name"], fc["seed"]); cond = fc["condition"]
        if key not in records or cond not in CONDITIONS:
            unknown += 1; continue
        if fc.get("ok"):
            vals = np.asarray(fc["values"], dtype=float)
            if vals.shape != records[key]["truth"].shape or not np.isfinite(vals).all():
                failures[cond]["wrong length or non-finite values"] += 1; continue
            samples[(key, cond)].append(vals)
        else:
            failures[cond][fc.get("reason", "unspecified")[:60]] += 1
    per_window = {}
    for (key, cond), vals in samples.items():
        rec = records[key]; stack = np.stack(vals)
        row = per_window.setdefault(key, {})
        row[cond] = {"n_samples": len(vals), "mae": mae(np.median(stack, axis=0), rec["truth"])}
        if len(vals) >= 2:
            c = float(crps(rec["truth"], stack).mean())
            row[cond].update({"crps": c, "rcrps": rec["scaling"] * c})
    paired = sorted(k for k, row in per_window.items() if all(c in row for c in CONDITIONS))
    report = {"windows_in_records": len(records), "forecast_lines_for_unknown_windows": unknown,
              "per_condition": {c: {"windows_with_a_forecast": sum(1 for row in per_window.values() if c in row),
                                    "failed_replies": int(sum(failures[c].values())), "failure_reasons": dict(failures[c])} for c in CONDITIONS},
              "paired_windows": len(paired), "windows_dropped_from_comparison": len(per_window) - len(paired)}
    if paired:
        get = lambda cond, metric: [per_window[k][cond][metric] for k in paired]
        report["mae"] = paired_stats(get("no_context", "mae"), get("with_context", "mae"))
        b = np.array([baselines(records[k]) for k in paired])
        report["baselines_on_paired_windows"] = {"mae_last_value": float(b[:, 0].mean()), "mae_daily_repeat": float(b[:, 1].mean())}
        if all("rcrps" in per_window[k][c] for k in paired for c in CONDITIONS):
            report["rcrps"] = paired_stats(get("no_context", "rcrps"), get("with_context", "rcrps"))
    return report, per_window


def csv_safe(value):
    """Spreadsheets run a text cell that starts with = + - or @ as a formula. Names and ids come from the
    records file, so such text gets a leading apostrophe. Numbers are left alone."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def print_report(rep):
    print(f"Windows in records: {rep['windows_in_records']}")
    for c in CONDITIONS:
        pc = rep["per_condition"][c]
        print(f"  {c:<13} windows with a forecast: {pc['windows_with_a_forecast']}, failed replies: {pc['failed_replies']} {pc['failure_reasons'] or ''}")
    print(f"Windows compared (a forecast in both conditions): {rep['paired_windows']}, dropped: {rep['windows_dropped_from_comparison']}")
    for metric in ("mae", "rcrps"):
        if metric not in rep:
            continue
        m = rep[metric]; ci = m.get("difference_ci95")
        print(f"\n{metric.upper()} (lower is better)")
        print(f"  with text:    {m['mean_with_context']:.4f}")
        print(f"  without text: {m['mean_no_context']:.4f}")
        print(f"  difference:   {m['mean_difference']:+.4f} ({m['percent_improvement']:+.1f}%; positive means text helped)"
              + (f", 95% interval {ci[0]:+.4f} to {ci[1]:+.4f}" if ci else ""))
        print(f"  windows: {m['windows_improved']} better with text, {m['windows_worse']} worse, {m['windows_tied']} tied")
        if ci:
            print("  the interval " + ("excludes zero, so the difference is unlikely to be chance" if m["ci_excludes_zero"] else "includes zero, so this could be chance"))
    if "baselines_on_paired_windows" in rep:
        b = rep["baselines_on_paired_windows"]
        print(f"\nBaselines on the same windows (MAE): last-value {b['mae_last_value']:.4f}, daily-repeat {b['mae_daily_repeat']:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records"); ap.add_argument("forecasts"); ap.add_argument("--out")
    args = ap.parse_args()
    records = load_records(args.records)
    with open(args.forecasts) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    rep, per_window = score(records, lines)
    print_report(rep)
    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        Path(str(out) + "_report.json").write_text(json.dumps(rep, indent=2) + "\n")
        with open(str(out) + "_per_window.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["name", "seed", "condition", "n_samples", "mae", "crps", "rcrps", "mae_last_value", "mae_daily_repeat"])
            for key in sorted(per_window):
                bl = baselines(records[key])
                for c in CONDITIONS:
                    if c in per_window[key]:
                        r = per_window[key][c]; w.writerow([csv_safe(key[0]), csv_safe(key[1]), c, r["n_samples"], r["mae"], r.get("crps", ""), r.get("rcrps", ""), bl[0], bl[1]])
        print(f"\nWrote {out}_report.json and {out}_per_window.csv")


if __name__ == "__main__":
    main()
