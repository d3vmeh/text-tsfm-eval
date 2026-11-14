"""Compare several forecast sets on exactly the same windows.

scorer.py compares the with-text and without-text prompts of one run. This script puts any number of
forecast sets side by side, for example a baseline model next to an LLM run. Each set is one
forecasts file and one condition in it. Only windows where every set has a forecast are used, so no
set gets an easier group of windows than another.

It uses scorer.py's own functions for the scores, the baselines and the 95% interval.

Usage:
  python3 compare_forecasts.py RECORDS.jsonl NAME=FORECASTS.jsonl:CONDITION [NAME=FORECASTS.jsonl:CONDITION ...] [--out FILE.json]

Example:
  python3 compare_forecasts.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl \\
      chronos2=results/validation_chronos2/forecasts.jsonl:no_context \\
      llm_no_text=results/validation_4090_think6000/forecasts.jsonl:no_context \\
      llm_with_text=results/validation_4090_think6000/forecasts.jsonl:with_context

The first set is the reference. For every other set it reports reference minus that set, so a
positive difference means that set had the lower error.
"""
import argparse, json
from pathlib import Path

import numpy as np

from scorer import CONDITIONS, baselines, load_records, paired_stats, score


def window_scores(records, path, condition):
    """{(name, seed): MAE} for one condition of one forecasts file. Rejected replies give no score."""
    with open(path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    _report, per_window = score(records, [l for l in lines if l.get("condition") == condition])
    return {key: row[condition]["mae"] for key, row in per_window.items() if condition in row}


def compare(records, sets):
    """sets: [(name, {(name, seed): MAE})]. Returns the report as a dict."""
    shared = sorted(set.intersection(*(set(scores) for _name, scores in sets)))
    report = {"windows_in_records": len(records), "windows_per_set": {name: len(scores) for name, scores in sets},
              "windows_compared": len(shared), "sets": {}, "against_reference": {}}
    if not shared:
        return report
    for name, scores in sets:
        report["sets"][name] = float(np.mean([scores[k] for k in shared]))
    b = np.array([baselines(records[k]) for k in shared])
    report["baselines"] = {"last_value": float(b[:, 0].mean()), "daily_repeat": float(b[:, 1].mean())}
    ref_name, ref = sets[0]
    for name, scores in sets[1:]:
        stats = paired_stats([ref[k] for k in shared], [scores[k] for k in shared])
        report["against_reference"][name] = {"reference": ref_name, "mean_difference": stats["mean_difference"],
                                             "difference_ci95": stats.get("difference_ci95"), "ci_excludes_zero": stats.get("ci_excludes_zero"),
                                             "windows_lower_error": stats["windows_improved"], "windows_higher_error": stats["windows_worse"]}
    return report


def print_report(rep):
    print(f"Windows in records: {rep['windows_in_records']}")
    for name, n in rep["windows_per_set"].items():
        print(f"  {name:<24} windows with a forecast: {n}")
    print(f"Windows compared (a forecast in every set): {rep['windows_compared']}")
    if not rep["windows_compared"]:
        return
    print("\nMAE on those windows (lower is better)")
    for name, value in rep["sets"].items():
        print(f"  {name:<24} {value:.4f}")
    print(f"  {'last-value baseline':<24} {rep['baselines']['last_value']:.4f}")
    print(f"  {'daily-repeat baseline':<24} {rep['baselines']['daily_repeat']:.4f}")
    for name, d in rep["against_reference"].items():
        ci = d["difference_ci95"]
        print(f"\n{d['reference']} minus {name}: {d['mean_difference']:+.4f} (positive means {name} had the lower error)"
              + (f", 95% interval {ci[0]:+.4f} to {ci[1]:+.4f}" if ci else ""))
        print(f"  windows: {name} lower on {d['windows_lower_error']}, higher on {d['windows_higher_error']}")
        if ci:
            print("  the interval " + ("excludes zero, so the difference is unlikely to be chance" if d["ci_excludes_zero"] else "includes zero, so this could be chance"))


def parse_set(text):
    name, _, rest = text.partition("=")
    path, _, condition = rest.rpartition(":")
    if not name or not path or condition not in CONDITIONS:
        raise SystemExit(f"cannot read {text!r}: write NAME=FORECASTS.jsonl:with_context or NAME=FORECASTS.jsonl:no_context")
    return name, path, condition


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records"); ap.add_argument("sets", nargs="+"); ap.add_argument("--out")
    args = ap.parse_args(argv)
    records = load_records(args.records)
    sets = [(name, window_scores(records, path, condition)) for name, path, condition in map(parse_set, args.sets)]
    rep = compare(records, sets)
    print_report(rep)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
