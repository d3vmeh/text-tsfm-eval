"""Tests for scorer.py. Run:  python3 test_scorer.py"""
import json
from pathlib import Path
import numpy as np
from scorer import load_records, score, crps, paired_stats, csv_safe

HERE = Path(__file__).resolve().parent
VALIDATION = HERE / "../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl"
BUILD_REPORT = HERE / "../text-tsfm-benchmarks/outputs/full/AECI_build_report.json"

def tiny_records():
    """Two made-up windows with a 3-hour horizon."""
    recs = {}
    for seed, truth in ((1, [600, 610, 620]), (2, [500, 505, 510])):
        recs[("Toy", seed)] = {"truth": np.array(truth, float), "past": np.arange(100, 148, dtype=float), "scaling": 0.01}
    return recs

def line(seed, cond, values=None, reason=None, sample=0):
    base = {"name": "Toy", "seed": seed, "condition": cond, "sample": sample}
    return {**base, "ok": True, "values": values} if values is not None else {**base, "ok": False, "reason": reason}

results = []
def check(name, cond):
    results.append(bool(cond)); print(f"{'pass' if cond else 'FAIL'} | {name}")

def main():
    recs = tiny_records()
    # 1. The worked example: errors 10, 5, 10 with text and 20, 30, 20 without.
    rep, _ = score(recs, [line(1, "with_context", [590, 615, 630]), line(1, "no_context", [620, 640, 600]),
                          line(2, "with_context", [500, 505, 510]), line(2, "no_context", [500, 505, 510])])
    m = rep["mae"]
    check("worked example: 8.33 with text, 23.33 without, on window 1", abs(m["mean_with_context"] - (25/3 + 0) / 2) < 1e-9 and abs(m["mean_no_context"] - (70/3 + 0) / 2) < 1e-9)
    check("difference is positive when text helps, with 1 better and 1 tied", m["mean_difference"] > 0 and m["windows_improved"] == 1 and m["windows_tied"] == 1 and m["windows_worse"] == 0)
    check("percent improvement = 64.3%", abs(m["percent_improvement"] - 100 * 15 / (70/3)) < 1e-9)
    # 2. The true answers as the forecast must score exactly zero.
    rep, _ = score(recs, [line(s, c, list(recs[("Toy", s)]["truth"])) for s in (1, 2) for c in ("with_context", "no_context")])
    check("true answers as the forecast score 0", rep["mae"]["mean_with_context"] == 0 and rep["mae"]["mean_no_context"] == 0)
    # 3. Failed replies are counted, and that window leaves the comparison.
    rep, _ = score(recs, [line(1, "with_context", [590, 615, 630]), line(1, "no_context", reason="missing 2 of 3 timestamps"),
                          line(2, "with_context", [1, 2, 3]), line(2, "no_context", [1, 2, 3])])
    check("a failed reply is counted with its reason and its window is dropped", rep["per_condition"]["no_context"]["failed_replies"] == 1
          and rep["paired_windows"] == 1 and rep["windows_dropped_from_comparison"] == 1 and "missing 2 of 3 timestamps" in rep["per_condition"]["no_context"]["failure_reasons"])
    # 4. Wrong length and non-finite forecasts are refused.
    rep, _ = score(recs, [line(1, "with_context", [1, 2]), line(1, "no_context", [1, 2, float("nan")])])
    check("wrong-length and nan forecasts are refused", rep["paired_windows"] == 0 and rep["per_condition"]["with_context"]["failed_replies"] == 1 and rep["per_condition"]["no_context"]["failed_replies"] == 1)
    # 5. Several samples: the point forecast is the per-hour median.
    rep, pw = score(recs, [line(1, "with_context", v, sample=i) for i, v in enumerate(([590, 605, 615], [600, 610, 620], [900, 900, 900]))]
                    + [line(1, "no_context", [600, 610, 620], sample=i) for i in range(3)])
    # Per-hour medians of those three samples are 600, 610, 620, which is the truth. The mean would be far off because of the 900s.
    check("with 3 samples the per-hour median is used (MAE 0 here) and CRPS is reported", pw[("Toy", 1)]["with_context"]["mae"] == 0 and "rcrps" in rep
          and pw[("Toy", 1)]["no_context"]["crps"] == 0)
    # 6. CRPS by hand: samples 1 and 3 around a truth of 2 give 0 with the unbiased estimator; a far-off pair gives its distance.
    # Third case: truth 0, samples 1 and 2. Mean distance to the truth is 1.5, half the distance between the samples is 0.5, so CRPS is 1.0.
    check("CRPS hand cases", abs(crps(np.array([2.0]), np.array([[1.0], [3.0]]))[0]) < 1e-12 and abs(crps(np.array([0.0]), np.array([[10.0], [10.0]]))[0] - 10) < 1e-12
          and abs(crps(np.array([0.0]), np.array([[1.0], [2.0]]))[0] - 1.0) < 1e-12)
    # 7. The interval is reproducible and brackets the mean.
    a, b = np.array([10, 12, 9, 15, 11.0]), np.array([8, 11, 9, 10, 10.0])
    p1, p2 = paired_stats(a, b), paired_stats(a, b)
    check("the 95% interval is reproducible and contains the mean difference", p1 == p2 and p1["difference_ci95"][0] <= p1["mean_difference"] <= p1["difference_ci95"][1])
    # 7b. Text that a spreadsheet would run as a formula is neutralized; numbers and normal names are untouched.
    check("formula-like text is neutralized in the CSV", csv_safe("=HYPERLINK(1)") == "'=HYPERLINK(1)" and csv_safe("@x") == "'@x" and csv_safe("-1+1") == "'-1+1"
          and csv_safe("CarbonCastAECI") == "CarbonCastAECI" and csv_safe(-12.5) == -12.5 and csv_safe(20211001) == 20211001)
    # 8. Real data: baselines pushed through the scorer must reproduce the dataset build's own numbers.
    if VALIDATION.exists() and BUILD_REPORT.exists():
        real = load_records(VALIDATION); want = json.loads(BUILD_REPORT.read_text())["splits"]["validation"]
        lines = []
        for (name, seed), r in real.items():
            last = [float(r["past"][-1])] * len(r["truth"]); daily = [float(r["past"][-24 + (k % 24)]) for k in range(len(r["truth"]))]
            lines += [{"name": name, "seed": seed, "condition": "no_context", "sample": 0, "ok": True, "values": last},
                      {"name": name, "seed": seed, "condition": "with_context", "sample": 0, "ok": True, "values": daily}]
        rep, _ = score(real, lines); m = rep["mae"]; bl = rep["baselines_on_paired_windows"]
        print(f"     validation: last-value {m['mean_no_context']:.4f} (build says {want['mae_last_value']:.4f}), daily-repeat {m['mean_with_context']:.4f} (build says {want['mae_daily_repeat']:.4f})")
        check(f"real validation set ({rep['paired_windows']} windows): scorer reproduces the build's two baseline MAEs", rep["paired_windows"] == want["windows"]
              and abs(m["mean_no_context"] - want["mae_last_value"]) < 1e-9 and abs(m["mean_with_context"] - want["mae_daily_repeat"]) < 1e-9
              and abs(bl["mae_last_value"] - want["mae_last_value"]) < 1e-9 and abs(bl["mae_daily_repeat"] - want["mae_daily_repeat"]) < 1e-9)
    else:
        print("skip | real validation check (dataset files not found next to this folder)")
    print(f"\n{len(results)} tests, {results.count(False)} failed")
    raise SystemExit(1 if not all(results) else 0)

if __name__ == "__main__":
    main()
