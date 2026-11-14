"""Forecast the windows with Chronos-2, a model built for forecasting, as a numbers-only baseline.

Chronos-2 (Amazon, https://huggingface.co/amazon/chronos-2) reads numbers, not text. Here it gets the
same past values the LLM prompts show, and nothing else: no text, no weather and no other window.
It answers the question "how well does a real forecasting model do on these windows?".

What it is careful about
  - It never reads the hidden answers. Only the past values and the timestamps to forecast are used.
  - The past values are rounded to 6 significant digits, exactly as prompt_builder.py prints them.
  - Each window is forecast on its own (one per call, cross-learning off). The windows overlap in
    time, so letting the model look across windows would leak one window's future into another.
  - The model snapshot is pinned and recorded, and the forecast has no randomness.

Outputs, for --run NAME
  results/NAME/forecasts.jsonl   one line per window in the format scorer.py reads. The condition is
                                 "no_context", because the model sees no text. `values` is the median.
  results/NAME/quantiles.jsonl   the 10% ... 90% quantile forecasts, for later use.
  results/NAME/settings.json     model, snapshot, package versions and the choices above.

To see the scores next to other runs, use compare_forecasts.py.

Needs PyTorch and the chronos-forecasting package, which live in .venv:
  .venv/bin/python chronos_baseline.py ../text-tsfm-benchmarks/outputs/cik_format/AECI_validation.jsonl --run validation_chronos2
"""
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path

from prompt_builder import _series, future_timestamps

MODEL = "amazon/chronos-2"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def history(record, max_digits=6):
    """The past values as the LLM prompt shows them (same rounding as prompt_builder.make_prompt)."""
    _stamps, values = _series(record["past_time"])
    return [float(f"{y:.{max_digits}g}") if y < 10**max_digits else float(f"{y:.0f}") for y in values]


def forecast_records(records, predict):
    """predict(past_values, horizon) -> one list of quantile values per future hour.
    Yields (forecast line, quantile line) per record. The hidden answers are never read."""
    middle = QUANTILES.index(0.5)
    for rec in records:
        horizon = len(future_timestamps(rec))
        rows = predict(history(rec), horizon)
        if len(rows) != horizon or any(len(r) != len(QUANTILES) for r in rows):
            raise ValueError(f"window {rec['name']} {rec['seed']}: the model returned the wrong shape")
        key = {"name": rec["name"], "seed": rec["seed"], "condition": "no_context", "sample": 0}
        yield ({**key, "ok": True, "values": [float(r[middle]) for r in rows]},
               {**key, "quantile_levels": QUANTILES, "quantiles": [[float(v) for v in r] for r in rows]})


def load_chronos(model=MODEL, revision=None):
    """Returns (predict function, facts about the model). Imports are here so tests don't need PyTorch."""
    import chronos, torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, revision=revision)
    pipeline = Chronos2Pipeline.from_pretrained(path, device_map="cpu")

    def predict(values, horizon):
        quantiles, _mean = pipeline.predict_quantiles([torch.tensor(values, dtype=torch.float32)], prediction_length=horizon,
                                                      quantile_levels=QUANTILES, cross_learning=False)
        return quantiles[0][0].tolist()                       # (hours, quantile levels)

    info = {"model": model, "snapshot": path.split("/snapshots/")[-1].strip("/"), "device": "cpu",
            "parameters": sum(p.numel() for p in pipeline.model.parameters()),
            "model_context_length": getattr(pipeline, "model_context_length", None),
            "chronos_forecasting_version": chronos.__version__, "torch_version": torch.__version__}
    return predict, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records"); ap.add_argument("--run", required=True)
    ap.add_argument("--revision", help="model snapshot to use (default: the newest, which is then recorded)")
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)
    with open(args.records) as f:
        records = [json.loads(line) for line in f if line.strip()]
    predict, info = load_chronos(revision=args.revision)
    out = Path(args.results_dir) / args.run
    out.mkdir(parents=True, exist_ok=True)
    started = time.time(); n = 0
    with open(out / "forecasts.jsonl", "w") as ff, open(out / "quantiles.jsonl", "w") as qf:
        for line, quantile_line in forecast_records(records, predict):
            ff.write(json.dumps(line) + "\n"); qf.write(json.dumps(quantile_line) + "\n"); n += 1
    settings = {**info, "records_file": args.records, "records_sha256": hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
                "windows": n, "input": "past values only, rounded to 6 significant digits as in the LLM prompt",
                "point_forecast": "the 0.5 quantile (median)", "quantile_levels": QUANTILES,
                "one_window_per_call": True, "cross_learning": False, "covariates": "none",
                "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds": round(time.time() - started, 2)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{n} windows forecast with {info['model']} (snapshot {info['snapshot'][:12]}) in {time.time() - started:.1f} s")
    print(f"Forecasts: {out / 'forecasts.jsonl'}\nSettings:  {out / 'settings.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
