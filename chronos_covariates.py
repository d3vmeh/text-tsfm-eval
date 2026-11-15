"""Forecast the windows with Chronos-2, giving it the weather forecast as numeric covariates.

chronos_baseline.py gives Chronos-2 the past values only. This gives it, in addition, hourly weather
series built by scripts/aeci_covariates.py in the dataset repo: values for the past hours and for the
hours being forecast ("known future covariates"). It is the same weather information the window text
describes, delivered as numbers, so it shows whether that information helps a forecasting model.

What it is careful about
  - It never reads the hidden answers. Only past values, the covariates and the horizon length are used.
  - The past values get the same rounding as in chronos_baseline.py and the LLM prompts.
  - Each window is forecast on its own (one per call, cross-learning off), as in the baseline.
  - The model snapshot is pinned by default to the one the baseline run used, and is recorded.

Outputs, for --run NAME: results/NAME/forecasts.jsonl, quantiles.jsonl and settings.json, as chronos_baseline.py.
The `condition` written to each line is --condition (default "with_covariates").

Needs PyTorch and chronos-forecasting, which live in .venv:
  .venv/bin/python chronos_covariates.py RECORDS.jsonl COVARIATES.jsonl --run NAME [--variables wind_speed_m_s,solar_w_m2]
"""
import argparse, hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path

from chronos_baseline import MODEL, QUANTILES, history
from prompt_builder import future_timestamps

PINNED_SNAPSHOT = "29ec3766d36d6f73f0696f85560a422f50e8498c"   # the snapshot the validation_chronos2 baseline used


def forecast_records(records, covariates, predict, variables, condition):
    """predict(past_values, past_covariates, future_covariates, horizon) -> one list of quantile values per future hour.
    Yields (forecast line, quantile line) per record. The hidden answers are never read."""
    middle = QUANTILES.index(0.5)
    by_key = {(c["name"], c["seed"]): c for c in covariates}
    for rec in records:
        key = (rec["name"], rec["seed"])
        if key not in by_key:
            raise KeyError(f"no covariates for window {key}")
        cov = by_key[key]; past = history(rec); horizon = len(future_timestamps(rec))
        pc = {v: cov["past_covariates"][v] for v in variables}; fc = {v: cov["future_covariates"][v] for v in variables}
        if any(len(x) != len(past) for x in pc.values()) or any(len(x) != horizon for x in fc.values()):
            raise ValueError(f"window {key}: covariate lengths do not match the window")
        rows = predict(past, pc, fc, horizon)
        if len(rows) != horizon or any(len(r) != len(QUANTILES) for r in rows):
            raise ValueError(f"window {key}: the model returned the wrong shape")
        k = {"name": rec["name"], "seed": rec["seed"], "condition": condition, "sample": 0}
        yield ({**k, "ok": True, "values": [float(r[middle]) for r in rows]},
               {**k, "quantile_levels": QUANTILES, "quantiles": [[float(v) for v in r] for r in rows]})


def load_chronos(model=MODEL, revision=PINNED_SNAPSHOT):
    import chronos, numpy as np, torch
    from chronos import Chronos2Pipeline
    from huggingface_hub import snapshot_download
    path = snapshot_download(model, revision=revision)
    pipeline = Chronos2Pipeline.from_pretrained(path, device_map="cpu")

    def predict(values, past_cov, future_cov, horizon):
        item = {"target": np.asarray(values, dtype=np.float32)}
        if past_cov:
            item["past_covariates"] = {k: np.asarray(v, dtype=np.float32) for k, v in past_cov.items()}
            item["future_covariates"] = {k: np.asarray(v, dtype=np.float32) for k, v in future_cov.items()}
        quantiles, _mean = pipeline.predict_quantiles([item], prediction_length=horizon, quantile_levels=QUANTILES, cross_learning=False)
        return quantiles[0][0].tolist()                       # (hours, quantile levels)

    info = {"model": model, "snapshot": path.split("/snapshots/")[-1].strip("/"), "device": "cpu",
            "parameters": sum(p.numel() for p in pipeline.model.parameters()),
            "chronos_forecasting_version": chronos.__version__, "torch_version": torch.__version__}
    return predict, info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records"); ap.add_argument("covariates"); ap.add_argument("--run", required=True)
    ap.add_argument("--variables", help="comma-separated covariate names to use (default: all in the file)")
    ap.add_argument("--condition", default="with_covariates"); ap.add_argument("--revision", default=PINNED_SNAPSHOT)
    ap.add_argument("--results-dir", default="results")
    args = ap.parse_args(argv)
    with open(args.records) as f: records = [json.loads(line) for line in f if line.strip()]
    with open(args.covariates) as f: covariates = [json.loads(line) for line in f if line.strip()]
    variables = args.variables.split(",") if args.variables else list(covariates[0]["variables"])
    predict, info = load_chronos(revision=args.revision)
    out = Path(args.results_dir) / args.run; out.mkdir(parents=True, exist_ok=True)
    started = time.time(); n = 0
    with open(out / "forecasts.jsonl", "w") as ff, open(out / "quantiles.jsonl", "w") as qf:
        for line, qline in forecast_records(records, covariates, predict, variables, args.condition):
            ff.write(json.dumps(line) + "\n"); qf.write(json.dumps(qline) + "\n"); n += 1
    settings = {**info, "records_file": args.records, "records_sha256": hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
                "covariates_file": args.covariates, "covariates_sha256": hashlib.sha256(Path(args.covariates).read_bytes()).hexdigest(),
                "covariate_alignment": covariates[0].get("alignment"), "covariates": variables, "covariates_given_for": "past and future hours",
                "windows": n, "input": "past values rounded to 6 significant digits as in the LLM prompt, plus the covariates",
                "point_forecast": "the 0.5 quantile (median)", "quantile_levels": QUANTILES, "one_window_per_call": True, "cross_learning": False,
                "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seconds": round(time.time() - started, 2)}
    (out / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"{n} windows forecast with {info['model']} (snapshot {info['snapshot'][:12]}) and covariates {variables} in {time.time() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
