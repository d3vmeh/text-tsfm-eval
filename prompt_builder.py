"""Build forecasting prompts from Context is Key (CiK) style records.

For each record it makes two prompts: one with the text (background, constraints, scenario) and one
without. Both show the same past values and ask for the same future timestamps. Neither contains
the hidden answers.

The prompt wording is the Direct Prompt template from the CiK benchmark code
(ServiceNow/context-is-key-forecasting, cik_benchmark/baselines/direct_prompt.py, Apache-2.0),
reproduced so results are comparable with the paper.

Usage:
  python3 prompt_builder.py RECORDS.jsonl --out prompts/NAME_prompts.jsonl [--examples examples/]
"""
import argparse, json, re
from pathlib import Path

import numpy as np

SYSTEM_MESSAGE = "You are a useful forecasting assistant."
CONDITIONS = {"with_context": True, "no_context": False}


def _series(field):
    """A CiK series field is JSON text: {"column": {"2019-01-01T06:00:00.000": 585.73, ...}}."""
    column = list(json.loads(field).values())[-1]
    stamps = [t[:19].replace("T", " ") for t in column]          # -> "2019-01-01 06:00:00"
    return stamps, list(column.values())


def future_timestamps(record):
    """The timestamps to forecast. The values stored beside them are the hidden answers and are not returned."""
    stamps, _hidden = _series(record["future_time"])
    return stamps


def make_prompt(record, use_context, max_digits=6):
    hist_time, hist_value = _series(record["past_time"])
    pred_time = np.array(future_timestamps(record), dtype=object)
    history = "\n".join(
        f"({x}, {y:.{max_digits}g})" if y < 10**max_digits else f"({x}, {y:.0f})"
        for x, y in zip(hist_time, hist_value)
    )
    context = ""
    if use_context:
        if record["background"]:
            context += f"Background: {record['background']}\n"
        if record["constraints"]:
            context += f"Constraints: {record['constraints']}\n"
        if record["scenario"]:
            context += f"Scenario: {record['scenario']}\n"

    prompt = f"""
I have a time series forecasting task for you.

Here is some context about the task. Make sure to factor in any background knowledge,
satisfy any constraints, and respect any scenarios.
<context>
{context}
</context>

Here is a historical time series in (timestamp, value) format:
<history>
{history}
</history>

Now please predict the value at the following timestamps: {pred_time}.

Return the forecast in (timestamp, value) format in between <forecast> and </forecast> tags.
Do not include any other information (e.g., comments) in the forecast.

Example:
<history>
(t1, v1)
(t2, v2)
(t3, v3)
</history>
<forecast>
(t4, v4)
(t5, v5)
</forecast>

"""
    return prompt


def make_messages(record, use_context):
    return [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": make_prompt(record, use_context)}]


def build_entries(record):
    """Two prompt entries per record. `future_timestamps` is what the reply parser should expect back."""
    for condition, use_context in CONDITIONS.items():
        yield {"name": record["name"], "seed": record["seed"], "condition": condition,
               "messages": make_messages(record, use_context), "future_timestamps": future_timestamps(record)}


def example_path(examples_dir, record, condition):
    """Where an example prompt is saved. Record fields are data, so they are cleaned before being used in
    a file name, and the result must stay inside the examples folder."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(record["name"])).lstrip(".") or "record"
    safe_condition = re.sub(r"[^A-Za-z0-9_-]", "_", str(condition))
    folder = Path(examples_dir).resolve()
    dest = (folder / f"{safe_name}_{int(record['seed'])}_{safe_condition}.txt").resolve()
    if dest.parent != folder:
        raise ValueError(f"refusing to write outside {folder}")
    return dest


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("records"); ap.add_argument("--out", required=True); ap.add_argument("--examples")
    args = ap.parse_args()
    with open(args.records) as f:
        records = [json.loads(line) for line in f if line.strip()]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.out, "w") as f:
        for i, rec in enumerate(records):
            for entry in build_entries(rec):
                f.write(json.dumps(entry) + "\n"); n += 1
                if args.examples and i == 0:
                    Path(args.examples).mkdir(parents=True, exist_ok=True)
                    example_path(args.examples, rec, entry["condition"]).write_text(entry["messages"][1]["content"])
    print(f"{len(records)} records -> {n} prompts in {args.out}")


if __name__ == "__main__":
    main()
