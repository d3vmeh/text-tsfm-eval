"""Tests for prompt_builder.py. Run:  python3 test_prompt_builder.py"""
import json, re, tempfile
from pathlib import Path
from prompt_builder import make_prompt, build_entries, example_path, future_timestamps

def record(name="Toy", seed=7, answers=(608.14, 615.55)):
    past = {f"2019-01-21T{h:02d}:00:00.000": 600.0 + h for h in range(24)}
    fut = {f"2019-01-22T{h:02d}:00:00.000": v for h, v in zip((6, 7), answers)}
    return {"name": name, "seed": seed, "background": "Background text.", "scenario": "Scenario text.", "constraints": "",
            "past_time": json.dumps({"x": past}), "future_time": json.dumps({"x": fut})}

results = []
def check(name, cond):
    results.append(bool(cond)); print(f"{'pass' if cond else 'FAIL'} | {name}")

def main():
    r = record(); a, b = make_prompt(r, True), make_prompt(r, False)
    strip = lambda p: re.sub(r"<context>\n.*?\n</context>", "<context></context>", p, flags=re.S)
    check("the two prompts differ only inside the context block", a != b and strip(a) == strip(b))
    check("text appears only in the with-context prompt", "Scenario text." in a and "Scenario text." not in b and "Background text." not in b)
    check("the hidden answers never appear, and changing them changes nothing", "608.14" not in a and "615.55" not in a
          and make_prompt(record(answers=(-1.0, -2.0)), True) == a and make_prompt(record(answers=(-1.0, -2.0)), False) == b)
    check("both requested timestamps are listed", all(t in a for t in future_timestamps(r)) and future_timestamps(r) == ["2019-01-22 06:00:00", "2019-01-22 07:00:00"])
    check("two entries per record, one per condition", [e["condition"] for e in build_entries(r)] == ["with_context", "no_context"])
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d).resolve()
        normal = example_path(d, record(name="CarbonCastAECI", seed=20190122), "with_context")
        check("normal names give the expected file name", normal == folder / "CarbonCastAECI_20190122_with_context.txt")
        hostile = [example_path(d, record(name=n), "no_context") for n in ("../../etc/passwd", "/tmp/evil", "..", "a/b\\c", "")]
        check("names with slashes or dots cannot leave the examples folder", all(p.parent == folder for p in hostile))
        try:
            example_path(d, record(seed="1/../../x"), "no_context"); bad_seed_ok = False
        except ValueError:
            bad_seed_ok = True
        check("a seed that is not a number is refused", bad_seed_ok)
    print(f"\n{len(results)} tests, {results.count(False)} failed")
    raise SystemExit(1 if not all(results) else 0)

if __name__ == "__main__":
    main()
