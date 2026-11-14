"""Turn a model's reply into forecast numbers, or say why it can't be used.

The prompt asks for:
    <forecast>
    (2019-01-22 06:00:00, 610.2)
    ...one line per requested timestamp...
    </forecast>

parse_reply() follows the reading rules of the Context is Key (CiK) benchmark code
(cik_benchmark/baselines/direct_prompt.py), with five deliberate differences, each marked DIFF below.
parse_reply_cik() is the CiK rule on its own, kept for comparison and tests.

Usage:
    result = parse_reply(reply_text, expected_timestamps)
    if result.ok: result.values   # one float per expected timestamp, in order
    else:         result.reason   # short explanation, for counting failure types
"""
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
FORECAST_BLOCK = re.compile(r"<forecast>(.*?)</forecast>", re.DOTALL)


@dataclass
class ParseResult:
    ok: bool
    values: Optional[List[float]] = None
    reason: str = ""
    extra_timestamps: int = 0          # lines for timestamps that were not asked for (ignored)
    notes: List[str] = field(default_factory=list)


def parse_reply(text, expected_timestamps):
    """expected_timestamps: strings like "2019-01-22 06:00:00", in forecast order."""
    if not isinstance(text, str) or not text.strip():
        return ParseResult(False, reason="empty reply")
    notes = []
    # DIFF 1: drop any <think>...</think> reasoning first, so tags quoted while thinking are not read as the answer.
    cleaned = THINK_BLOCK.sub("", text)
    if cleaned != text:
        notes.append("removed think block")
    blocks = FORECAST_BLOCK.findall(cleaned)
    if not blocks:
        if "<forecast>" in cleaned:
            return ParseResult(False, reason="forecast block not closed (reply may be cut off)", notes=notes)
        return ParseResult(False, reason="no forecast block", notes=notes)
    body = blocks[0].strip()                     # same as CiK: the first block is the answer
    forecast = {}
    for raw in body.replace("(", "").replace(")", "").split("\n"):
        line = raw.strip()
        if not line:                             # DIFF 2: blank lines are skipped (CiK rejects the reply)
            continue
        parts = line.split(",")
        if len(parts) != 2:                      # DIFF 3: exactly one comma per line (CiK reads "16,754.8" as 16)
            return ParseResult(False, reason=f"line is not (timestamp, value): {line[:60]!r}", notes=notes)
        stamp = parts[0].replace("'", "").replace('"', "").strip()   # DIFF 4: spaces around the timestamp are allowed
        try:
            value = float(parts[1])
        except ValueError:
            return ParseResult(False, reason=f"value is not a number: {line[:60]!r}", notes=notes)
        if not math.isfinite(value):             # DIFF 5: nan and inf are rejected (CiK accepts them)
            return ParseResult(False, reason=f"value is not finite: {line[:60]!r}", notes=notes)
        forecast[stamp] = value
    missing = [t for t in expected_timestamps if t not in forecast]
    if missing:
        return ParseResult(False, reason=f"missing {len(missing)} of {len(expected_timestamps)} timestamps (first: {missing[0]})", notes=notes)
    return ParseResult(True, values=[forecast[t] for t in expected_timestamps],
                       extra_timestamps=len(forecast) - len(expected_timestamps), notes=notes)


def parse_reply_cik(text, expected_timestamps):
    """The CiK benchmark's own reading rule, unchanged. Returns the values, or raises on any problem."""
    content = [m.strip() for m in FORECAST_BLOCK.findall(text)]
    forecast = content[0]
    forecast = forecast.replace("(", "").replace(")", "")
    forecast = forecast.split("\n")
    forecast = {x.split(",")[0].replace("'", "").replace('"', ""): float(x.split(",")[1]) for x in forecast}
    return [forecast[t] for t in expected_timestamps]


def forecast_regex(expected_timestamps):
    """A pattern that only a complete, well-formed reply matches. Use it to force the format
    (constrained decoding) on a local model server. Same pattern as the CiK code builds."""
    lines = "".join(r"\(\s*{}\s*,\s*[-+]?\d+(\.\d+)?\)\n".format(re.escape(ts)) for ts in expected_timestamps)
    return r"<forecast>\n{}<\/forecast>".format(lines)
