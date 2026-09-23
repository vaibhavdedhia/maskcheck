"""Best-effort recovery of JSON from unconstrained LLM output.

This module defines the `repaired` scoring policy. It exists to keep the
constrained-vs-unconstrained comparison honest: production code does not
throw away output that merely has a markdown fence around it, so neither
do we. Every stage is named in the result so the benchmark can report how
much of the baseline's score depended on repair rather than on the model.
"""

import json
import re
from typing import Any, List, Optional, Tuple

# Ordered most-conservative first. A record is attributed to the first
# stage that yields parseable JSON.
STAGES = ("direct", "fence", "balanced", "trailing_comma", "failed")

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)\s*```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _balanced_spans(text: str) -> List[str]:
    """Yield top-level {...} / [...] substrings, respecting strings+escapes.

    A naive regex or a rfind('}') both mis-handle braces that appear inside
    string values, which is exactly what happens when a model emits prose
    containing JSON. This walks the text once with a depth counter.
    """
    spans: List[str] = []
    depth = 0
    start = -1
    in_str = False
    escaped = False
    opener = ""
    for i, ch in enumerate(text):
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch in "{[":
            if depth == 0:
                start = i
                opener = ch
            depth += 1
        elif ch in "}]":
            if depth == 0:
                continue  # stray closer, ignore
            # Guard against {..] mismatches producing garbage spans.
            if depth == 1 and opener and ((opener == "{") != (ch == "}")):
                depth = 0
                start = -1
                continue
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : i + 1])
                start = -1
    return spans


def repair(text: str) -> Tuple[Optional[Any], str]:
    """Return (parsed, stage). `parsed` is None iff stage == 'failed'."""
    if text is None:
        return None, "failed"

    # 1. The model did the right thing.
    stripped = text.strip()
    try:
        return json.loads(stripped), "direct"
    except (ValueError, TypeError):
        pass

    # 2. Wrapped in a markdown code fence.
    for m in _FENCE_RE.finditer(text):
        try:
            return json.loads(m.group(1)), "fence"
        except (ValueError, TypeError):
            continue

    # 3. Embedded in prose ("Sure, here is your JSON: {...} Let me know...").
    spans = _balanced_spans(text)
    for span in spans:
        try:
            return json.loads(span), "balanced"
        except (ValueError, TypeError):
            continue

    # 4. Structurally right but with trailing commas.
    for candidate in ([stripped] + spans):
        fixed = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        if fixed == candidate:
            continue
        try:
            return json.loads(fixed), "trailing_comma"
        except (ValueError, TypeError):
            continue

    return None, "failed"
