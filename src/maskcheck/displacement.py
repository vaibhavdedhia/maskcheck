"""Displacement: how hard the mask fought the model, per schema field.

No ground truth required. The reference distribution is the model's *own*
unconstrained preference, obtained by teacher-forcing the constrained output
back through the model and reading the logprob at each position:

    displacement(token) = 1 - P_unconstrained(token | prefix)

High displacement means the grammar pushed the model somewhere it did not
want to go. Attributed to a JSON path, that localises the damage:

    invoice.total   override 8/10   mean displacement 0.72   <- suspect

This works from outside the engine -- no mask internals, no forks -- so one
code path covers llama.cpp, MLX and transformers.

A caveat kept in front of the user rather than buried: displacement is a
*suspicion* signal, not an error measurement. A field can be displaced and
still correct (the model's preferred phrasing was merely unavailable), or
undisplaced and wrong. It tells you where to look. `--labels` upgrades
suspicion to measured accuracy via `scoring.py`.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .pathtrack import VALUE_REGIONS, path_at_offsets

# Below this, a field is producing tokens the free model was fine with.
DEFAULT_SUSPECT_DISPLACEMENT = 0.5
# A field must contribute at least this many value tokens to be judged.
DEFAULT_MIN_STEPS = 2


class Step(object):
    """One generated token, with the unconstrained model's opinion of it.

    `logprob` is the *unconstrained* logprob of the token that was actually
    emitted. `top_logprob`/`top_token` describe what the unconstrained model
    would have picked instead, when the engine reports alternatives.
    """

    __slots__ = ("text", "offset", "logprob", "top_token", "top_logprob")

    def __init__(self, text, offset, logprob, top_token=None, top_logprob=None):
        # type: (str, int, float, Optional[str], Optional[float]) -> None
        if logprob > 0.0:
            raise ValueError("logprob must be <= 0, got %r" % (logprob,))
        self.text = text
        self.offset = offset
        self.logprob = logprob
        self.top_token = top_token
        self.top_logprob = top_logprob

    @property
    def probability(self) -> float:
        return math.exp(self.logprob)

    @property
    def displacement(self) -> float:
        """Probability mass the unconstrained model put elsewhere."""
        return 1.0 - self.probability

    @property
    def overridden(self) -> Optional[bool]:
        """Did the mask beat the unconstrained argmax? None if unknown."""
        if self.top_token is None:
            return None
        return self.top_token != self.text


class FieldReport(object):
    __slots__ = ("path", "steps", "mean_displacement", "max_displacement",
                 "override_rate", "known_overrides", "suspect")

    def __init__(self, path):
        # type: (str) -> None
        self.path = path
        self.steps = 0
        self.mean_displacement = 0.0
        self.max_displacement = 0.0
        self.override_rate = 0.0
        self.known_overrides = 0
        self.suspect = False

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "steps": self.steps,
            "mean_displacement": round(self.mean_displacement, 6),
            "max_displacement": round(self.max_displacement, 6),
            "override_rate": round(self.override_rate, 6),
            "suspect": self.suspect,
        }


def bucket(text, steps):
    # type: (str, Sequence[Step], ) -> Dict[str, List[Step]]
    """Group value-region steps by JSON path for one generation."""
    if not steps:
        return {}
    starts = [s.offset for s in steps]
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1]:
            raise ValueError("steps must be ordered by offset")
    probes = [min(o + 1, len(text)) for o in starts]
    out = {}  # type: Dict[str, List[Step]]
    for step, (path, region) in zip(steps, path_at_offsets(text, probes)):
        if region in VALUE_REGIONS and path:
            out.setdefault(path, []).append(step)
    return out


def summarise(buckets,
              suspect_threshold=DEFAULT_SUSPECT_DISPLACEMENT,
              min_steps=DEFAULT_MIN_STEPS):
    # type: (Dict[str, List[Step]], float, int) -> List[FieldReport]
    """Turn pooled buckets into reports, worst-first."""
    reports = []  # type: List[FieldReport]
    for path, group in buckets.items():
        r = FieldReport(path)
        r.steps = len(group)
        disps = [s.displacement for s in group]
        r.mean_displacement = sum(disps) / len(disps)
        r.max_displacement = max(disps)
        known = [s for s in group if s.overridden is not None]
        r.known_overrides = sum(1 for s in known if s.overridden)
        r.override_rate = (r.known_overrides / len(known)) if known else 0.0
        r.suspect = (r.steps >= min_steps
                     and r.mean_displacement >= suspect_threshold)
        reports.append(r)
    reports.sort(key=lambda r: (-r.mean_displacement, r.path))
    return reports


def analyze_many(pairs,
                 suspect_threshold=DEFAULT_SUSPECT_DISPLACEMENT,
                 min_steps=DEFAULT_MIN_STEPS):
    # type: (Sequence[Tuple[str, Sequence[Step]]], float, int) -> List[FieldReport]
    """Pool many (text, steps) generations into one per-field report."""
    pooled = {}  # type: Dict[str, List[Step]]
    for text, steps in pairs:
        for path, group in bucket(text, steps).items():
            pooled.setdefault(path, []).extend(group)
    return summarise(pooled, suspect_threshold, min_steps)


def analyze(text,
            steps,
            suspect_threshold=DEFAULT_SUSPECT_DISPLACEMENT,
            min_steps=DEFAULT_MIN_STEPS):
    # type: (str, Sequence[Step], float, int) -> List[FieldReport]
    """Attribute per-token displacement to JSON paths.

    Only value-region tokens count. Structural tokens (braces, commas,
    colons) and object keys are excluded: the grammar is supposed to pin
    those, so their displacement measures the mask succeeding, and including
    them would inflate every field toward the same meaningless number.

    Steps must be ordered by offset. Returns reports sorted worst-first.
    """
    return summarise(bucket(text, steps), suspect_threshold, min_steps)


def worst_displacement(reports: Iterable[FieldReport]) -> float:
    """Headline number for `--fail-on-displacement`, over judged fields only."""
    vals = [r.mean_displacement for r in reports if r.steps >= DEFAULT_MIN_STEPS]
    return max(vals) if vals else 0.0
