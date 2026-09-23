"""Field-level accuracy against gold labels.

The headline metric of this benchmark is NOT schema validity -- every
constrained engine scores 100% there by construction, which is why the
number is uninteresting. What nobody has published is whether the mask
costs you correct *values*. That requires ground truth, so scoring here
is against independently-known gold JSON, never an LLM judge.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

_WS_RE = re.compile(r"\s+")

# Leaf sentinel: empty containers are values, not paths to recurse into.
EMPTY_DICT = "<empty-dict>"
EMPTY_LIST = "<empty-list>"


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Map an object to {leaf_path: value}.

    Dicts extend the path with `.key`; lists with `[i]`. Order-sensitivity
    for lists is deliberate -- see `score_record`'s `list_order` note.
    """
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            out[prefix or "$"] = EMPTY_DICT
            return out
        for k, v in obj.items():
            child = "{}.{}".format(prefix, k) if prefix else str(k)
            out.update(flatten(v, child))
        return out
    if isinstance(obj, list):
        if not obj:
            out[prefix or "$"] = EMPTY_LIST
            return out
        for i, v in enumerate(obj):
            child = "{}[{}]".format(prefix, i)
            out.update(flatten(v, child))
        return out
    out[prefix or "$"] = obj
    return out


def norm_string(s: str) -> str:
    return _WS_RE.sub(" ", s.strip()).casefold()


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


_BOOL_WORDS = {"true": True, "false": False}


def _as_bool(v):
    """Coerce only genuine booleans and the words 'true'/'false'.

    Deliberately returns None for 1/0 so that integer 1 never matches
    boolean True -- a model emitting 1 for a boolean field is a real type
    error, not a formatting difference.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return _BOOL_WORDS.get(v.strip().casefold())
    return None


def compare(gold: Any, pred: Any, coercion: bool = True) -> Tuple[bool, bool]:
    """Return (match, used_coercion).

    `used_coercion` is tracked and reported separately so the headline
    accuracy can be recomputed without it. Models routinely emit "42" for
    42 or "true" for true; counting those as wrong would overstate the
    penalty, and counting them silently would hide a real type difference.
    """
    # Same-type fast paths. bool must precede numeric: bool is an int subclass.
    if isinstance(gold, bool) and isinstance(pred, bool):
        return gold == pred, False
    if gold is None or pred is None:
        return gold is None and pred is None, False
    if _is_num(gold) and _is_num(pred):
        return math.isclose(gold, pred, rel_tol=1e-9, abs_tol=1e-12), False
    if isinstance(gold, str) and isinstance(pred, str):
        return norm_string(gold) == norm_string(pred), False

    if not coercion:
        return False, False

    # Cross-type: one side stringified the other.
    if isinstance(gold, bool) or isinstance(pred, bool):
        g, p = _as_bool(gold), _as_bool(pred)
        if g is None or p is None:
            return False, False
        return g == p, True
    if _is_num(gold) or _is_num(pred):
        try:
            g = float(gold.strip()) if isinstance(gold, str) else float(gold)
            p = float(pred.strip()) if isinstance(pred, str) else float(pred)
        except (ValueError, TypeError):
            return False, False
        return math.isclose(g, p, rel_tol=1e-9, abs_tol=1e-12), True
    return gold == pred, False


class RecordScore(object):
    __slots__ = ("total", "correct", "missing", "wrong", "hallucinated", "coerced", "parsed")

    def __init__(self):
        self.total = 0          # gold leaves
        self.correct = 0
        self.missing = 0        # gold leaf absent from prediction
        self.wrong = 0          # present but different value
        self.hallucinated = 0   # predicted leaf with no gold counterpart
        self.coerced = 0        # correct only via numeric coercion
        self.parsed = False

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0

    @property
    def accuracy_strict_types(self) -> float:
        """Accuracy with coercion-dependent matches counted as wrong."""
        if not self.total:
            return 0.0
        return (self.correct - self.coerced) / self.total

    def as_dict(self) -> Dict[str, Any]:
        return {
            "parsed": self.parsed,
            "total": self.total,
            "correct": self.correct,
            "missing": self.missing,
            "wrong": self.wrong,
            "hallucinated": self.hallucinated,
            "coerced": self.coerced,
            "accuracy": round(self.accuracy, 6),
            "accuracy_strict_types": round(self.accuracy_strict_types, 6),
        }


def score_record(gold: Any, pred: Optional[Any], coercion: bool = True) -> RecordScore:
    """Score one prediction. `pred=None` means unparseable -> accuracy 0.

    Note on lists: paths are index-sensitive, so a correct set of items in a
    different order scores as wrong. For the extraction corpus this is
    intended -- gold order is defined by source-document order, which is the
    only order a correct extractor can produce. Set-valued fields must be
    declared as such in the corpus, not papered over here.
    """
    s = RecordScore()
    gold_flat = flatten(gold)
    s.total = len(gold_flat)
    if pred is None:
        s.missing = s.total
        return s
    s.parsed = True
    pred_flat = flatten(pred)
    for path, gv in gold_flat.items():
        if path not in pred_flat:
            s.missing += 1
            continue
        ok, coerced = compare(gv, pred_flat[path], coercion)
        if ok:
            s.correct += 1
            if coerced:
                s.coerced += 1
        else:
            s.wrong += 1
    s.hallucinated = len(set(pred_flat) - set(gold_flat))
    return s


def aggregate(scores: List[RecordScore]) -> Dict[str, Any]:
    """Micro-average over leaves, plus the rates the README table needs."""
    n = len(scores)
    if not n:
        return {}
    tot = sum(s.total for s in scores)
    cor = sum(s.correct for s in scores)
    coe = sum(s.coerced for s in scores)
    return {
        "records": n,
        "parse_rate": round(sum(1 for s in scores if s.parsed) / n, 6),
        "leaves": tot,
        "field_accuracy": round(cor / tot, 6) if tot else 0.0,
        "field_accuracy_strict_types": round((cor - coe) / tot, 6) if tot else 0.0,
        "exact_record_rate": round(sum(1 for s in scores if s.total and s.correct == s.total) / n, 6),
        "missing_rate": round(sum(s.missing for s in scores) / tot, 6) if tot else 0.0,
        "wrong_rate": round(sum(s.wrong for s in scores) / tot, 6) if tot else 0.0,
        "hallucinated_per_record": round(sum(s.hallucinated for s in scores) / n, 6),
        "coercion_share": round(coe / cor, 6) if cor else 0.0,
    }
