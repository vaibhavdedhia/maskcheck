"""Rendering. No dependencies -- plain text tables and JSON."""

import json
from typing import Any, Dict, List, Optional, Sequence

from .displacement import FieldReport, worst_displacement

_HEADERS = ("FIELD", "STEPS", "FORCED", "HIT RATE", "MEAN DISP", "OVERRIDE", "")


def _rows(reports: Sequence[FieldReport]) -> List[List[str]]:
    rows = []
    for r in reports:
        rows.append([
            r.path,
            str(r.steps),
            str(r.forced_tokens),
            "%d%%" % round(r.hit_rate * 100),
            "%.2f" % r.mean_displacement,
            "%d%%" % round(r.override_rate * 100),
            "SUSPECT" if r.suspect else "",
        ])
    return rows


def render_table(reports: Sequence[FieldReport]) -> str:
    if not reports:
        return "No value tokens attributed -- nothing to report.\n"
    rows = _rows(reports)
    widths = [len(h) for h in _HEADERS]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells):
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.ljust(widths[i]) if i == 0 else cell.rjust(widths[i]))
        return "  ".join(out).rstrip()

    # The trailing flag column has a blank header, so it gets no rule.
    rule = ["-" * w for w in widths[:-1]] + [""]
    parts = [line(list(_HEADERS)), "  ".join(rule).rstrip()]
    parts.extend(line(r) for r in rows)
    return "\n".join(parts) + "\n"


def render_summary(reports, threshold=None, scores=None):
    # type: (Sequence[FieldReport], Optional[float], Optional[Dict[str, Any]]) -> str
    worst = worst_displacement(reports)
    suspects = [r for r in reports if r.suspect]
    lines = ["", "%d field(s) analysed, %d suspect. Worst mean displacement: %.2f"
             % (len(reports), len(suspects), worst)]
    if suspects:
        lines.append("Suspect: " + ", ".join(
            "%s (%d/%d tokens forced, override %d%%)"
            % (r.path, r.forced_tokens, r.steps, round(r.override_rate * 100))
            for r in suspects))
        lines.append("Displacement flags where the grammar fought the model; "
                     "it is not proof of a wrong value. Re-run with --labels "
                     "to measure accuracy.")
    if scores:
        lines.append("Field accuracy vs labels: %.3f (parse rate %.3f)"
                     % (scores.get("field_accuracy", 0.0), scores.get("parse_rate", 0.0)))
    if threshold is not None:
        verdict = "FAIL" if worst > threshold else "PASS"
        lines.append("%s: worst displacement %.2f vs threshold %.2f"
                     % (verdict, worst, threshold))
    return "\n".join(lines) + "\n"


def render_json(reports, threshold=None, scores=None, extra=None):
    # type: (Sequence[FieldReport], Optional[float], Optional[Dict[str, Any]], Optional[Dict[str, Any]]) -> str
    payload = {
        "fields": [r.as_dict() for r in reports],
        "worst_mean_displacement": round(worst_displacement(reports), 6),
        "suspect": [r.path for r in reports if r.suspect],
    }
    if threshold is not None:
        payload["threshold"] = threshold
        payload["passed"] = worst_displacement(reports) <= threshold
    if scores:
        payload["labels"] = scores
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"
