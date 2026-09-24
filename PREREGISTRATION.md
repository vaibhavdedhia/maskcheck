# Pre-registration: held-out evaluation of the suspect rule

Written and committed **before** the held-out corpora were ever run. The git
timestamp on this file precedes the timestamp on the results commit; that
ordering is the only thing making the claims below worth anything.

## Why this document exists

The detector's aggregation rule was revised three times, each time after
looking at results from the same three conflict types. That is tuning on the
test set. Those three types (`enum`, `integer`, `maxLength`) are therefore
**contaminated** and are demoted to a development set. Nothing about the
detector's accuracy can honestly be claimed from them.

## The frozen rule

A field is **suspect** when it contributed at least 2 value tokens and:

    hit_rate >= 0.25   OR   override_rate >= 0.25

where

    forced token  := a token the unconstrained model gave < 0.5 probability
    hit_rate      := forced tokens / tokens attributed to the field
    override_rate := tokens where the mask beat the unconstrained argmax
                     / tokens with a known argmax

Both thresholds are chosen **a priori**, not fitted. `0.5` is "worse than
even odds" — the model would more likely than not have written something
else. `0.25` is "a quarter of this field's tokens". No numeric search was
run over either value.

Mean displacement was tried twice and failed twice in opposite directions:
clean samples diluted conflicting ones, then a happy closing token diluted an
unhappy value. A mean cannot express *rarely but catastrophically*, which is
the shape of a schema that contradicts part of its data. A fraction can, and
unlike `max` it is not moved by a single outlier token.

**This rule will not be changed after the held-out results are seen.**

## Held-out conflict types (never run against this rule)

Each has 8 cases: 4 where the schema contradicts the document, 4 controls.
Each exercises a mechanism absent from the development set.

| corpus     | constraint          | mechanism            |
|------------|---------------------|----------------------|
| `minlen`   | `sku minLength: 10` | forced **expansion** |
| `nested`   | `unit enum [kg,lb]` | substitution **one level down** |
| `required` | `approver required` | forced **fabrication** of an absent field |

Run across four models: Qwen2.5-1.5B, Llama-3.2-1B, Gemma-2-2B, Qwen2.5-3B.

## Predictions

**P1 — `minlen` will be DETECTED.** The model wants to close the string after
`A1`; the grammar forbids closing and forces eight more characters it had no
intention of writing. Those characters should carry low unconstrained
probability, so hit_rate should clear 0.25 on the conflicting cases.

**P2 — `nested` will be DETECTED.** Same substitution mechanism as the
development `enum`, one level deeper. If nesting alone breaks detection, the
path-attribution layer is at fault, not the rule.

**P3 — `required` will be MISSED.** Stated as a predicted failure. A model
asked to supply an approver that the document never mentions can invent a
plausible name *confidently* — and confident fabrication has low
displacement by construction. There is no fight to measure. Displacement
detects the grammar overriding the model, not the model inventing things.

**P4 — accuracy.** Constrained accuracy should fall below unconstrained for
`minlen` and `nested`. For `required` it should barely move, because the gold
records omit the field entirely; the harm appears as hallucinated fields, not
as wrong values. If P4 holds, "constrained decoding costs accuracy" will have
replicated on 8 of 12 new cells.

## What would falsify the approach

- `minlen` or `nested` missed on most models → the rule does not generalise
  beyond the conflicts it was built against, and the detector is not ready.
- Controls flagged → false positives, which would matter more than any miss.
- `required` detected → P3 wrong, and displacement measures more than I claim.

## Commitment

The held-out set is run **once**. Results are reported in full — including
misses, including false positives — with no changes to the rule, the
thresholds, or the corpora afterwards.
