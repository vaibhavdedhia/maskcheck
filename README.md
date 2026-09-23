# maskcheck

**Constrained decoding gives you 100% schema validity. It does not give you correct values.**

Grammar-constrained generation — llama.cpp GBNF, XGrammar, Outlines, MLX's guided
generation — guarantees your local model emits parseable, schema-conformant JSON.
That guarantee is real, and it solved the problem everyone complains about.

It also quietly moved the problem. Validity became free, so the ecosystem stopped
measuring it — and stopped measuring anything else. Every engine reports schema
conformance and tokens/sec. None report whether the *values* survived.

```
{"invoice_id": "INV-2291", "vendor": "Acme Corp", "total": 48.00}
```

Parses. Validates. `total` is wrong — the model wanted `1240.50`, and the mask
took the tokens away.

`maskcheck` finds those fields.

## Quickstart

No model required for the demo:

```bash
git clone https://github.com/vaibhavdedhia/maskcheck
cd maskcheck && pip install -e .
maskcheck demo
```

```
FIELD       STEPS  MEAN DISP  MAX DISP  OVERRIDE
----------  -----  ---------  --------  --------
total           4       0.79      0.88       75%  SUSPECT
vendor          2       0.10      0.12        0%
invoice_id      3       0.05      0.07        0%

3 field(s) analysed, 1 suspect. Worst mean displacement: 0.79
```

## Measured: a schema that contradicts the data

Qwen2.5-1.5B-Instruct (Q4_K_M) via llama.cpp 0.4.1, 8 invoice extractions.
The schema restricts `currency` to `USD/EUR/GBP/INR`. Four of the eight
documents are denominated in **CHF, JPY, CAD and AUD** — so the grammar
*cannot* emit the right answer. The other four are controls.

|                          | constrained | unconstrained |
|--------------------------|-------------|---------------|
| parse rate               | **1.000**   | 1.000         |
| field accuracy vs gold   | **0.917**   | **1.000**     |
| worst mean displacement  | 0.44 (`currency`) | 0.01    |
| fields flagged           | `currency`  | none          |

```
FIELD              STEPS  MEAN DISP  MAX DISP  OVERRIDE
-----------------  -----  ---------  --------  --------
currency               9       0.44      1.00       44%  SUSPECT
line_items[0].qty     10       0.01      0.08        0%
vendor                30       0.01      0.17        0%
total                 39       0.00      0.03        0%
line_items[0].sku     25       0.00      0.02        0%
invoice_id            42       0.00      0.00        0%
```

**Constrained decoding produced 100% schema-valid JSON that was less
accurate than no constraint at all.** 44 of 48 gold leaves correct; the four
errors are exactly the four forced currencies. Freed of the grammar, the
model said `CHF` and scored 1.000.

maskcheck flagged `currency` and nothing else — without labels.

### What this does and does not show

It shows the failure mode is real and that displacement localises it. It is
**not** a claim about how often this happens in the wild: 8 cases, one model,
documents generated from the records, and a conflict I constructed
deliberately. A schema drifting from its data is a common bug, but its
frequency is not measured here.

An earlier corpus without conflicts produced displacement ≈ 0.00 and 1.000
accuracy on both arms — a genuine null result, reported rather than tuned
away. Displacement finds schemas fighting models; where none is fighting,
it correctly says nothing.

## You do not need labelled data

This is the part that matters. Most teams running structured output have no gold
dataset for their schema — which is precisely *why* they can't tell the mask is
hurting them.

So the reference isn't ground truth. It's the model's own unconstrained preference:

```
                  ┌────────────────────────────┐
   prompt ───────►│  engine + grammar/schema   │───► constrained output
                  └────────────────────────────┘              │
                                                              │ same text,
                                                              │ teacher-forced
                  ┌────────────────────────────┐              ▼
                  │  SAME model, NO grammar    │───► per-token logprobs
                  └────────────────────────────┘              │
                                                              ▼
                     displacement = 1 − P_unconstrained(token)
                                                              │
                        attributed to a JSON path ◄───────────┘
                                                              │
                                                              ▼
                              invoice.total   0.79   SUSPECT
```

High displacement means the grammar pushed the model somewhere it did not want to
go. Attributed per field, that localises where your schema is fighting your model.

Got labels? `--labels gold.jsonl` upgrades suspicion to measured field-level accuracy.

## Why not just read the mask?

Reading the logit mask from inside an engine means forking llama.cpp, XGrammar and
MLX separately and tracking all three forever. Teacher-forcing recovers the same
signal from *outside* the engine, so one code path works anywhere prefix logprobs
are available. No patches, no forks.

## CI gate

```bash
maskcheck run --engine llamacpp --model ./qwen2.5-1.5b.gguf \
  --schema invoice.json --prompts cases.jsonl \
  --fail-on-displacement 0.5
```

Exit `1` when any field exceeds the threshold. `--json -` emits machine-readable
results on stdout for piping into `jq`.

## Honest limitations

**Displacement is a suspicion signal, not an error measurement.** A field can be
displaced and still correct — the model's preferred phrasing was merely unavailable.
A field can be undisplaced and wrong. It tells you where to look. `--labels` is what
measures.

**Structural tokens are excluded.** A grammar is *supposed* to pin `{`, `,`, `:` and
object keys, so displacement there measures the mask working correctly. Including
them would drag every field toward the same meaningless number. Only value regions
are scored.

**Attribution follows the parsed span.** A model that narrates before
answering can emit more than one complete JSON value — a live run produced
`Line items: [{"sku": "NS-1", "qty": 4}]` inside its prose preamble before
the real object. Displacement is confined to the span `repair()` parsed, so
the two halves of the tool always score the same bytes. Where several values
parse, the last one wins: models narrate first and answer second.

**Pooled means dilute.** If only half your documents conflict with the
schema, a field that is badly wrong half the time averages near 0.44 and
slips under a 0.5 threshold. A field is therefore also flagged when the mask
beat the model's own argmax on 25% or more of its tokens, which does not
dilute the same way.

**Teacher-forcing needs prefix logprobs.** Local engines expose these. Several hosted
APIs have dropped echo logprobs, so remote coverage is partial and documented per
provider rather than claimed universally.

## Status

Core is complete and tested — 93 tests, zero dependencies, Python 3.9+.

| component | state |
|---|---|
| Path attribution, displacement, scoring, repair, CLI | done |
| `maskcheck demo` (fixture, no model) | done |
| llama.cpp adapter | **verified against a live llama-server** |
| MLX / transformers adapters | planned |

### A note on how llama.cpp scoring works

`llama-server` has no echo-logprobs parameter — `n_probs` reports top-N only
for *generated* tokens, and `n_predict: 0` fills the cache without returning
prompt probabilities. So there is no single call that scores a supplied
string.

The adapter therefore walks the text one token at a time, requesting a single
token at each prefix and reading the probability the unconstrained model gave
to the token that actually followed. `cache_prompt` keeps the KV cache warm,
so cost is ~one forward pass per token rather than a quadratic re-evaluation.

When the emitted token falls outside the top-N window, its probability is
unknown — but such a token is *by definition* heavily displaced, so the
adapter records a **lower bound** and marks the argmax comparison unknown
rather than inventing a number. Transformers-based engines can do this in one
forward pass and will use that faster path.

## License

MIT
