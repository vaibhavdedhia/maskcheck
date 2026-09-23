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

**Teacher-forcing needs prefix logprobs.** Local engines expose these. Several hosted
APIs have dropped echo logprobs, so remote coverage is partial and documented per
provider rather than claimed universally.

## Status

Core is complete and tested — 86 tests, zero dependencies, Python 3.9+.

| component | state |
|---|---|
| Path attribution, displacement, scoring, repair, CLI | done |
| `maskcheck demo` (fixture, no model) | done |
| llama.cpp adapter | implemented; **not yet run against a live server** |
| MLX / transformers adapters | planned |

The llama.cpp adapter is unit-tested against a stub transport, so its request
shapes and error paths are covered, but no number in this README came from a
real model yet. That is stated here rather than discovered by you.

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
