"""Engine interface: generate under constraint, then score without it.

Deliberately two phases rather than one. Reading the mask from inside an
engine would mean forking llama.cpp, XGrammar and MLX separately and
tracking all three forever. Teacher-forcing the constrained output back
through the *unconstrained* model recovers the same signal from outside,
so one code path serves every engine that can report prefix logprobs.
"""

import abc
from typing import List, Optional, Sequence

from ..displacement import Step


class EngineError(RuntimeError):
    pass


class Engine(abc.ABC):
    """Minimal surface an engine must expose to be diagnosable."""

    name = "engine"

    @abc.abstractmethod
    def generate(self, prompt: str, schema: Optional[dict], max_tokens: int = 512) -> str:
        """Generate, grammar-constrained when `schema` is given."""

    @abc.abstractmethod
    def score(self, prompt: str, text: str) -> List[Step]:
        """Unconstrained per-token logprobs for `text` forced after `prompt`."""

    def close(self) -> None:
        pass


def steps_from_tokens(tokens,
                      logprobs,
                      top_tokens=None,
                      top_logprobs=None):
    # type: (Sequence[str], Sequence[float], Optional[Sequence[Optional[str]]], Optional[Sequence[Optional[float]]]) -> List[Step]
    """Build Steps, deriving char offsets by concatenation.

    Raises if the pieces do not reassemble, because a tokenizer whose
    detokenised pieces don't concatenate to the generated text would shift
    every offset and silently misattribute the entire report. Better to
    fail loudly than to publish a confidently wrong table.
    """
    if len(tokens) != len(logprobs):
        raise EngineError("tokens/logprobs length mismatch: %d vs %d"
                          % (len(tokens), len(logprobs)))
    steps = []  # type: List[Step]
    offset = 0
    for i, tok in enumerate(tokens):
        steps.append(Step(
            text=tok,
            offset=offset,
            logprob=logprobs[i],
            top_token=top_tokens[i] if top_tokens else None,
            top_logprob=top_logprobs[i] if top_logprobs else None,
        ))
        offset += len(tok)
    return steps


def verify_alignment(text: str, steps: Sequence[Step]) -> None:
    """Confirm the steps actually tile `text`."""
    rebuilt = "".join(s.text for s in steps)
    if rebuilt != text:
        raise EngineError(
            "token pieces do not reassemble the generated text "
            "(%d chars from tokens vs %d generated); offsets would be wrong"
            % (len(rebuilt), len(text)))
