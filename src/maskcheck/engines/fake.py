"""A deterministic engine with no model behind it.

Exists so `maskcheck demo` works on a clean machine with nothing
installed, and so the whole pipeline is testable without GBs of weights.
It is not a simulation of any real model -- it replays a scripted
generation, which is exactly what a fixture should do.
"""

import json
from typing import Dict, List, Optional

from ..displacement import Step
from .base import Engine, steps_from_tokens, verify_alignment


class FakeEngine(Engine):
    """Replays a fixed output with fixed per-token probabilities.

    `script` maps each token to the unconstrained probability the
    "model" assigned it, plus what it would have preferred instead.
    """

    name = "fake"

    def __init__(self, tokens, probs, top_tokens=None):
        # type: (List[str], List[float], Optional[List[Optional[str]]]) -> None
        if len(tokens) != len(probs):
            raise ValueError("tokens/probs length mismatch")
        self._tokens = tokens
        self._probs = probs
        self._top = top_tokens

    @classmethod
    def demo(cls) -> "FakeEngine":
        """A schema-valid invoice whose *total* was heavily displaced.

        The shape the tool exists to catch: perfect JSON, and the one
        numeric field the model was fighting is the wrong number.
        """
        tokens = ['{"', 'invoice_id', '": "', 'INV', '-', '2291', '", "',
                  'vendor', '": "', 'Acme', ' Corp', '", "',
                  'total', '": ', '48', '.', '0', '0', '}']
        probs = [0.99, 0.97, 0.99, 0.95, 0.96, 0.93, 0.99,
                 0.98, 0.99, 0.91, 0.88, 0.99,
                 0.97, 0.99, 0.12, 0.31, 0.19, 0.22, 0.99]
        top = [None, None, None, None, None, None, None,
               None, None, None, None, None,
               None, None, '1', '2', '4', '0', None]
        return cls(tokens, probs, top)

    def generate(self, prompt, schema=None, max_tokens=512):
        # type: (str, Optional[dict], int) -> str
        return "".join(self._tokens)

    def score(self, prompt, text):
        # type: (str, str) -> List[Step]
        import math
        steps = steps_from_tokens(
            self._tokens,
            [math.log(p) for p in self._probs],
            top_tokens=self._top,
        )
        verify_alignment(text, steps)
        return steps
