"""llama-server adapter.

`llama-server` has no echo-logprobs parameter: `n_probs` reports top-N only
for *generated* tokens, and `n_predict: 0` evaluates the prompt into the
cache without returning probabilities for it. So there is no way to score a
supplied string in one call.

Instead this walks the text one token at a time, asking for a single token
at each prefix and reading what probability the unconstrained model gave to
the token that actually followed. `cache_prompt` keeps the KV cache warm
across calls, so the cost is ~one forward pass per token plus HTTP overhead
rather than a quadratic re-evaluation.

Uses urllib from the standard library -- no `requests`, so the adapter
keeps the project's zero-dependency promise.
"""

import json
import math
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..displacement import Step
from .base import Engine, EngineError, verify_alignment

# llama.cpp caps n_probs; 20 is widely supported and deep enough that a
# token outside it is unambiguously displaced.
DEFAULT_N_PROBS = 20

# Floor for tokens that fall outside the top-N window. Their true
# probability is below the smallest reported one, so displacement is a
# lower bound -- flagged in `Step.top_logprob is None` and reported as such.
OUTSIDE_TOPN_PROB = 1e-6

Transport = Callable[[str, Dict[str, Any]], Dict[str, Any]]


def _http_transport(base_url: str, timeout: float) -> Transport:
    def post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = base_url.rstrip("/") + path
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise EngineError("llama-server unreachable at %s: %s" % (url, exc))
        except ValueError as exc:
            raise EngineError("llama-server returned non-JSON: %s" % exc)
    return post


class LlamaCppEngine(Engine):
    name = "llamacpp"

    def __init__(self, base_url="http://127.0.0.1:8080", n_probs=DEFAULT_N_PROBS,
                 timeout=120.0, transport=None):
        # type: (str, int, float, Optional[Transport]) -> None
        self.base_url = base_url
        self.n_probs = n_probs
        self._post = transport or _http_transport(base_url, timeout)

    # ---- prompt templating -----------------------------------------
    def apply_template(self, prompt: str) -> str:
        """Wrap `prompt` in the model's own chat template.

        Instruct-tuned models fed a raw prompt can fail badly -- Llama-3.2-1B
        degenerates into repeating a line forever, which looks exactly like
        "this model cannot produce JSON" and is really "this model was never
        addressed properly". The template lives in the GGUF metadata, so
        llama-server can apply the right one rather than the caller
        hardcoding one per family.
        """
        resp = self._post("/apply-template",
                          {"messages": [{"role": "user", "content": prompt}]})
        out = resp.get("prompt")
        if not isinstance(out, str):
            raise EngineError("/apply-template returned no 'prompt'")
        return out

    # ---- generation -------------------------------------------------
    def generate(self, prompt, schema=None, max_tokens=512):
        # type: (str, Optional[dict], int) -> str
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0.0,   # deterministic: displacement should not
                                  # be confounded by sampling noise
            "cache_prompt": True,
        }
        if schema is not None:
            payload["json_schema"] = schema
        resp = self._post("/completion", payload)
        content = resp.get("content")
        if not isinstance(content, str):
            raise EngineError("no 'content' in /completion response")
        return content

    # ---- scoring ----------------------------------------------------
    def _pieces(self, text: str) -> List[str]:
        """Split `text` into the model's own token pieces."""
        resp = self._post("/tokenize", {"content": text, "with_pieces": True})
        toks = resp.get("tokens")
        if not isinstance(toks, list) or not toks:
            raise EngineError(
                "/tokenize returned no tokens (does this build support "
                "with_pieces?)")
        pieces = []  # type: List[str]
        for t in toks:
            if isinstance(t, dict):
                piece = t.get("piece")
                if not isinstance(piece, str):
                    raise EngineError("token entry missing 'piece'")
                pieces.append(piece)
            else:
                raise EngineError(
                    "/tokenize did not return pieces; upgrade llama.cpp or "
                    "pass with_pieces support")
        return pieces

    @staticmethod
    def _lookup(probs: Sequence[Dict[str, Any]], piece: str) -> Tuple[float, bool]:
        """Probability the model gave `piece`, and whether it was in top-N.

        Falls back to a floor when the token is outside the reported window,
        which makes the resulting displacement a lower bound rather than a
        fabricated number.
        """
        smallest = 1.0
        for entry in probs:
            tok = entry.get("token", entry.get("tok_str"))
            p = entry.get("prob")
            if p is None and "logprob" in entry:
                p = math.exp(entry["logprob"])
            if p is None:
                continue
            if tok == piece:
                return max(float(p), OUTSIDE_TOPN_PROB), True
            smallest = min(smallest, float(p))
        return max(min(smallest, OUTSIDE_TOPN_PROB), 1e-12), False

    def score(self, prompt, text):
        # type: (str, str) -> List[Step]
        pieces = self._pieces(text)
        steps = []  # type: List[Step]
        offset = 0
        for i, piece in enumerate(pieces):
            resp = self._post("/completion", {
                "prompt": prompt + "".join(pieces[:i]),
                "n_predict": 1,
                "n_probs": self.n_probs,
                "temperature": 0.0,
                "cache_prompt": True,
            })
            probs = self._top_probs(resp)
            p, in_topn = self._lookup(probs, piece)
            top_token, top_logprob = self._argmax(probs)
            steps.append(Step(
                text=piece,
                offset=offset,
                logprob=math.log(p),
                # A token outside the window has no trustworthy argmax
                # comparison, so the override flag is left unknown.
                top_token=top_token if in_topn else None,
                top_logprob=top_logprob if in_topn else None,
            ))
            offset += len(piece)
        verify_alignment(text, steps)
        return steps

    @staticmethod
    def _top_probs(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
        cps = resp.get("completion_probabilities")
        if not cps:
            raise EngineError(
                "no 'completion_probabilities' in response; start "
                "llama-server without --no-prob-output and pass n_probs > 0")
        first = cps[0]
        for key in ("probs", "top_probs", "top_logprobs"):
            if key in first:
                return list(first[key])
        raise EngineError("unrecognised completion_probabilities shape")

    @staticmethod
    def _argmax(probs: Sequence[Dict[str, Any]]) -> Tuple[Optional[str], Optional[float]]:
        best, best_p = None, -1.0
        for entry in probs:
            p = entry.get("prob")
            if p is None and "logprob" in entry:
                p = math.exp(entry["logprob"])
            if p is None:
                continue
            if float(p) > best_p:
                best_p = float(p)
                best = entry.get("token", entry.get("tok_str"))
        if best is None:
            return None, None
        return best, math.log(max(best_p, 1e-12))
