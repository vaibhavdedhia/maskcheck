import math
import unittest

from maskcheck.engines.base import EngineError
from maskcheck.engines.llamacpp import OUTSIDE_TOPN_PROB, LlamaCppEngine


class StubTransport(object):
    """Records requests; replays canned responses per endpoint."""

    def __init__(self, pieces=None, probs_per_step=None, content="{}"):
        self.calls = []
        self.pieces = pieces or []
        self.probs_per_step = probs_per_step or []
        self.content = content
        self._step = 0

    def __call__(self, path, payload):
        self.calls.append((path, payload))
        if path == "/tokenize":
            return {"tokens": [{"piece": p} for p in self.pieces]}
        if path == "/completion":
            if payload.get("n_predict", 0) > 1:
                return {"content": self.content}
            probs = self.probs_per_step[self._step]
            self._step += 1
            return {"completion_probabilities": [{"probs": probs}]}
        raise AssertionError("unexpected path " + path)


def probs(*pairs):
    return [{"token": t, "prob": p} for t, p in pairs]


class TestGenerate(unittest.TestCase):
    def test_schema_is_passed_through(self):
        t = StubTransport(content='{"a": 1}')
        e = LlamaCppEngine(transport=t)
        out = e.generate("prompt", schema={"type": "object"}, max_tokens=64)
        self.assertEqual(out, '{"a": 1}')
        _, payload = t.calls[0]
        self.assertEqual(payload["json_schema"], {"type": "object"})

    def test_unconstrained_omits_schema(self):
        t = StubTransport(content="hi")
        LlamaCppEngine(transport=t).generate("p", schema=None, max_tokens=64)
        self.assertNotIn("json_schema", t.calls[0][1])

    def test_generation_is_deterministic(self):
        # Sampling noise would confound displacement.
        t = StubTransport(content="x")
        LlamaCppEngine(transport=t).generate("p", max_tokens=64)
        self.assertEqual(t.calls[0][1]["temperature"], 0.0)

    def test_missing_content_raises(self):
        class Bad(StubTransport):
            def __call__(self, path, payload):
                return {}
        with self.assertRaises(EngineError):
            LlamaCppEngine(transport=Bad()).generate("p")


class TestScore(unittest.TestCase):
    def test_offsets_and_logprobs(self):
        pieces = ['{"a"', ":", " 1", "}"]
        t = StubTransport(pieces=pieces, probs_per_step=[
            probs(('{"a"', 0.9), ("x", 0.1)),
            probs((":", 0.8), ("y", 0.2)),
            probs((" 1", 0.4), (" 2", 0.6)),
            probs(("}", 0.95), ("z", 0.05)),
        ])
        steps = LlamaCppEngine(transport=t).score("P", '{"a": 1}')
        self.assertEqual([s.text for s in steps], pieces)
        self.assertEqual([s.offset for s in steps], [0, 4, 5, 7])
        self.assertAlmostEqual(steps[2].probability, 0.4, places=6)
        # ' 2' was preferred, so the mask overrode the argmax here.
        self.assertTrue(steps[2].overridden)
        self.assertFalse(steps[0].overridden)

    def test_prefix_grows_and_cache_is_reused(self):
        pieces = ["a", "b"]
        t = StubTransport(pieces=pieces, probs_per_step=[
            probs(("a", 0.5)), probs(("b", 0.5))])
        LlamaCppEngine(transport=t).score("P", "ab")
        completions = [p for path, p in t.calls if path == "/completion"]
        self.assertEqual(completions[0]["prompt"], "P")
        self.assertEqual(completions[1]["prompt"], "Pa")
        self.assertTrue(all(c["cache_prompt"] for c in completions))
        self.assertTrue(all(c["n_predict"] == 1 for c in completions))

    def test_token_outside_topn_becomes_lower_bound(self):
        # The emitted token isn't in the reported window at all.
        t = StubTransport(pieces=["q"], probs_per_step=[
            probs(("a", 0.5), ("b", 0.3), ("c", 0.2))])
        step = LlamaCppEngine(transport=t).score("P", "q")[0]
        # Tolerance because the value round-trips through log/exp.
        self.assertAlmostEqual(step.probability, OUTSIDE_TOPN_PROB, delta=1e-12)
        self.assertGreater(step.displacement, 0.99)
        # No trustworthy argmax comparison, so override stays unknown
        # rather than being asserted.
        self.assertIsNone(step.overridden)

    def test_logprob_uses_logprob_field_when_prob_absent(self):
        t = StubTransport(pieces=["a"], probs_per_step=[
            [{"token": "a", "logprob": math.log(0.25)}]])
        step = LlamaCppEngine(transport=t).score("P", "a")[0]
        self.assertAlmostEqual(step.probability, 0.25, places=6)

    def test_misaligned_pieces_raise_rather_than_misattribute(self):
        # Pieces that don't reassemble would shift every offset.
        t = StubTransport(pieces=["a", "b"], probs_per_step=[
            probs(("a", 0.5)), probs(("b", 0.5))])
        with self.assertRaises(EngineError):
            LlamaCppEngine(transport=t).score("P", "aXb")

    def test_missing_probabilities_gives_actionable_error(self):
        class NoProbs(StubTransport):
            def __call__(self, path, payload):
                self.calls.append((path, payload))
                if path == "/tokenize":
                    return {"tokens": [{"piece": "a"}]}
                return {"content": "a"}
        with self.assertRaises(EngineError) as ctx:
            LlamaCppEngine(transport=NoProbs()).score("P", "a")
        self.assertIn("n_probs", str(ctx.exception))

    def test_tokenize_without_pieces_gives_actionable_error(self):
        class OldBuild(StubTransport):
            def __call__(self, path, payload):
                return {"tokens": [1, 2, 3]}
        with self.assertRaises(EngineError) as ctx:
            LlamaCppEngine(transport=OldBuild()).score("P", "abc")
        self.assertIn("pieces", str(ctx.exception))

    def test_empty_tokenize_raises(self):
        class Empty(StubTransport):
            def __call__(self, path, payload):
                return {"tokens": []}
        with self.assertRaises(EngineError):
            LlamaCppEngine(transport=Empty()).score("P", "a")


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestChatTemplate(unittest.TestCase):
    def test_apply_template_returns_prompt(self):
        class T(StubTransport):
            def __call__(self, path, payload):
                self.calls.append((path, payload))
                assert path == "/apply-template"
                return {"prompt": "<|user|>hi<|assistant|>"}
        t = T()
        self.assertEqual(LlamaCppEngine(transport=t).apply_template("hi"),
                         "<|user|>hi<|assistant|>")
        self.assertEqual(t.calls[0][1]["messages"][0]["content"], "hi")

    def test_missing_prompt_field_raises(self):
        class T(StubTransport):
            def __call__(self, path, payload):
                return {}
        with self.assertRaises(EngineError):
            LlamaCppEngine(transport=T()).apply_template("hi")
