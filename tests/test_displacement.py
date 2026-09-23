import math
import unittest

from maskcheck.displacement import Step, analyze, worst_displacement


def lp(p):
    """Logprob of a probability, for readable fixtures."""
    return math.log(p)


def steps_for(text, spec):
    """spec: list of (substring, probability, top_token_or_None)."""
    out, cursor = [], 0
    for sub, prob, top in spec:
        off = text.index(sub, cursor)
        out.append(Step(sub, off, lp(prob), top_token=top))
        cursor = off + len(sub)
    return out


DOC = '{"id": "A1", "total": 1240.5}'


class TestStep(unittest.TestCase):
    def test_displacement_is_complement_of_probability(self):
        s = Step("x", 0, lp(0.25))
        self.assertAlmostEqual(s.displacement, 0.75)

    def test_positive_logprob_rejected(self):
        with self.assertRaises(ValueError):
            Step("x", 0, 0.5)

    def test_override_unknown_without_top_token(self):
        self.assertIsNone(Step("x", 0, lp(0.5)).overridden)

    def test_override_detected(self):
        self.assertTrue(Step("x", 0, lp(0.1), top_token="y").overridden)
        self.assertFalse(Step("x", 0, lp(0.9), top_token="x").overridden)


class TestAnalyze(unittest.TestCase):
    def test_attributes_displacement_to_the_right_field(self):
        # 'total' is heavily displaced; 'id' is not.
        spec = [("A1", 0.95, "A1"), ("1240", 0.10, "9"), (".5", 0.20, "0")]
        reports = analyze(DOC, steps_for(DOC, spec))
        by_path = {r.path: r for r in reports}
        self.assertAlmostEqual(by_path["id"].mean_displacement, 0.05, places=6)
        self.assertAlmostEqual(by_path["total"].mean_displacement, 0.85, places=6)

    def test_worst_first_ordering(self):
        spec = [("A1", 0.95, "A1"), ("1240", 0.10, "9"), (".5", 0.20, "0")]
        reports = analyze(DOC, steps_for(DOC, spec))
        self.assertEqual([r.path for r in reports], ["total", "id"])

    def test_structural_and_key_tokens_excluded(self):
        # Tokens on '{', the key "id", ':' and ',' must not create fields.
        text = '{"id": "A1"}'
        steps = [
            Step("{", 0, lp(0.99)),
            Step('"id"', 1, lp(0.01), top_token='"x"'),   # key: ignored
            Step(":", 5, lp(0.99)),
            Step(" ", 6, lp(0.99)),
            Step("A1", 8, lp(0.90), top_token="A1"),      # value: counted
        ]
        reports = analyze(text, steps)
        self.assertEqual([r.path for r in reports], ["id"])
        self.assertEqual(reports[0].steps, 1)

    def test_suspect_requires_threshold_and_min_steps(self):
        spec = [("1240", 0.10, "9"), (".5", 0.10, "0")]
        reports = analyze(DOC, steps_for(DOC, spec), suspect_threshold=0.5)
        self.assertTrue(reports[0].suspect)
        # Same displacement, but one step is below min_steps.
        reports = analyze(DOC, steps_for(DOC, [("1240", 0.10, "9")]),
                          suspect_threshold=0.5, min_steps=2)
        self.assertFalse(reports[0].suspect)

    def test_override_rate_uses_only_known_steps(self):
        text = '{"a": "xy"}'
        steps = [Step("x", 7, lp(0.5), top_token="z"),   # overridden
                 Step("y", 8, lp(0.5))]                  # unknown -> excluded
        r = analyze(text, steps)[0]
        self.assertEqual(r.steps, 2)
        self.assertEqual(r.override_rate, 1.0)
        self.assertEqual(r.known_overrides, 1)

    def test_empty_input(self):
        self.assertEqual(analyze(DOC, []), [])
        self.assertEqual(worst_displacement([]), 0.0)

    def test_descending_offsets_rejected(self):
        with self.assertRaises(ValueError):
            analyze(DOC, [Step("a", 10, lp(0.5)), Step("b", 2, lp(0.5))])

    def test_nested_paths(self):
        text = '{"inv": {"lines": [{"sku": "AB"}]}}'
        i = text.index("AB")
        r = analyze(text, [Step("AB", i, lp(0.2), top_token="CD")])[0]
        self.assertEqual(r.path, "inv.lines[0].sku")

    def test_worst_displacement_headline(self):
        spec = [("A1", 0.95, "A1"), ("1240", 0.10, "9"), (".5", 0.20, "0")]
        reports = analyze(DOC, steps_for(DOC, spec))
        self.assertAlmostEqual(worst_displacement(reports), 0.85, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
