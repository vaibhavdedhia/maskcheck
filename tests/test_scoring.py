import unittest

from maskcheck.repair import repair, _balanced_spans
from maskcheck.scoring import aggregate, compare, flatten, score_record

GOLD = {"invoice": {"id": "INV-77", "total": 1240.5, "paid": False},
        "lines": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]}


class TestRepair(unittest.TestCase):
    def test_clean_json_is_direct(self):
        self.assertEqual(repair('{"a": 1}'), ({"a": 1}, "direct"))

    def test_the_canonical_failure_mode(self):
        # The exact output shape that motivates this whole benchmark.
        raw = 'Sure, here is your JSON!\n\n```json\n{"a": 1}\n```\n\nLet me know!'
        self.assertEqual(repair(raw), ({"a": 1}, "fence"))

    def test_prose_wrapped_without_fence(self):
        got, stage = repair('Here you go: {"a": 1, "b": "x"} Hope that helps.')
        self.assertEqual((got, stage), ({"a": 1, "b": "x"}, "balanced"))

    def test_braces_inside_strings_do_not_break_scanning(self):
        raw = 'text {"note": "use {curly} braces", "n": 3} tail'
        got, stage = repair(raw)
        self.assertEqual(stage, "balanced")
        self.assertEqual(got["n"], 3)

    def test_trailing_comma(self):
        got, stage = repair('{"a": 1, "b": 2,}')
        self.assertEqual((got, stage), ({"a": 1, "b": 2}, "trailing_comma"))

    def test_unrecoverable_returns_failed(self):
        self.assertEqual(repair("I cannot help with that request."), (None, "failed"))

    def test_mismatched_brackets_do_not_yield_garbage(self):
        # {..] must not be reported as a valid span.
        self.assertEqual(repair('{"a": 1]')[1], "failed")

    def test_escaped_quote_inside_string(self):
        got, stage = repair(r'prefix {"q": "say \"hi\"", "n": 1} suffix')
        self.assertEqual(stage, "balanced")
        self.assertEqual(got["n"], 1)


class TestFlatten(unittest.TestCase):
    def test_paths(self):
        f = flatten(GOLD)
        self.assertIn("invoice.id", f)
        self.assertIn("lines[1].sku", f)
        self.assertEqual(len(f), 7)

    def test_empty_containers_are_leaves(self):
        self.assertEqual(flatten({"a": {}, "b": []}), {"a": "<empty-dict>", "b": "<empty-list>"})


class TestCompare(unittest.TestCase):
    def test_string_normalisation(self):
        self.assertEqual(compare("  Hello   World ", "hello world"), (True, False))

    def test_stringified_number_is_coercion(self):
        self.assertEqual(compare(42, "42"), (True, True))

    def test_int_one_is_not_true(self):
        # The classic bool/int subclass trap.
        self.assertEqual(compare(True, 1), (False, False))
        self.assertEqual(compare(1, True), (False, False))

    def test_bool_word_is_coercion(self):
        self.assertEqual(compare(False, "false"), (True, True))

    def test_none_only_matches_none(self):
        self.assertEqual(compare(None, None), (True, False))
        self.assertEqual(compare(None, ""), (False, False))
        self.assertEqual(compare(0, None), (False, False))

    def test_coercion_can_be_disabled(self):
        self.assertEqual(compare(42, "42", coercion=False), (False, False))


class TestScoreRecord(unittest.TestCase):
    def test_perfect(self):
        s = score_record(GOLD, GOLD)
        self.assertEqual((s.accuracy, s.parsed, s.hallucinated), (1.0, True, 0))

    def test_unparseable_scores_zero_not_crash(self):
        s = score_record(GOLD, None)
        self.assertEqual((s.accuracy, s.parsed, s.missing), (0.0, False, 7))

    def test_partial_credit_and_hallucination(self):
        pred = {"invoice": {"id": "INV-77", "total": 999.0, "paid": False, "vendor": "Acme"},
                "lines": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}]}
        s = score_record(GOLD, pred)
        self.assertEqual(s.correct, 6)
        self.assertEqual(s.wrong, 1)
        self.assertEqual(s.hallucinated, 1)

    def test_coercion_is_excluded_from_strict_types(self):
        pred = {"invoice": {"id": "INV-77", "total": "1240.5", "paid": "false"},
                "lines": [{"sku": "A1", "qty": "2"}, {"sku": "B2", "qty": 1}]}
        s = score_record(GOLD, pred)
        self.assertEqual(s.accuracy, 1.0)
        self.assertEqual(s.coerced, 3)
        self.assertAlmostEqual(s.accuracy_strict_types, 4 / 7)


class TestAggregate(unittest.TestCase):
    def test_micro_average_and_rates(self):
        agg = aggregate([score_record(GOLD, GOLD), score_record(GOLD, None)])
        self.assertEqual(agg["records"], 2)
        self.assertEqual(agg["parse_rate"], 0.5)
        self.assertEqual(agg["leaves"], 14)
        self.assertEqual(agg["field_accuracy"], 0.5)
        self.assertEqual(agg["exact_record_rate"], 0.5)

    def test_empty_is_safe(self):
        self.assertEqual(aggregate([]), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
