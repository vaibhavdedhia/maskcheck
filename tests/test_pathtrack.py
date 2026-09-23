import random
import unittest

from maskcheck.pathtrack import PathTracker, path_at_offsets

DOC = ('{"invoice": {"id": "INV-77", "total": 1240.5, "paid": false}, '
       '"lines": [{"sku": "A1", "qty": 2}, {"sku": "B2", "qty": 1}], '
       '"note": null}')


def path_of(prefix):
    return PathTracker().feed(prefix).path()


class TestPartialPrefixes(unittest.TestCase):
    def test_nested_object(self):
        self.assertEqual(path_of('{"a": 1, "b": {"c": '), "b.c")

    def test_array_of_objects(self):
        self.assertEqual(path_of('{"lines": [{"sku": "A'), "lines[0].sku")

    def test_second_array_element(self):
        self.assertEqual(path_of('{"lines": [{"sku": "A1"}, {"sku": "B'), "lines[1].sku")

    def test_structural_chars_in_string_are_inert(self):
        # The reason bracket-counting cannot do this job.
        self.assertEqual(path_of('{"a": "x{y}[z]", "b":'), "b")

    def test_escaped_quote_in_value(self):
        self.assertEqual(path_of(r'{"a": "say \"hi\"", "b": '), "b")

    def test_escaped_quote_in_key(self):
        self.assertEqual(path_of(r'{"we\"ird": '), 'we"ird')

    def test_empty_prefix_is_root(self):
        self.assertEqual(path_of(""), "")
        self.assertEqual(path_of("{"), "")

    def test_key_not_yet_closed_is_not_a_path(self):
        # Mid-key: the key is incomplete, so it must not be reported.
        self.assertEqual(path_of('{"inv'), "")

    def test_in_key_flag(self):
        self.assertTrue(PathTracker().feed('{"inv').in_key())
        self.assertFalse(PathTracker().feed('{"inv": ').in_key())

    def test_bare_array_root(self):
        self.assertEqual(path_of('[{"a": [1, 2'), "[0].a[1]")

    def test_number_then_comma_advances(self):
        self.assertEqual(path_of('{"a": 1240.5, "b": '), "b")

    def test_literal_then_comma_advances(self):
        self.assertEqual(path_of('{"a": false, "b": '), "b")
        self.assertEqual(path_of('{"a": null, "b": '), "b")

    def test_closing_nested_object_restores_parent(self):
        self.assertEqual(path_of('{"a": {"b": 1}, "c": '), "c")

    def test_deep_nesting(self):
        self.assertEqual(path_of('{"a": [{"b": {"c": [{"d": "'), "a[0].b.c[0].d")

    def test_completed_document(self):
        t = PathTracker().feed(DOC)
        self.assertEqual(t.depth(), 0)
        self.assertEqual(t.path(), "")


class TestChunkInvariance(unittest.TestCase):
    """Token boundaries are arbitrary; chunking must not change the answer."""

    def _all_chunkings(self, prefix, expected):
        self.assertEqual(path_of(prefix), expected, "whole-string")
        # char by char
        t = PathTracker()
        for ch in prefix:
            t.feed(ch)
        self.assertEqual(t.path(), expected, "char-by-char")
        # random splits, incl. splits inside strings, numbers and escapes
        rnd = random.Random(1234)
        for trial in range(200):
            t = PathTracker()
            i = 0
            while i < len(prefix):
                j = min(len(prefix), i + rnd.randint(1, 5))
                t.feed(prefix[i:j])
                i = j
            self.assertEqual(t.path(), expected, "random split trial %d" % trial)

    def test_invariance_nested(self):
        self._all_chunkings('{"lines": [{"sku": "A1", "qty": 2}, {"sku": "B', "lines[1].sku")

    def test_invariance_with_escapes(self):
        self._all_chunkings(r'{"a": "say \"hi\"", "b": [1, 2, 3', "b[2]")

    def test_invariance_every_prefix_of_doc(self):
        # Feeding DOC[:n] whole vs char-by-char must agree for every n.
        for n in range(len(DOC) + 1):
            whole = PathTracker().feed(DOC[:n]).path()
            t = PathTracker()
            for ch in DOC[:n]:
                t.feed(ch)
            self.assertEqual(whole, t.path(), "prefix length %d" % n)


class TestPathAtOffsets(unittest.TestCase):
    def test_matches_independent_reparse(self):
        offsets = list(range(0, len(DOC) + 1, 3))
        fast = path_at_offsets(DOC, offsets)
        slow = [(PathTracker().feed(DOC[:o]).path(),
                 PathTracker().feed(DOC[:o]).region()) for o in offsets]
        self.assertEqual(fast, slow)

    def test_rejects_descending_offsets(self):
        with self.assertRaises(ValueError):
            path_at_offsets(DOC, [10, 5])

    def test_attributes_expected_fields(self):
        i = DOC.index("1240.5")
        self.assertEqual(path_at_offsets(DOC, [i + 2])[0], ("invoice.total", "number"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
