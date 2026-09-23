import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from maskcheck.cli import EXIT_ERROR, EXIT_OK, EXIT_THRESHOLD, main
from maskcheck.displacement import FieldReport
from maskcheck.report import render_json, render_table


def run_cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestDemo(unittest.TestCase):
    def test_demo_runs_clean(self):
        code, out, _ = run_cli(["demo"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("SUSPECT", out)
        self.assertIn("total", out)

    def test_demo_output_is_valid_json(self):
        # The whole premise: the output parses, and is still wrong.
        _, out, _ = run_cli(["demo"])
        generated = out.splitlines()[1].strip()
        self.assertEqual(json.loads(generated)["total"], 48.0)

    def test_caveat_is_shown_when_flagging(self):
        # Never let a suspect field be read as a proven error.
        _, out, _ = run_cli(["demo"])
        self.assertIn("not proof of a wrong value", out)


class TestCIGate(unittest.TestCase):
    def test_exceeding_threshold_exits_1(self):
        code, _, _ = run_cli(["demo", "--fail-on-displacement", "0.5"])
        self.assertEqual(code, EXIT_THRESHOLD)

    def test_under_threshold_exits_0(self):
        code, out, _ = run_cli(["demo", "--fail-on-displacement", "0.95"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("PASS", out)

    def test_threshold_verdict_printed(self):
        _, out, _ = run_cli(["demo", "--fail-on-displacement", "0.5"])
        self.assertIn("FAIL", out)

    def test_high_override_survives_a_raised_threshold(self):
        # Raising the displacement threshold must NOT silence a field the
        # mask is overriding constantly -- that is the dilution case.
        _, out, _ = run_cli(["demo", "--suspect-threshold", "0.99"])
        self.assertIn("SUSPECT", out)
        self.assertIn("override 75%", out)


class TestJSONOutput(unittest.TestCase):
    def test_stdout_json_is_parseable_and_suppresses_table(self):
        code, out, _ = run_cli(["demo", "--json", "-"])
        self.assertEqual(code, EXIT_OK)
        payload = json.loads(out)
        self.assertEqual(payload["suspect"], ["total"])
        self.assertNotIn("SUSPECT", out)

    def test_json_to_file(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            code, out, _ = run_cli(["demo", "--json", path])
            self.assertEqual(code, EXIT_OK)
            with open(path) as fh:
                payload = json.load(fh)
            self.assertIn("worst_mean_displacement", payload)
            self.assertIn("SUSPECT", out)  # table still printed
        finally:
            os.unlink(path)

    def test_json_records_pass_fail(self):
        _, out, _ = run_cli(["demo", "--json", "-", "--fail-on-displacement", "0.5"])
        self.assertIs(json.loads(out)["passed"], False)


class TestUsage(unittest.TestCase):
    def test_no_command_prints_help(self):
        code, out, _ = run_cli([])
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("usage", out.lower())

    def test_run_requires_prompts(self):
        code, _, err = run_cli(["run", "--engine", "llamacpp"])
        self.assertEqual(code, EXIT_ERROR)
        self.assertIn("--prompts is required", err)

    def test_fake_engine_via_run_matches_demo(self):
        code, out, _ = run_cli(["run", "--engine", "fake"])
        self.assertEqual(code, EXIT_OK)
        self.assertIn("SUSPECT", out)

    def test_label_count_mismatch_is_rejected(self):
        # Silently zipping mismatched files would score the wrong pairs.
        d = tempfile.mkdtemp()
        prompts = os.path.join(d, "p.jsonl")
        labels = os.path.join(d, "l.jsonl")
        with open(prompts, "w") as fh:
            fh.write('"one"\n"two"\n')
        with open(labels, "w") as fh:
            fh.write('{"a": 1}\n')
        try:
            code, _, err = run_cli(["run", "--engine", "llamacpp",
                                    "--prompts", prompts, "--labels", labels])
            self.assertEqual(code, EXIT_ERROR)
            self.assertIn("1 records but", err)
        finally:
            os.unlink(prompts); os.unlink(labels); os.rmdir(d)

    def test_malformed_jsonl_names_the_line(self):
        d = tempfile.mkdtemp()
        prompts = os.path.join(d, "p.jsonl")
        with open(prompts, "w") as fh:
            fh.write('"ok"\nnot json\n')
        try:
            code, _, err = run_cli(["run", "--engine", "llamacpp",
                                    "--prompts", prompts])
            self.assertEqual(code, EXIT_ERROR)
            self.assertIn("line 2", err)
        finally:
            os.unlink(prompts); os.rmdir(d)


class TestRender(unittest.TestCase):
    def test_empty_reports(self):
        self.assertIn("nothing to report", render_table([]))

    def test_rule_has_no_column_for_blank_header(self):
        r = FieldReport("a")
        r.steps = 1
        lines = render_table([r]).splitlines()
        # Five titled columns get a rule; the blank flag column does not.
        self.assertEqual(len(lines[1].split()), 5)

    def test_json_without_threshold_omits_passed(self):
        self.assertNotIn("passed", json.loads(render_json([])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
