"""maskcheck command line.

Exit codes: 0 clean, 1 threshold exceeded (for CI), 2 usage/runtime error.
"""

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .displacement import DEFAULT_SUSPECT_DISPLACEMENT, analyze, worst_displacement
from .report import render_json, render_summary, render_table

EXIT_OK = 0
EXIT_THRESHOLD = 1
EXIT_ERROR = 2


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="maskcheck",
        description="Find the fields where constrained decoding is silently "
                    "mangling your values.")
    p.add_argument("--version", action="version", version="maskcheck " + __version__)
    sub = p.add_subparsers(dest="command")

    demo = sub.add_parser("demo", help="run against a scripted fixture; no model needed")
    _add_common(demo)

    run = sub.add_parser("run", help="run against a real engine")
    run.add_argument("--engine", required=True, choices=["llamacpp", "fake"])
    run.add_argument("--model", help="model path or server URL")
    run.add_argument("--schema", help="JSON Schema file to constrain with")
    run.add_argument("--prompts", help="JSONL file of prompts")
    run.add_argument("--labels", help="JSONL of gold records; upgrades to measured accuracy")
    run.add_argument("--max-tokens", type=int, default=512)
    _add_common(run)
    return p


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--fail-on-displacement", type=float, metavar="X",
                   help="exit 1 if worst mean displacement exceeds X (CI gate)")
    p.add_argument("--suspect-threshold", type=float,
                   default=DEFAULT_SUSPECT_DISPLACEMENT,
                   help="mean displacement at which a field is flagged "
                        "(default: %(default)s)")
    p.add_argument("--json", dest="json_out", metavar="PATH",
                   help="write machine-readable results here ('-' for stdout)")


def _emit(reports, args, scores=None, extra=None) -> int:
    threshold = args.fail_on_displacement
    if args.json_out:
        blob = render_json(reports, threshold, scores, extra)
        if args.json_out == "-":
            sys.stdout.write(blob)
        else:
            with open(args.json_out, "w") as fh:
                fh.write(blob)
    if args.json_out != "-":
        sys.stdout.write(render_table(reports))
        sys.stdout.write(render_summary(reports, threshold, scores))
    if threshold is not None and worst_displacement(reports) > threshold:
        return EXIT_THRESHOLD
    return EXIT_OK


def _cmd_demo(args) -> int:
    from .engines.fake import FakeEngine
    engine = FakeEngine.demo()
    text = engine.generate("")
    reports = analyze(text, engine.score("", text),
                      suspect_threshold=args.suspect_threshold)
    if args.json_out != "-":
        # stdout must stay pure JSON when it IS the JSON sink.
        sys.stdout.write("Generated (schema-valid):\n  %s\n\n" % text)
    return _emit(reports, args, extra={"generated": text, "engine": engine.name})


def _cmd_run(args) -> int:
    if args.engine == "fake":
        return _cmd_demo(args)
    sys.stderr.write(
        "engine '%s' is not wired up yet in this build.\n"
        "Run `maskcheck demo` to see the analysis on a fixture.\n" % args.engine)
    return EXIT_ERROR


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return EXIT_ERROR
    try:
        if args.command == "demo":
            return _cmd_demo(args)
        if args.command == "run":
            return _cmd_run(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write("maskcheck: %s\n" % exc)
        return EXIT_ERROR
    parser.print_help()
    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
