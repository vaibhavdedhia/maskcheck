"""maskcheck command line.

Exit codes: 0 clean, 1 threshold exceeded (for CI), 2 usage/runtime error.
"""

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .displacement import (DEFAULT_SUSPECT_DISPLACEMENT, analyze,
                           analyze_many, worst_displacement)
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
    run.add_argument("--chat", action="store_true",
                     help="wrap each prompt in the model's chat template "
                          "(needed for instruct models; without it they may "
                          "not answer at all)")
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


def _read_jsonl(path: str) -> List[object]:
    out = []  # type: List[object]
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError as exc:
                raise ValueError("%s line %d: %s" % (path, lineno, exc))
    return out


def _as_prompt(entry: object) -> str:
    """Accept either a bare JSON string or {"prompt": ...} per line."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
        return entry["prompt"]
    raise ValueError("each prompt line must be a string or have a 'prompt' key")


def _cmd_run(args) -> int:
    if args.engine == "fake":
        return _cmd_demo(args)

    if not args.prompts:
        sys.stderr.write("--prompts is required for --engine %s\n" % args.engine)
        return EXIT_ERROR

    from .engines.llamacpp import LlamaCppEngine
    engine = LlamaCppEngine(base_url=args.model or "http://127.0.0.1:8080")

    schema = None
    if args.schema:
        with open(args.schema) as fh:
            schema = json.load(fh)

    prompts = [_as_prompt(e) for e in _read_jsonl(args.prompts)]
    labels = _read_jsonl(args.labels) if args.labels else None
    if labels is not None and len(labels) != len(prompts):
        sys.stderr.write("--labels has %d records but --prompts has %d\n"
                         % (len(labels), len(prompts)))
        return EXIT_ERROR

    pairs = []
    texts = []
    for i, prompt in enumerate(prompts):
        if getattr(args, "chat", False):
            # Same templated prompt for generation AND scoring, or the
            # teacher-forced logprobs would be conditioned on a different
            # context than the one that produced the text.
            prompt = engine.apply_template(prompt)
        text = engine.generate(prompt, schema, args.max_tokens)
        pairs.append((text, engine.score(prompt, text)))
        texts.append(text)
        sys.stderr.write("\rscored %d/%d" % (i + 1, len(prompts)))
    sys.stderr.write("\n")

    reports = analyze_many(pairs, suspect_threshold=args.suspect_threshold)

    scores = None
    if labels is not None:
        from .repair import repair
        from .scoring import aggregate, score_record
        per = []
        for text, gold in zip(texts, labels):
            parsed, _stage = repair(text)
            per.append(score_record(gold, parsed))
        scores = aggregate(per)

    return _emit(reports, args, scores=scores,
                 extra={"engine": engine.name, "prompts": len(prompts),
                        "schema": args.schema})


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
