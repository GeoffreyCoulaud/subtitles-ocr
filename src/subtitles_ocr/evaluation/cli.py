"""subtitles-ocr-evaluate CLI entry point (ADR-0006 §7.4)."""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

from subtitles_ocr.evaluation.report import Weights, default_weights
from subtitles_ocr.evaluation.score import score


def _parse_fps(raw: str) -> Fraction:
    if "/" in raw:
        n, d = raw.split("/", 1)
        return Fraction(int(n), int(d))
    return Fraction(int(raw))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="subtitles-ocr-evaluate")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--fps", required=True, type=_parse_fps)
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--json", action="store_true")
    return p


def _load_weights(path: Path | None) -> Weights:
    if path is None:
        return default_weights()
    return Weights.model_validate_json(path.read_text(encoding="utf-8"))


def _format_human(report) -> str:
    lines = [f"final: {report.final:.4f}", "sub-scores:"]
    for name, value in report.sub_scores.items():
        rendered = "null" if value is None else f"{value:.4f}"
        weight = report.effective_weights.get(name, 0)
        lines.append(f"  {name:<12} = {rendered}   (weight {weight})")
    lines.append(f"matched: {report.n_matched}/{report.n_ref} ref, {report.n_matched}/{report.n_out} out")
    if report.warnings:
        lines.append("warnings:")
        lines.extend(f"  - {w}" for w in report.warnings)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    weights = _load_weights(args.weights)
    try:
        report = score(args.output, args.reference, weights=weights, fps=args.fps)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(report.model_dump_json())
    else:
        print(_format_human(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
