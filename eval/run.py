"""Run the evaluation suite and regenerate ``eval/RESULTS.md``.

Usage::

    python -m eval.run                                  # offline fallback + ablation
    python -m eval.run --systems zero_shot_claude hybrid --judge   # needs ANTHROPIC_API_KEY
    python -m eval.run --systems distilbert_only        # needs the ``ml`` extra + weights

Systems that cannot run in the current environment are skipped with the
reason printed, and their cells stay ``pendiente (...)`` in the report.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import asdict
from pathlib import Path

from src.config import Settings, get_settings
from src.eval.baselines import (
    SYSTEM_DISTILBERT,
    SYSTEM_HYBRID,
    SYSTEM_OFFLINE,
    SYSTEM_ZERO_SHOT,
    ablate_confidence,
    build_distilbert_only,
    build_hybrid,
    build_offline_fallback,
    build_zero_shot_claude,
    check_availability,
)
from src.eval.judge import ResponseJudge, judge_batch
from src.eval.report import RESULTS_MD, RUNS_DIR, update_results, write_run
from src.eval.runner import format_report, load_cases, run_pipeline_eval
from src.llm.claude import ClaudeClient
from src.observability import configure_logging

ALL_SYSTEMS = (SYSTEM_OFFLINE, SYSTEM_ZERO_SHOT, SYSTEM_DISTILBERT, SYSTEM_HYBRID)


async def run_system(name: str, settings: Settings, *, judge: bool, runs_dir: Path) -> Path | None:
    """Run one system if its requirements are met; return the run file path."""
    avail = check_availability(settings)
    client = (
        ClaudeClient(
            api_key=settings.anthropic_api_key or "",
            model=settings.anthropic_model,
            timeout_s=settings.llm_timeout_s,
            max_attempts=settings.llm_max_attempts,
        )
        if avail.llm
        else None
    )
    if name == SYSTEM_OFFLINE:
        pipeline = await build_offline_fallback(settings)
    elif name == SYSTEM_ZERO_SHOT:
        if client is None:
            print(f"[skip] {name}: {avail.reasons['llm']}")
            return None
        pipeline = await build_zero_shot_claude(settings, client)
    elif name == SYSTEM_DISTILBERT:
        if not avail.hf_classifier:
            print(f"[skip] {name}: {avail.reasons['hf_classifier']}")
            return None
        pipeline = await build_distilbert_only(settings)
    elif name == SYSTEM_HYBRID:
        if client is None or not avail.hf_classifier:
            print(f"[skip] {name}: {'; '.join(avail.reasons.values())}")
            return None
        pipeline = await build_hybrid(settings, client)
    else:
        raise ValueError(f"unknown system {name!r}")

    cases = load_cases()
    report = await run_pipeline_eval(pipeline, cases, system=name)
    print(format_report(report))
    payload: dict[str, object] = {
        "system": name,
        "label": "fallback determinista, sin LLM" if name == SYSTEM_OFFLINE else name,
        "model": settings.anthropic_model,
        "report": report.model_dump(),
        "ablation": [asdict(r) for r in ablate_confidence(report, cases, settings)],
    }
    if judge and client is not None:
        items = [
            (c.query, r.draft or "", [s.resolution_snippet for s in r.similar])
            for c, r in zip(cases, report.cases, strict=True)
        ]
        summary = await judge_batch(ResponseJudge(client), items)
        payload["judge"] = {**asdict(summary), "usage": asdict(summary.usage)}
    elif judge:
        print("[skip] judge: requiere ANTHROPIC_API_KEY")
    path = write_run(payload, name, runs_dir)
    print(f"[ok] wrote {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", default=[SYSTEM_OFFLINE], choices=ALL_SYSTEMS)
    parser.add_argument("--judge", action="store_true", help="also run the LLM-as-judge")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_MD)
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    settings = get_settings()
    for name in args.systems:
        asyncio.run(run_system(name, settings, judge=args.judge, runs_dir=args.runs_dir))
    out = update_results(args.runs_dir, args.results)
    print(f"[ok] regenerated {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
