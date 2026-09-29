"""Build the demo notebooks with real outputs, without Jupyter.

Each notebook is assembled from Markdown and code cells; every code cell is
executed with ``exec`` in a shared namespace (offline pipeline, no API key)
and its stdout is stored as the cell output. The result opens in Jupyter and
can be re-run there; it needs only the ``dev`` extra.

Usage::

    python scripts/build_notebook.py                 # writes notebooks/*.ipynb
    python scripts/build_notebook.py --out /tmp/nb   # elsewhere (tests)
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path
from textwrap import dedent
from typing import Any

import nbformat

from src.observability import configure_logging

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"

PREAMBLE = dedent(
    """
    import asyncio, json, os
    from pathlib import Path
    os.environ.pop("ANTHROPIC_API_KEY", None)          # offline: deterministic fallback
    os.environ["USE_LOCAL_CLASSIFIER"] = "false"
    from src.config import Settings
    from src.api.schemas import TicketIn
    from src.eval.runner import build_offline_pipeline, run_pipeline_eval, load_cases, format_report
    settings = Settings(anthropic_api_key=None, use_local_classifier=False)
    pipeline = asyncio.run(build_offline_pipeline(settings))
    print("classifier:", pipeline.classifier.name, "| llm_enabled:", pipeline.llm_enabled)
    """
).strip()

DEMO_INTRO = (
    "# Customer Support Triage Agent — demo end-to-end\n\n"
    "Runs the full LangGraph flow on a few tickets. Without `ANTHROPIC_API_KEY` and the "
    "`ml` extra every component uses its **deterministic fallback** (keyword classifier, "
    "heuristic priority, template reply, hashed encoder); the outputs below are labelled "
    "accordingly and are not the production system's quality."
)

DEMO_TRIAGE = dedent(
    """
    tickets = [
        ("t-1", "Where is my order #66721? It has been a week."),
        ("t-2", "This is unacceptable, I want to file a complaint about the courier."),
        ("t-3", "What is your refund policy on opened items?"),
    ]
    for tid, body in tickets:
        out = asyncio.run(pipeline.triage(TicketIn(ticket_id=tid, body=body)))
        print(
            f"{tid}: intent={out.intent} ({out.intent_confidence:.2f}) priority={out.priority} "
            f"sentiment={out.sentiment} -> {out.decision.decision} "
            f"conf={out.decision.confidence:.2f}"
        )
        print("   draft:", out.draft_response[:110], "...")
        print("   trace:", out.trace_id, "| tokens:", out.input_tokens, out.output_tokens,
              "| cost:", out.cost_usd)
    """
).strip()

DEMO_AUDIT = dedent(
    """
    for row in asyncio.run(pipeline.audit.recent(5)):
        print(row.ticket_id, row.decision, f"{row.confidence:.2f}", row.llm_used,
              row.latency_ms, "ms")
    """
).strip()

EVAL_INTRO = (
    "# 02 — Evaluation pipeline\n\nRuns `run_pipeline_eval` on the 108 hand-written synthetic "
    "tickets (`data/eval/queries.jsonl`). This is the **offline fallback**; the Zero-shot Claude "
    "/ DistilBERT-only / Hybrid rows need an API key and the `ml` extra "
    "(`python -m eval.run --systems ...`)."
)

EVAL_RUN = dedent(
    """
    cases = load_cases()
    report = asyncio.run(run_pipeline_eval(pipeline, cases))
    print("\\n".join(format_report(report).splitlines()[:14]))
    """
).strip()

EVAL_PER_INTENT = dedent(
    """
    ranked = sorted(report.per_intent.items(), key=lambda kv: kv[1]["intent_accuracy"])
    for intent, b in ranked:
        if b["intent_accuracy"] < 1:
            print(f"{intent:<26} n={int(b['n'])} intent_acc={b['intent_accuracy']:.2f} "
                  f"decision_acc={b['decision_accuracy']:.2f}")
    """
).strip()

BASE_INTRO = (
    "# 03 — Baselines comparison\n\nThe mandatory table compares Zero-shot Claude, "
    "DistilBERT-only and the Hybrid system. `check_availability` reports which ones can run "
    "here; the ablation of the confidence score runs on the recorded offline signals."
)

BASE_AVAIL = dedent(
    """
    from src.eval.baselines import check_availability, ablate_confidence
    avail = check_availability(settings)
    print("llm:", avail.llm, "| hf_classifier:", avail.hf_classifier)
    for k, v in avail.reasons.items():
        print(f"  {k}: {v}")
    """
).strip()

BASE_ABLATION = dedent(
    """
    cases = load_cases()
    report = asyncio.run(run_pipeline_eval(pipeline, cases))
    print(f"{'variant':<22} agreement   auto  false_esc  false_auto")
    for row in ablate_confidence(report, cases, settings):
        print(f"{row.variant:<22} {row.decision_agreement:>8.1%} {row.auto_resolve_rate:>6.1%} "
              f"{row.false_escalation_rate:>9.1%} {row.false_auto_resolve_rate:>10.1%}")
    """
).strip()

BASE_OUTRO = (
    "With the hashed encoder the similarity term is noise: removing it raises decision "
    "agreement and lowers false escalations. The thresholds are calibrated for the semantic "
    "encoder (`all-mpnet-base-v2`), which is why the fallback is never reported as the system."
)

NOTEBOOKS_SPEC: dict[str, list[tuple[str, str]]] = {
    "demo.ipynb": [
        ("markdown", DEMO_INTRO),
        ("code", PREAMBLE),
        ("code", DEMO_TRIAGE),
        (
            "markdown",
            "## Audit log\n\nEvery triage writes an `AuditRecord`; the API exposes "
            "them at `GET /api/audit/recent` for the Kanban.",
        ),
        ("code", DEMO_AUDIT),
    ],
    "02_eval_pipeline.ipynb": [
        ("markdown", EVAL_INTRO),
        ("code", PREAMBLE),
        ("code", EVAL_RUN),
        (
            "markdown",
            "## Per-intent accuracy\n\nWhere the keyword fallback misses (synonyms "
            "and articles it does not list); see `docs/error_analysis.md`.",
        ),
        ("code", EVAL_PER_INTENT),
    ],
    "03_baselines_comparison.ipynb": [
        ("markdown", BASE_INTRO),
        ("code", PREAMBLE),
        ("code", BASE_AVAIL),
        ("code", BASE_ABLATION),
        ("markdown", BASE_OUTRO),
    ],
}


def build(cells: list[tuple[str, str]]) -> nbformat.NotebookNode:
    """Assemble a notebook and execute its code cells, capturing stdout."""
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    namespace: dict[str, Any] = {}
    count = 0
    for kind, source in cells:
        if kind == "markdown":
            nb.cells.append(nbformat.v4.new_markdown_cell(source))
            continue
        cell = nbformat.v4.new_code_cell(source)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(source, "<cell>", "exec"), namespace)  # noqa: S102 - repo-owned code
        count += 1
        cell.execution_count = count
        text = buf.getvalue()
        if text:
            cell.outputs = [nbformat.v4.new_output("stream", name="stdout", text=text)]
        nb.cells.append(cell)
    return nb


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=NOTEBOOKS)
    args = parser.parse_args(argv)
    configure_logging("WARNING")
    args.out.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS_SPEC.items():
        nbformat.write(build(cells), str(args.out / name))
        print(f"wrote {args.out / name}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
