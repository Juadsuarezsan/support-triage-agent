"""End-to-end evaluation of a :class:`~src.agents.orchestrator.TriagePipeline`.

Runs every case of ``data/eval/queries.jsonl`` through the graph and reports
classification quality (macro-F1 over the 27 intents, top-1/top-3), routing
quality (decision agreement, auto-resolve / false-escalation / false
auto-resolve rates), priority band match, latency percentiles and cost.

CLI (offline, deterministic fallback without an API key)::

    python -m src.eval.runner            # human-readable summary
    python -m src.eval.runner --json     # full report as JSON
    python -m src.eval.runner --fast     # kept for backwards compatibility
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.agents.orchestrator import TriagePipeline, build_pipeline
from src.api.schemas import BITEXT_INTENTS, Decision, IntentScore, SimilarTicket, TicketIn
from src.classifier.intent_classifier import IntentClassifier
from src.config import Settings
from src.eval.metrics import classification_metrics, percentile, triage_rates
from src.llm.claude import ClaudeClient
from src.observability import trace_logger
from src.persistence.audit import InMemoryAuditStore
from src.retrieval.encoders import build_encoder
from src.retrieval.similar_tickets import InMemoryTicketStore, TicketStore, load_jsonl

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "eval"
QUERIES_PATH = DATA_DIR / "queries.jsonl"
SEEDS_PATH = DATA_DIR / "seed_tickets.jsonl"


class EvalCase(BaseModel):
    """One hand-labelled ticket of the evaluation set."""

    id: str
    query: str
    expected_intent: str
    expected_decision: Decision
    expected_priority_range: list[str] = Field(default_factory=list)
    synthetic: bool = True


class CaseResult(BaseModel):
    """Outcome of one case, with the raw signals needed for ablations."""

    id: str
    expected_intent: str
    got_intent: str
    intent_ok: bool
    top3_ok: bool
    expected_decision: Decision
    got_decision: Decision
    decision_ok: bool
    priority: str
    priority_ok: bool
    confidence: float
    latency_ms: int
    cost_usd: float
    intents: list[IntentScore] = Field(default_factory=list)
    similar: list[SimilarTicket] = Field(default_factory=list)
    sentiment: str = "neu"
    urgency_score: float = 0.0
    draft: str | None = None


class EvalReport(BaseModel):
    """Aggregate metrics of one evaluation run."""

    system: str
    n: int
    llm_used: bool
    classifier_backend: str
    encoder: str
    model: str
    macro_f1: float
    intent_top1_accuracy: float
    intent_top3_accuracy: float
    decision_agreement: float
    auto_resolve_rate: float
    false_escalation_rate: float
    false_auto_resolve_rate: float
    priority_band_match: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    total_cost_usd: float
    cost_per_ticket_usd: float
    per_class_f1: dict[str, float] = Field(default_factory=dict)
    per_intent: dict[str, dict[str, float]] = Field(default_factory=dict)
    cases: list[CaseResult] = Field(default_factory=list)


def load_cases(path: Path = QUERIES_PATH) -> list[EvalCase]:
    """Load and validate the evaluation set."""
    return [EvalCase.model_validate(row) for row in load_jsonl(path)]


async def build_offline_pipeline(
    settings: Settings | None = None,
    *,
    classifier: IntentClassifier | None = None,
    client: ClaudeClient | None = None,
    seeds_path: Path = SEEDS_PATH,
    store: TicketStore | None = None,
) -> TriagePipeline:
    """Build a pipeline over in-memory stores seeded from ``seeds_path``.

    Args:
        settings: Configuration; defaults to a keyless, deterministic setup.
        classifier: Optional classifier override.
        client: Optional Claude client override (tests inject mocks).
        seeds_path: Resolved tickets to index.
        store: Optional pre-built ticket store (skips seeding).
    """
    cfg = settings or Settings(
        anthropic_api_key=None,
        use_local_classifier=False,
        encoder_backend="deterministic",
    )
    if store is None:
        store = InMemoryTicketStore(build_encoder(cfg.encoder_backend))
        if seeds_path.exists():
            await store.index(load_jsonl(seeds_path))
    return build_pipeline(
        cfg, store=store, audit=InMemoryAuditStore(), classifier=classifier, client=client
    )


async def run_pipeline_eval(
    pipeline: TriagePipeline,
    cases: list[EvalCase] | None = None,
    *,
    system: str = "offline_fallback",
) -> EvalReport:
    """Evaluate ``pipeline`` on ``cases`` (default: the full eval set)."""
    cases = cases if cases is not None else load_cases()
    results: list[CaseResult] = []
    for case in cases:
        out = await pipeline.triage(TicketIn(ticket_id=case.id, body=case.query))
        top3 = [s.intent for s in out.top_intents[:3]]
        results.append(
            CaseResult(
                id=case.id,
                expected_intent=case.expected_intent,
                got_intent=out.intent,
                intent_ok=out.intent == case.expected_intent,
                top3_ok=case.expected_intent in top3,
                expected_decision=case.expected_decision,
                got_decision=out.decision.decision,
                decision_ok=out.decision.decision == case.expected_decision,
                priority=out.priority,
                priority_ok=out.priority in case.expected_priority_range,
                confidence=out.decision.confidence,
                latency_ms=out.latency_ms,
                cost_usd=out.cost_usd,
                intents=out.top_intents,
                similar=out.similar_resolved,
                sentiment=out.sentiment,
                urgency_score=out.urgency_score,
                draft=out.draft_response,
            )
        )
    encoder_name = getattr(pipeline.store, "encoder", None)
    return summarize(
        results,
        system=system,
        llm_used=any(r.cost_usd > 0 for r in results) or pipeline.llm_enabled,
        classifier_backend=pipeline.classifier.name,
        encoder=getattr(encoder_name, "name", "unknown"),
        model=pipeline.settings.anthropic_model,
    )


def summarize(
    results: list[CaseResult],
    *,
    system: str,
    llm_used: bool,
    classifier_backend: str,
    encoder: str,
    model: str,
) -> EvalReport:
    """Aggregate :class:`CaseResult` rows into an :class:`EvalReport`."""
    n = len(results)
    cls = classification_metrics(
        [r.expected_intent for r in results], [r.got_intent for r in results], BITEXT_INTENTS
    )
    rates = triage_rates([r.expected_decision for r in results], [r.got_decision for r in results])
    latencies = [float(r.latency_ms) for r in results]
    per_intent: dict[str, dict[str, float]] = {}
    for r in results:
        bucket = per_intent.setdefault(
            r.expected_intent, {"n": 0.0, "intent_accuracy": 0.0, "decision_accuracy": 0.0}
        )
        bucket["n"] += 1
        bucket["intent_accuracy"] += float(r.intent_ok)
        bucket["decision_accuracy"] += float(r.decision_ok)
    for bucket in per_intent.values():
        bucket["intent_accuracy"] = round(bucket["intent_accuracy"] / bucket["n"], 4)
        bucket["decision_accuracy"] = round(bucket["decision_accuracy"] / bucket["n"], 4)
    total_cost = round(sum(r.cost_usd for r in results), 6)
    return EvalReport(
        system=system,
        n=n,
        llm_used=llm_used,
        classifier_backend=classifier_backend,
        encoder=encoder,
        model=model,
        macro_f1=cls.macro_f1,
        intent_top1_accuracy=cls.accuracy,
        intent_top3_accuracy=round(sum(r.top3_ok for r in results) / n, 6) if n else 0.0,
        decision_agreement=rates.decision_agreement,
        auto_resolve_rate=rates.auto_resolve_rate,
        false_escalation_rate=rates.false_escalation_rate,
        false_auto_resolve_rate=rates.false_auto_resolve_rate,
        priority_band_match=round(sum(r.priority_ok for r in results) / n, 6) if n else 0.0,
        latency_p50_ms=round(percentile(latencies, 50), 3),
        latency_p95_ms=round(percentile(latencies, 95), 3),
        latency_mean_ms=round(statistics.mean(latencies), 3) if latencies else 0.0,
        total_cost_usd=total_cost,
        cost_per_ticket_usd=round(total_cost / n, 6) if n else 0.0,
        per_class_f1=cls.per_class_f1,
        per_intent=per_intent,
        cases=results,
    )


async def run_eval(use_fast_encoder: bool = True) -> dict[str, Any]:
    """Backwards-compatible entry point: offline evaluation as a plain dict."""
    if not use_fast_encoder:
        trace_logger().warning("HF encoder requested; use ENCODER_BACKEND=hf with the ml extra")
    pipeline = await build_offline_pipeline()
    return (await run_pipeline_eval(pipeline)).model_dump()


def format_report(r: EvalReport) -> str:
    """Render a compact human-readable summary."""
    sep = "=" * 72
    lines = [
        sep,
        f"SUPPORT TRIAGE EVAL  system={r.system}  n={r.n}  llm_used={r.llm_used}",
        f"classifier={r.classifier_backend}  encoder={r.encoder}",
        sep,
        f"Macro-F1 (27 intents):   {r.macro_f1:.3f}",
        f"Intent top-1 / top-3:    {r.intent_top1_accuracy:.1%} / {r.intent_top3_accuracy:.1%}",
        f"Decision agreement:      {r.decision_agreement:.1%}",
        f"Auto-resolve rate:       {r.auto_resolve_rate:.1%}",
        f"False escalation rate:   {r.false_escalation_rate:.1%}",
        f"False auto-resolve rate: {r.false_auto_resolve_rate:.1%}",
        f"Priority band match:     {r.priority_band_match:.1%}",
        f"Latency p50 / p95 (ms):  {r.latency_p50_ms:.0f} / {r.latency_p95_ms:.0f}",
        f"Cost per ticket (USD):   {r.cost_per_ticket_usd:.6f}",
        "",
    ]
    for c in r.cases:
        flag = "OK" if c.intent_ok and c.decision_ok else ".."
        lines.append(
            f"  [{flag}] {c.id:<6} expected={c.expected_intent:<26} got={c.got_intent:<26} "
            f"decision={c.got_decision:<12} conf={c.confidence:.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; exits 1 when top-1 accuracy drops below 30 %."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the full report as JSON")
    parser.add_argument("--fast", action="store_true", help="(kept) deterministic encoder")
    args = parser.parse_args(argv)

    async def _run() -> EvalReport:
        return await run_pipeline_eval(await build_offline_pipeline())

    report = asyncio.run(_run())
    if args.json:
        print(json.dumps(report.model_dump(), indent=2, default=str))
    else:
        print(format_report(report))
    return 1 if report.intent_top1_accuracy < 0.30 else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
