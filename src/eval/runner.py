"""End-to-end triage eval.

Loads seed_tickets.jsonl into an in-memory similar-ticket store, then runs
queries.jsonl through the full LangGraph triage flow. Reports:
  - Intent accuracy (top-1) and top-3 accuracy
  - Triage decision agreement with expected band
  - Priority agreement with expected priority range
  - Latency p50 / p95
  - Per-intent breakdown for debugging

CLI:
    python -m src.eval.runner
    python -m src.eval.runner --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.agents.orchestrator import build_graph, triage_ticket
from src.retrieval.similar_tickets import DeterministicEncoder, InMemoryTicketStore

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "eval"


def load(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


async def run_eval(use_fast_encoder: bool = False) -> dict[str, Any]:
    seeds = load(DATA_DIR / "seed_tickets.jsonl")
    queries = load(DATA_DIR / "queries.jsonl")

    encoder = DeterministicEncoder() if use_fast_encoder else None
    store = InMemoryTicketStore(encoder=encoder)
    await store.index(seeds)
    logger.info(f"Indexed {store.count()} seed tickets")

    graph = build_graph(store)

    latencies: list[float] = []
    intent_hits = 0
    top3_hits = 0
    decision_hits = 0
    priority_hits = 0
    per_intent: dict[str, dict] = {}

    cases: list[dict] = []
    for q in queries:
        out = await triage_ticket(graph, ticket_id=q["id"], body=q["query"])
        latencies.append(out.latency_ms)

        intent_ok = out.intent == q["expected_intent"]
        top3 = [s.intent for s in out.top_intents[:3]]
        top3_ok = q["expected_intent"] in top3
        decision_ok = out.decision.decision == q["expected_decision"]
        priority_ok = out.priority in q.get("expected_priority_range", [])

        intent_hits += int(intent_ok)
        top3_hits += int(top3_ok)
        decision_hits += int(decision_ok)
        priority_hits += int(priority_ok)

        bucket = per_intent.setdefault(q["expected_intent"], {"n": 0, "intent_hit": 0, "decision_hit": 0})
        bucket["n"] += 1
        bucket["intent_hit"] += int(intent_ok)
        bucket["decision_hit"] += int(decision_ok)

        cases.append({
            "id": q["id"], "expected_intent": q["expected_intent"], "got_intent": out.intent,
            "intent_ok": intent_ok, "top3_ok": top3_ok,
            "expected_decision": q["expected_decision"], "got_decision": out.decision.decision,
            "decision_ok": decision_ok,
            "priority": out.priority, "priority_ok": priority_ok,
            "confidence": out.decision.confidence,
        })

    n = len(queries)
    return {
        "n": n,
        "intent_top1_accuracy":  intent_hits  / n if n else 0.0,
        "intent_top3_accuracy":  top3_hits    / n if n else 0.0,
        "decision_agreement":    decision_hits/ n if n else 0.0,
        "priority_band_match":   priority_hits/ n if n else 0.0,
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_p95_ms": _percentile(latencies, 95),
        "latency_mean_ms": statistics.mean(latencies) if latencies else 0.0,
        "per_intent": {
            k: {"n": v["n"],
                "intent_accuracy": v["intent_hit"] / v["n"],
                "decision_accuracy": v["decision_hit"] / v["n"]}
            for k, v in per_intent.items()
        },
        "cases": cases,
    }


def _percentile(xs: list[float], p: int) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _format(r: dict) -> str:
    sep = "=" * 70
    lines = [sep, f"SUPPORT TRIAGE EVAL  n={r['n']}", sep,
             f"Intent top-1 accuracy:  {r['intent_top1_accuracy']:.1%}",
             f"Intent top-3 accuracy:  {r['intent_top3_accuracy']:.1%}",
             f"Decision agreement:     {r['decision_agreement']:.1%}",
             f"Priority band match:    {r['priority_band_match']:.1%}",
             f"Latency p50 / p95 (ms): {r['latency_p50_ms']:.0f} / {r['latency_p95_ms']:.0f}",
             ""]
    for c in r["cases"]:
        flag = "OK" if c["intent_ok"] and c["decision_ok"] else ".."
        lines.append(
            f"  [{flag}] {c['id']}  expected={c['expected_intent']:<22} got={c['got_intent']:<22}  "
            f"decision={c['got_decision']:<12} conf={c['confidence']:.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--fast", action="store_true",
                        help="Use deterministic encoder (no model download) — faster in CI")
    args = parser.parse_args()
    report = asyncio.run(run_eval(use_fast_encoder=args.fast))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_format(report))
    if report["intent_top1_accuracy"] < 0.30:
        sys.exit(1)


if __name__ == "__main__":
    main()
