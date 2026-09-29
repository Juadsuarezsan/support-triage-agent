"""Pre-bake demo tickets by running the repository's own triage pipeline.

The static Kanban in ``demo/index.html`` reads ``demo/predictions.json`` so a
visitor sees real outputs without a backend. Without ``ANTHROPIC_API_KEY``
the run uses the deterministic fallback (keyword classifier, heuristic
priority, template reply, hashed encoder) and the JSON says so explicitly in
``mode``; with a key the same script produces LLM-backed outputs.

Usage::

    python scripts/bake_demo_predictions.py            # -> demo/predictions.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from src.api.main import API_VERSION
from src.api.schemas import Channel, TicketIn
from src.config import get_settings
from src.eval.runner import build_offline_pipeline
from src.observability import configure_logging

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "demo" / "predictions.json"

DEMO_TICKETS: list[dict[str, str]] = [
    {
        "ticket_id": "d-101",
        "channel": "email",
        "body": "I'd like a refund for the order I placed yesterday, it arrived broken.",
    },
    {
        "ticket_id": "d-102",
        "channel": "chat",
        "body": "Where is my package? It was supposed to arrive 3 days ago.",
    },
    {
        "ticket_id": "d-103",
        "channel": "slack",
        "body": "This is unacceptable, three weeks without an answer. I want to file a complaint.",
    },
    {
        "ticket_id": "d-104",
        "channel": "chat",
        "body": "Can you help me reset my password? The reset email never came.",
    },
    {
        "ticket_id": "d-105",
        "channel": "email",
        "body": "I want to talk to a real person, the bot keeps misunderstanding me.",
    },
    {
        "ticket_id": "d-106",
        "channel": "twitter",
        "body": "What is your refund policy on opened items?",
    },
    {
        "ticket_id": "d-107",
        "channel": "email",
        "body": "Please delete my account and erase all my personal data per GDPR.",
    },
    {
        "ticket_id": "d-108",
        "channel": "chat",
        "body": "I was charged twice for the same purchase, please look into it.",
    },
]


async def bake() -> dict[str, object]:
    """Run every demo ticket through the pipeline and collect the outputs."""
    settings = get_settings()
    pipeline = await build_offline_pipeline(settings if settings.llm_enabled else None)
    outputs = []
    for t in DEMO_TICKETS:
        ticket = TicketIn(
            ticket_id=t["ticket_id"], channel=cast(Channel, t["channel"]), body=t["body"]
        )
        out = await pipeline.triage(ticket)
        outputs.append({"ticket": t, "result": out.model_dump()})
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "api_version": API_VERSION,
        "mode": "llm" if pipeline.llm_enabled else "fallback determinista, sin LLM",
        "classifier_backend": pipeline.classifier.name,
        "model": settings.anthropic_model if pipeline.llm_enabled else None,
        "note": (
            "Outputs produced by src/ with the deterministic fallback (keyword classifier, "
            "heuristic priority, template reply, hashed encoder). They are NOT the production "
            "system's quality; run with ANTHROPIC_API_KEY and the ml extra for real outputs."
        ),
        "tickets": outputs,
    }


def main() -> int:
    """CLI entry point."""
    configure_logging("WARNING")
    payload = asyncio.run(bake())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(DEMO_TICKETS)} tickets, mode={payload['mode']})")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
