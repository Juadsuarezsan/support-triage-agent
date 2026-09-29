"""LLM-as-judge for drafted replies with an explicit, numbered rubric.

The rubric is data (:data:`RUBRIC`) rendered into the prompt, so the exact
criteria the judge applied are versioned with the code. Each dimension is
scored 1-5 with anchored descriptions and a worked example per anchor.

Running the judge requires ``ANTHROPIC_API_KEY``; without it
:func:`judge_batch` is never called and the evaluation reports the cell as
pending.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from src.llm.claude import ClaudeClient, parse_model
from src.observability import ZERO_USAGE, TokenUsage

#: Judge rubric. Keys are the scored dimensions; each has numbered criteria,
#: 1-5 anchors and one example per extreme anchor.
RUBRIC: dict[str, dict[str, Any]] = {
    "helpfulness": {
        "criteria": [
            "1. The reply addresses the specific problem in the ticket, not a generic topic.",
            "2. The reply gives the customer at least one concrete next step they can take.",
            "3. The reply uses information from the similar resolved tickets when relevant.",
        ],
        "anchors": {
            1: "Ignores the problem or answers a different question.",
            2: "Mentions the problem but offers no actionable step.",
            3: "Addresses the problem with a vague or partial next step.",
            4: "Addresses the problem with one clear, correct next step.",
            5: "Addresses the problem, gives a clear next step and anticipates the follow-up.",
        },
        "examples": {
            1: "Ticket asks about a refund; reply explains how to reset a password.",
            5: "Ticket asks about a refund; reply confirms the 5-day refund window, tells the "
            "customer where to see the status and offers a replacement if preferred.",
        },
    },
    "tone": {
        "criteria": [
            "4. The reply is polite and acknowledges the customer's frustration when present.",
            "5. The reply never blames the customer and never sounds robotic or dismissive.",
            "6. The reply is concise (under ~100 words) and free of jargon.",
        ],
        "anchors": {
            1: "Rude, blaming or dismissive.",
            2: "Neutral but cold; ignores visible frustration.",
            3: "Polite but formulaic.",
            4: "Polite, warm and appropriately brief.",
            5: "Polite, empathetic, brief and matched to the customer's emotional state.",
        },
        "examples": {
            1: "'You should have read the policy before ordering.'",
            5: "'I'm sorry the package is late - that's frustrating. Here is what I've done...'",
        },
    },
    "correctness": {
        "criteria": [
            "7. Every factual claim (timelines, policies, fees) is supported by the similar "
            "resolved tickets or is clearly hedged.",
            "8. The reply does not promise actions the system cannot verify (no invented refunds, "
            "credits or shipments).",
            "9. When evidence is insufficient the reply says a human will follow up instead of "
            "guessing.",
        ],
        "anchors": {
            1: "Contains invented facts or promises that contradict the evidence.",
            2: "Contains at least one unsupported claim presented as fact.",
            3: "Mostly grounded, with minor unhedged extrapolation.",
            4: "All claims grounded or hedged; no invented commitments.",
            5: "All claims grounded, hedged where needed, and the limits of the evidence stated.",
        },
        "examples": {
            1: "Promises a full refund today although no similar ticket mentions refunds.",
            5: "States the 30-day policy from the evidence and notes that opened items are "
            "reviewed case by case.",
        },
    },
}

MIN_SCORE = 1
MAX_SCORE = 5


def render_rubric() -> str:
    """Render :data:`RUBRIC` as the text embedded in the judge prompt."""
    parts: list[str] = []
    for dimension, spec in RUBRIC.items():
        parts.append(f"## {dimension.upper()}")
        parts.extend(spec["criteria"])
        parts.append("Anchors:")
        for score, text in spec["anchors"].items():
            parts.append(f"  {score}: {text}")
        parts.append("Examples:")
        for score, text in spec["examples"].items():
            parts.append(f"  score {score}: {text}")
        parts.append("")
    return "\n".join(parts)


JUDGE_SYSTEM = f"""You are a strict evaluator of customer-support replies.

Score the DRAFT reply to the TICKET on three dimensions using ONLY the rubric
below. The SIMILAR section is the evidence the drafter had; treat any fact not
in it as unsupported.

{render_rubric()}
Return JSON only:
{{"helpfulness": <1-5>, "tone": <1-5>, "correctness": <1-5>, "justification": "<one sentence>"}}
"""


class JudgeScore(BaseModel):
    """Scores of one drafted reply."""

    helpfulness: int = Field(..., ge=MIN_SCORE, le=MAX_SCORE)
    tone: int = Field(..., ge=MIN_SCORE, le=MAX_SCORE)
    correctness: int = Field(..., ge=MIN_SCORE, le=MAX_SCORE)
    justification: str = ""

    @property
    def mean(self) -> float:
        """Unweighted mean of the three dimensions."""
        return round((self.helpfulness + self.tone + self.correctness) / 3, 4)


@dataclass(frozen=True)
class JudgeResult:
    """Score plus the tokens the judge consumed."""

    score: JudgeScore
    usage: TokenUsage


@dataclass(frozen=True)
class JudgeSummary:
    """Aggregate over a batch of judged replies."""

    n: int
    helpfulness: float
    tone: float
    correctness: float
    mean: float
    usage: TokenUsage


class ResponseJudge:
    """Judge drafted replies with Claude.

    Args:
        client: Shared Claude client (the judge should use the same pinned model
            as the system so results are comparable across runs).
    """

    def __init__(self, client: ClaudeClient) -> None:
        self._client = client

    @staticmethod
    def build_user_prompt(ticket: str, draft: str, evidence: Sequence[str]) -> str:
        """Compose the judged material as tagged sections."""
        similar = "\n".join(f"- {e}" for e in evidence) or "(none)"
        return (
            f"<ticket>{ticket}</ticket>\n<similar>\n{similar}\n</similar>\n<draft>{draft}</draft>"
        )

    async def judge(self, ticket: str, draft: str, evidence: Sequence[str]) -> JudgeResult:
        """Score one reply.

        Raises:
            LLMError: When the API is unreachable.
            LLMOutputError: When the judge output violates :class:`JudgeScore`.
        """
        result = await self._client.complete(
            system=JUDGE_SYSTEM,
            user=self.build_user_prompt(ticket, draft, evidence),
            max_tokens=200,
            temperature=0.0,
        )
        return JudgeResult(score=parse_model(result.text, JudgeScore), usage=result.usage)


async def judge_batch(
    judge: ResponseJudge,
    items: Sequence[tuple[str, str, Sequence[str]]],
    *,
    concurrency: int = 4,
) -> JudgeSummary:
    """Judge ``items`` (``(ticket, draft, evidence)``) with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one(item: tuple[str, str, Sequence[str]]) -> JudgeResult:
        async with semaphore:
            return await judge.judge(*item)

    results = await asyncio.gather(*(one(i) for i in items))
    n = len(results)
    usage = ZERO_USAGE
    for r in results:
        usage = usage + r.usage
    if n == 0:
        return JudgeSummary(0, 0.0, 0.0, 0.0, 0.0, usage)
    return JudgeSummary(
        n=n,
        helpfulness=round(sum(r.score.helpfulness for r in results) / n, 4),
        tone=round(sum(r.score.tone for r in results) / n, 4),
        correctness=round(sum(r.score.correctness for r in results) / n, 4),
        mean=round(sum(r.score.mean for r in results) / n, 4),
        usage=usage,
    )
