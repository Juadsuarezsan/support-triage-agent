"""Sentiment, priority and urgency in a single structured Claude call."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from src.api.schemas import Priority, Sentiment
from src.errors import LLMOutputError
from src.llm.claude import ClaudeClient, parse_model
from src.observability import ZERO_USAGE, TokenUsage, trace_logger

SYSTEM = """You analyze a customer support ticket and return JSON only.

Output schema:
{
  "sentiment": "neg" | "neu" | "pos",
  "priority": "P0" | "P1" | "P2" | "P3",
  "urgency_score": <0.0 - 1.0>,
  "rationale": "<one sentence>"
}

Priority guidelines:
- P0: production down, money lost, customer threatens to leave or sue, security incident
- P1: blocking workflow, paying customer, time-sensitive (today/tomorrow)
- P2: degraded experience, can wait a few days
- P3: feature request, general question, satisfied tone
"""

URGENT_WORDS = (
    "urgent",
    "asap",
    "immediately",
    "production",
    "down",
    "lawsuit",
    "sue",
    "fraud",
    "stolen",
    "unauthorized",
    "hacked",
    "right now",
    "today",
)
NEGATIVE_WORDS = (
    "terrible",
    "unhappy",
    "angry",
    "worst",
    "garbage",
    "scam",
    "unacceptable",
    "disgusted",
    "furious",
    "ridiculous",
    "never again",
    "disappointed",
    "frustrated",
)
POSITIVE_WORDS = ("thanks", "thank you", "appreciate", "great", "love", "awesome", "perfect")


class PriorityAnalysis(BaseModel):
    """Validated structured output of the analyzer."""

    sentiment: Sentiment
    priority: Priority
    urgency_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str = ""


@dataclass(frozen=True)
class PriorityResult:
    """Analysis plus provenance.

    Attributes:
        analysis: Sentiment/priority/urgency/rationale.
        source: ``"llm"`` or ``"heuristic"``.
        usage: Claude tokens consumed (zero for the heuristic).
    """

    analysis: PriorityAnalysis
    source: str
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)


class PrioritySentimentAnalyzer:
    """Derive sentiment, priority and urgency from a ticket body.

    Args:
        client: Shared Claude client, or ``None`` to force the heuristic.
    """

    def __init__(self, client: ClaudeClient | None) -> None:
        self._client = client

    async def analyze(self, text: str) -> PriorityResult:
        """Analyze ``text``.

        With a client configured the method calls Claude; a schema violation
        after the client's retries is logged and answered with the heuristic so
        a single bad completion never blocks a ticket. Transport errors
        (:class:`~src.errors.LLMError`) propagate.
        """
        if self._client is None:
            return self.heuristic(text)
        result = await self._client.complete(
            system=SYSTEM, user=text[:2000], max_tokens=200, temperature=0.0
        )
        try:
            analysis = parse_model(result.text, PriorityAnalysis)
        except LLMOutputError as exc:
            trace_logger().warning("priority analyzer: malformed LLM output ({}); heuristic", exc)
            fallback = self.heuristic(text)
            return PriorityResult(
                analysis=fallback.analysis.model_copy(
                    update={"rationale": "heuristic fallback after malformed LLM output"}
                ),
                source="heuristic_after_llm_error",
                usage=result.usage,
            )
        return PriorityResult(analysis=analysis, source="llm", usage=result.usage)

    @staticmethod
    def heuristic(text: str) -> PriorityResult:
        """Keyword-based analysis used offline and as the no-LLM baseline."""
        lowered = text.lower()
        urgent = any(w in lowered for w in URGENT_WORDS)
        negative = any(w in lowered for w in NEGATIVE_WORDS)
        positive = any(w in lowered for w in POSITIVE_WORDS) and not negative
        priority: Priority = "P0" if urgent and negative else "P1" if urgent or negative else "P3"
        sentiment: Sentiment = "neg" if negative else "pos" if positive else "neu"
        urgency = 0.95 if urgent and negative else 0.75 if urgent else 0.55 if negative else 0.2
        analysis = PriorityAnalysis(
            sentiment=sentiment,
            priority=priority,
            urgency_score=urgency,
            rationale="heuristic: keyword detection",
        )
        return PriorityResult(analysis=analysis, source="heuristic")
