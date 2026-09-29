"""Solution drafter: composes a reply grounded in similar resolved tickets."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.api.schemas import SimilarTicket
from src.llm.claude import ClaudeClient
from src.observability import ZERO_USAGE, TokenUsage

SYSTEM = """You are a customer support response drafter.

Given an incoming ticket and up to 5 similar tickets that were already resolved
(with their resolutions), draft a reply that:
- Acknowledges the customer's concern in a polite, professional tone
- Is grounded in the resolution patterns of the similar past tickets
- Stays under 100 words
- Ends with one concrete next step

Never invent facts. If the similar tickets do not give you enough information,
say explicitly that a human teammate will follow up.
"""


@dataclass(frozen=True)
class DraftResult:
    """A drafted reply plus provenance.

    Attributes:
        text: The reply.
        source: ``"llm"`` or ``"template"``.
        usage: Claude tokens consumed (zero for the template).
    """

    text: str
    source: str
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)


class Drafter:
    """Draft a customer-facing reply.

    Args:
        client: Shared Claude client, or ``None`` to use the template.
    """

    def __init__(self, client: ClaudeClient | None) -> None:
        self._client = client

    async def draft(self, ticket_body: str, similar: list[SimilarTicket]) -> DraftResult:
        """Draft a reply for ``ticket_body`` using ``similar`` as evidence."""
        if self._client is None:
            return self.template(similar)
        result = await self._client.complete(
            system=SYSTEM,
            user=self._user_prompt(ticket_body, similar),
            max_tokens=400,
            temperature=0.4,
        )
        text = result.text.strip() or self.template(similar).text
        return DraftResult(text=text, source="llm", usage=result.usage)

    @staticmethod
    def _user_prompt(ticket_body: str, similar: list[SimilarTicket]) -> str:
        context = (
            "\n".join(
                f"- [{t.intent} · sim {t.similarity:.2f}] Q: {t.body_snippet}\n"
                f"  A: {t.resolution_snippet}"
                for t in similar
            )
            or "(no similar tickets found)"
        )
        return f"<ticket>{ticket_body[:2000]}</ticket>\n\n<similar>\n{context}\n</similar>"

    @staticmethod
    def template(similar: list[SimilarTicket]) -> DraftResult:
        """Deterministic reply used offline; reuses the top resolution verbatim."""
        if not similar or not similar[0].resolution_snippet:
            text = (
                "Thanks for reaching out. A teammate will follow up shortly with details "
                "specific to your case."
            )
        else:
            top = similar[0]
            text = (
                f"Thanks for getting in touch. Based on similar cases ({top.intent}), the usual "
                f"resolution is: {top.resolution_snippet} Reply to this thread if that does not "
                "fit your situation."
            )
        return DraftResult(text=text, source="template")
