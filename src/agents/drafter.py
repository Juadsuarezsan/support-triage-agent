"""Solution drafter — composes a response based on similar resolved tickets."""
from __future__ import annotations

from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import SimilarTicket

SYSTEM = """You are a customer support response drafter.

Given an incoming ticket and 3-5 examples of similar tickets that were already
resolved (with their resolutions), draft a response that:
- Acknowledges the customer's concern in a polite, professional tone
- Is grounded in the resolution patterns from the similar past tickets
- Stays under 100 words
- Ends with one concrete next step

Never invent facts. If similar tickets don't give you enough information, say
explicitly that the case needs a human teammate.
"""


class Drafter:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5), reraise=True)
    async def draft(self, ticket_body: str, similar: list[SimilarTicket]) -> str:
        if not self.api_key:
            return self._template(ticket_body, similar)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0.4, max_tokens=400, timeout=15.0)
        ctx = "\n".join(
            f"- [{t.intent} · sim {t.similarity:.2f}] Q: {t.body_snippet}\n  A: {t.resolution_snippet}"
            for t in similar
        ) or "(no similar tickets found)"
        user = f"<ticket>{ticket_body[:2000]}</ticket>\n\n<similar>\n{ctx}\n</similar>"
        resp = await chat.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=user)])
        return resp.content if isinstance(resp.content, str) else str(resp.content)

    @staticmethod
    def _template(ticket_body: str, similar: list[SimilarTicket]) -> str:
        if not similar:
            return "Thanks for reaching out. A teammate will follow up shortly with details specific to your case."
        top = similar[0]
        return (
            f"Thanks for getting in touch. Based on similar cases ({top.intent}), the typical resolution is: "
            f"{top.resolution_snippet}. Reply to this thread if that doesn't fit your situation."
        )
