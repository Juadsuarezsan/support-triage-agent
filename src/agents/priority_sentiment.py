"""Priority + sentiment analyzer (single Claude call, structured output)."""
from __future__ import annotations

import json
from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import Priority, Sentiment

SYSTEM = """You analyze a customer support ticket and return JSON only.

Output schema:
{
  "sentiment": "neg" | "neu" | "pos",
  "priority": "P0" | "P1" | "P2" | "P3",
  "urgency_score": 0.0 - 1.0,
  "rationale": "one sentence"
}

Priority guidelines:
- P0: production-down, money lost, customer threatens to leave/sue, security incident
- P1: blocking workflow, paid user, time-sensitive (today/tomorrow)
- P2: degraded experience, can wait a few days
- P3: feature request, general question, satisfied tone
"""


class PrioritySentimentAnalyzer:
    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6), reraise=True)
    async def analyze(self, text: str) -> tuple[Priority, Sentiment, float, str]:
        if not self.api_key:
            return self._heuristic(text)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0, max_tokens=200, timeout=15.0)
        resp = await chat.ainvoke([SystemMessage(content=SYSTEM), HumanMessage(content=text[:2000])])
        body = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = body.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:].lstrip()
        raw = json.loads(body)
        return raw["priority"], raw["sentiment"], float(raw["urgency_score"]), raw.get("rationale", "")

    @staticmethod
    def _heuristic(text: str) -> tuple[Priority, Sentiment, float, str]:
        t = text.lower()
        urgent = any(w in t for w in ("urgent", "asap", "production", "down", "lawsuit", "sue", "fraud"))
        negative = any(w in t for w in ("terrible", "unhappy", "angry", "worst", "garbage", "scam"))
        positive = any(w in t for w in ("thanks", "appreciate", "great"))
        priority: Priority = "P0" if urgent else ("P1" if negative else "P3")
        sentiment: Sentiment = "neg" if negative else ("pos" if positive else "neu")
        urgency = 0.9 if urgent else (0.6 if negative else 0.2)
        return priority, sentiment, urgency, "heuristic: keyword detection"
