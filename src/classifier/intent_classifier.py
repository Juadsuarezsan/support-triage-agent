"""Intent classifier with two backends:

  1. HuggingFace `transformers` pipeline loaded from a fine-tuned DistilBERT+LoRA
     model (CLASSIFIER_MODEL env var). Use this in production.
  2. Zero-shot Claude with structured output. Works without GPU / without
     training. Slower and more expensive but doesn't require model hosting.

To train your own classifier, see notebooks/01_classifier_training.ipynb
(builds on the Bitext dataset, exports to HF Hub).
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Protocol

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.api.schemas import BITEXT_INTENTS, IntentScore
from src.config import get_settings


class IntentClassifier(Protocol):
    name: str
    async def classify(self, text: str) -> list[IntentScore]: ...


# ---------------------------------------------------------------------------
# DistilBERT + LoRA (production)
# ---------------------------------------------------------------------------

class HFTransformerClassifier:
    """Loads a HuggingFace text-classification pipeline on first call."""

    name = "distilbert_lora_hf"

    def __init__(self, model_id: str) -> None:
        from transformers import pipeline
        logger.info(f"Loading HF classifier from: {model_id}")
        self._pipe = pipeline("text-classification", model=model_id, top_k=None)

    async def classify(self, text: str) -> list[IntentScore]:
        raw = self._pipe(text[:512])
        out = raw[0] if isinstance(raw[0], list) else raw
        return [IntentScore(intent=r["label"], score=float(r["score"])) for r in out]


# ---------------------------------------------------------------------------
# Zero-shot Claude
# ---------------------------------------------------------------------------

ZS_SYSTEM_PROMPT = f"""You classify a customer support ticket into one of these 27 intents:

{", ".join(BITEXT_INTENTS)}

Return JSON only:
{{
  "top_intents": [
    {{ "intent": "<from the list above>", "score": 0.0-1.0 }},
    ...up to 3 entries, ordered by score descending
  ]
}}
"""


class ClaudeZeroShotClassifier:
    """Zero-shot fallback that works without a fine-tuned model."""

    name = "claude_zero_shot"

    def __init__(self, model: str, api_key: str | None) -> None:
        self.model = model
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    async def classify(self, text: str) -> list[IntentScore]:
        if not self.api_key:
            return self._heuristic(text)
        from langchain_anthropic import ChatAnthropic
        from langchain_core.messages import HumanMessage, SystemMessage
        chat = ChatAnthropic(model=self.model, api_key=self.api_key, temperature=0, max_tokens=300, timeout=15.0)
        resp = await chat.ainvoke([SystemMessage(content=ZS_SYSTEM_PROMPT),
                                    HumanMessage(content=text[:2000])])
        body = resp.content if isinstance(resp.content, str) else str(resp.content)
        body = body.strip()
        if body.startswith("```"):
            body = body.strip("`")
            if body.lower().startswith("json"):
                body = body[4:].lstrip()
        raw = json.loads(body)
        items = raw.get("top_intents", [])[:3]
        return [IntentScore(intent=str(it["intent"]), score=float(it["score"])) for it in items]

    @staticmethod
    def _heuristic(text: str) -> list[IntentScore]:
        t = text.lower()
        # Tiny keyword router used when no API key is set (CI / offline tests).
        rules: list[tuple[tuple[str, ...], str, float]] = [
            (("refund", "money back"), "get_refund", 0.7),
            (("cancel", "stop my order"), "cancel_order", 0.7),
            (("track", "where is", "delivery"), "track_order", 0.7),
            (("invoice", "receipt"), "get_invoice", 0.7),
            (("password", "reset password"), "recover_password", 0.7),
            (("human", "agent", "person"), "contact_human_agent", 0.85),
            (("change my order", "modify order"), "change_order", 0.7),
            (("create account", "sign up", "register"), "create_account", 0.7),
            (("delete account", "close my account"), "delete_account", 0.7),
            (("complaint", "unhappy", "terrible"), "complaint", 0.65),
            (("payment", "card declined", "payment issue"), "payment_issue", 0.7),
        ]
        for keywords, intent, score in rules:
            if any(k in t for k in keywords):
                return [IntentScore(intent=intent, score=score)]
        return [IntentScore(intent="general_question", score=0.4)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_classifier() -> IntentClassifier:
    s = get_settings()
    if s.use_local_classifier and s.classifier_model:
        try:
            return HFTransformerClassifier(s.classifier_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"HF classifier load failed ({exc}). Falling back to zero-shot Claude.")
    return ClaudeZeroShotClassifier(model=s.anthropic_model, api_key=s.anthropic_api_key)
