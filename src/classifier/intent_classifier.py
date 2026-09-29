"""Intent classification with three interchangeable backends.

1. :class:`HFTransformerClassifier` - the fine-tuned DistilBERT+LoRA model
   loaded through ``transformers`` (installed with the ``ml`` extra). Imports
   are deferred so the API boots without torch.
2. :class:`ClaudeZeroShotClassifier` - Claude prompted with the 27 labels;
   works without a trained model when an API key is present.
3. :class:`KeywordHeuristicClassifier` - deterministic keyword rules covering
   all 27 intents. It is the offline fallback and the "no-AI" baseline in the
   evaluation; it is never presented as the production system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol

from pydantic import BaseModel, Field

from src.api.schemas import BITEXT_INTENTS, UNKNOWN_INTENT, IntentScore
from src.config import Settings, get_settings
from src.llm.claude import ClaudeClient, parse_model
from src.observability import ZERO_USAGE, TokenUsage, trace_logger


@dataclass(frozen=True)
class ClassificationResult:
    """Ranked intents plus the token usage that produced them.

    Attributes:
        intents: Intents ordered by descending score (at least one entry).
        usage: Claude tokens consumed (zero for local backends).
        backend: Name of the classifier that produced the result.
    """

    intents: list[IntentScore]
    backend: str
    usage: TokenUsage = field(default_factory=lambda: ZERO_USAGE)

    @property
    def top(self) -> IntentScore:
        """The highest-scoring intent."""
        return self.intents[0]


class IntentClassifier(Protocol):
    """Anything that maps ticket text to ranked intents."""

    name: str

    async def classify(self, text: str) -> ClassificationResult:
        """Classify ``text`` into ranked Bitext intents."""
        ...


# ---------------------------------------------------------------------------
# DistilBERT + LoRA (production, requires the ``ml`` extra)
# ---------------------------------------------------------------------------


class HFTransformerClassifier:
    """Fine-tuned DistilBERT+LoRA served through a ``transformers`` pipeline.

    The pipeline is built lazily on the first call so importing this module
    never requires torch. Loading failures surface as ``ImportError`` or
    ``OSError`` to the factory, which then picks the next backend.

    Args:
        model_id: HF Hub ID or local path with the merged model or adapter.
        pipeline_factory: Injectable replacement for ``transformers.pipeline``
            (tests pass a fake).
    """

    name = "distilbert_lora_hf"

    def __init__(self, model_id: str, pipeline_factory: Any | None = None) -> None:
        self.model_id = model_id
        self._pipeline_factory = pipeline_factory
        self._pipe: Any | None = None

    def _load(self) -> Any:
        if self._pipe is None:
            factory = self._pipeline_factory
            if factory is None:
                from transformers import pipeline as hf_pipeline  # heavy, deferred

                factory = hf_pipeline
            trace_logger().info("loading HF classifier model_id={}", self.model_id)
            self._pipe = factory("text-classification", model=self.model_id, top_k=None)
        return self._pipe

    async def classify(self, text: str) -> ClassificationResult:
        """Run the pipeline and return every label sorted by score."""
        raw = self._load()(text[:512])
        rows: list[dict[str, Any]] = raw[0] if raw and isinstance(raw[0], list) else raw
        intents = sorted(
            (IntentScore(intent=str(r["label"]), score=float(r["score"])) for r in rows),
            key=lambda s: s.score,
            reverse=True,
        )
        return ClassificationResult(intents=intents or _unknown(), backend=self.name)


# ---------------------------------------------------------------------------
# Zero-shot Claude
# ---------------------------------------------------------------------------

ZS_SYSTEM_PROMPT = f"""You classify a customer support ticket into one of these 27 intents:

{", ".join(BITEXT_INTENTS)}

Return JSON only, no prose:
{{
  "top_intents": [
    {{"intent": "<one of the 27 labels>", "score": <0.0-1.0>}},
    ... up to 3 entries, ordered by score descending, scores summing to at most 1.0
  ]
}}
"""


class _ZeroShotPayload(BaseModel):
    top_intents: list[IntentScore] = Field(..., min_length=1, max_length=3)


class ClaudeZeroShotClassifier:
    """Zero-shot classifier that prompts Claude with the label set.

    Args:
        client: Shared :class:`ClaudeClient`.
    """

    name = "claude_zero_shot"

    def __init__(self, client: ClaudeClient) -> None:
        self._client = client

    async def classify(self, text: str) -> ClassificationResult:
        """Ask Claude for up to three ranked intents.

        Unknown labels returned by the model are dropped; if none remain the
        result is :data:`UNKNOWN_INTENT` with score 0.

        Raises:
            LLMError: Propagated from the client when the API is unreachable.
            LLMOutputError: When the model output does not match the schema.
        """
        result = await self._client.complete(
            system=ZS_SYSTEM_PROMPT, user=text[:2000], max_tokens=256, temperature=0.0
        )
        payload = parse_model(result.text, _ZeroShotPayload)
        intents = [s for s in payload.top_intents if s.intent in BITEXT_INTENTS]
        intents.sort(key=lambda s: s.score, reverse=True)
        return ClassificationResult(
            intents=intents or _unknown(), backend=self.name, usage=result.usage
        )


# ---------------------------------------------------------------------------
# Deterministic keyword heuristic (offline fallback + no-AI baseline)
# ---------------------------------------------------------------------------

#: ``(keywords, intent, score)`` evaluated in order; the first hit wins.
#: Specific phrases go before generic ones so "cancellation fee" is not
#: swallowed by "cancel".
KEYWORD_RULES: tuple[tuple[tuple[str, ...], str, float], ...] = (
    (
        ("human", "real person", "live agent", "speak to someone", "talk to a person"),
        "contact_human_agent",
        0.85,
    ),
    (
        ("cancellation fee", "fee for cancel", "charge for cancel", "penalty"),
        "check_cancellation_fee",
        0.75,
    ),
    (
        ("complaint", "unacceptable", "terrible", "worst", "disgusted", "manager", "file a claim"),
        "complaint",
        0.7,
    ),
    (
        ("track my refund", "refund status", "where is my refund", "refund yet", "money back yet"),
        "track_refund",
        0.75,
    ),
    (
        ("refund policy", "return policy", "can i get a refund", "eligible for a refund"),
        "check_refund_policy",
        0.7,
    ),
    (("refund", "money back", "reimburse"), "get_refund", 0.7),
    (("cancel", "stop my order", "call off"), "cancel_order", 0.7),
    (
        ("track", "where is my", "status of my order", "hasn't arrived", "not arrived"),
        "track_order",
        0.7,
    ),
    (
        ("payment method", "pay with", "accept paypal", "accept credit", "ways to pay"),
        "check_payment_methods",
        0.7,
    ),
    (
        (
            "charged twice",
            "double charged",
            "card declined",
            "payment failed",
            "payment error",
            "payment issue",
            "transaction failed",
        ),
        "payment_issue",
        0.7,
    ),
    (
        ("wrong vat", "invoice is wrong", "invoice shows", "check my invoice", "invoice for"),
        "check_invoice",
        0.65,
    ),
    (("invoice", "receipt", "bill copy"), "get_invoice", 0.7),
    (("password", "log in", "login", "locked out"), "recover_password", 0.7),
    (
        (
            "registration",
            "sign-up error",
            "signup error",
            "cannot register",
            "can't register",
            "error creating",
        ),
        "registration_problems",
        0.7,
    ),
    (
        ("create account", "sign up", "open an account", "register", "new account"),
        "create_account",
        0.7,
    ),
    (
        (
            "delete my account",
            "delete account",
            "close my account",
            "remove my account",
            "erase my data",
            "gdpr",
        ),
        "delete_account",
        0.7,
    ),
    (
        (
            "switch account",
            "change account type",
            "premium account",
            "upgrade my account",
            "downgrade",
        ),
        "switch_account",
        0.7,
    ),
    (
        (
            "edit my account",
            "update my profile",
            "change my email",
            "change my phone",
            "update my details",
            "edit account",
        ),
        "edit_account",
        0.7,
    ),
    (
        (
            "change shipping address",
            "change my shipping address",
            "change the shipping address",
            "change delivery address",
            "update the address",
            "wrong address",
        ),
        "change_shipping_address",
        0.7,
    ),
    (
        (
            "add a shipping address",
            "set up a shipping address",
            "new shipping address",
            "add an address",
            "save an address",
        ),
        "set_up_shipping_address",
        0.7,
    ),
    (
        (
            "change my order",
            "modify order",
            "modify my order",
            "edit my order",
            "add an item",
            "add another item",
        ),
        "change_order",
        0.7,
    ),
    (
        (
            "delivery option",
            "shipping option",
            "next-day",
            "next day",
            "express",
            "same-day",
            "ship to",
        ),
        "delivery_options",
        0.7,
    ),
    (
        (
            "how long",
            "delivery time",
            "when will it arrive",
            "delivery period",
            "eta",
            "shipping time",
        ),
        "delivery_period",
        0.7,
    ),
    (("newsletter", "unsubscribe", "subscribe", "mailing list"), "newsletter_subscription", 0.7),
    (("review", "feedback", "rate my", "leave a comment"), "review", 0.65),
    (("place an order", "place my order", "buy", "purchase", "order some"), "place_order", 0.65),
    (
        (
            "customer service",
            "support hours",
            "contact support",
            "opening hours",
            "how do i contact",
        ),
        "contact_customer_service",
        0.65,
    ),
)


class KeywordHeuristicClassifier:
    """Deterministic keyword router covering the 27 Bitext intents.

    This backend has no learned parameters. It exists so the whole pipeline
    runs offline (CI, demos) and as the non-AI baseline in the evaluation.
    """

    name = "keyword_heuristic"

    async def classify(self, text: str) -> ClassificationResult:
        """Return the first matching rule, or :data:`UNKNOWN_INTENT`."""
        lowered = text.lower()
        for keywords, intent, score in KEYWORD_RULES:
            if any(k in lowered for k in keywords):
                return ClassificationResult(
                    intents=[IntentScore(intent=intent, score=score)], backend=self.name
                )
        return ClassificationResult(intents=_unknown(), backend=self.name)


def _unknown() -> list[IntentScore]:
    return [IntentScore(intent=UNKNOWN_INTENT, score=0.0)]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_classifier(settings: Settings) -> IntentClassifier:
    """Pick the best available backend for ``settings``.

    Order: HF classifier (if enabled and importable) -> Claude zero-shot (if an
    API key is set) -> keyword heuristic. Every fallback is logged at WARNING
    so an offline run is never mistaken for the production configuration.
    """
    log = trace_logger()
    if settings.use_local_classifier and settings.classifier_model:
        clf = HFTransformerClassifier(settings.classifier_model)
        try:
            clf._load()
        except (ImportError, OSError, ValueError) as exc:
            log.warning("HF classifier unavailable ({}); trying next backend", exc)
        else:
            return clf
    if settings.anthropic_api_key:
        client = ClaudeClient(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout_s=settings.llm_timeout_s,
            max_attempts=settings.llm_max_attempts,
        )
        return ClaudeZeroShotClassifier(client)
    log.warning("no classifier model and no API key: using keyword heuristic (offline mode)")
    return KeywordHeuristicClassifier()


@lru_cache(maxsize=1)
def get_classifier() -> IntentClassifier:
    """Process-wide classifier built from :func:`src.config.get_settings`."""
    return build_classifier(get_settings())
