"""The three systems of the mandatory comparison table, plus ablations.

===================  ===========================================  ==============
System               Components                                   Needs
===================  ===========================================  ==============
zero_shot_claude     Claude classifies + analyzes + drafts, no    ANTHROPIC_API_KEY
                     retrieval and no fine-tuned model
distilbert_only      Fine-tuned DistilBERT+LoRA, heuristic        ``ml`` extra +
                     priority, template reply, no LLM             model weights
hybrid               DistilBERT + Claude + retrieval (this        both of the above
                     system in production configuration)
offline_fallback     Keyword heuristic + heuristic priority +     nothing
                     template reply + hashed encoder
===================  ===========================================  ==============

Only ``offline_fallback`` can be measured without keys or model weights; the
report labels it as "fallback determinista, sin LLM" and never presents it as
the production system.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.orchestrator import TriagePipeline
from src.agents.triage_decision import decide
from src.api.schemas import Decision
from src.classifier.intent_classifier import (
    ClaudeZeroShotClassifier,
    HFTransformerClassifier,
    KeywordHeuristicClassifier,
)
from src.config import Settings
from src.eval.metrics import triage_rates
from src.eval.runner import CaseResult, EvalCase, EvalReport, build_offline_pipeline
from src.llm.claude import ClaudeClient
from src.retrieval.encoders import DeterministicEncoder
from src.retrieval.similar_tickets import InMemoryTicketStore, TicketStore

SYSTEM_ZERO_SHOT = "zero_shot_claude"
SYSTEM_DISTILBERT = "distilbert_only"
SYSTEM_HYBRID = "hybrid"
SYSTEM_OFFLINE = "offline_fallback"


@dataclass(frozen=True)
class Availability:
    """Which systems can run in the current environment and why not otherwise."""

    llm: bool
    hf_classifier: bool
    reasons: dict[str, str]


def check_availability(settings: Settings) -> Availability:
    """Probe the API key and the HF stack without importing torch eagerly."""
    reasons: dict[str, str] = {}
    llm = settings.llm_enabled
    if not llm:
        reasons["llm"] = "requiere ANTHROPIC_API_KEY"
    hf = False
    if settings.use_local_classifier:
        try:
            HFTransformerClassifier(settings.classifier_model)._load()
            hf = True
        except (ImportError, OSError, ValueError) as exc:
            reasons["hf_classifier"] = (
                f"requiere extra `ml` y pesos del modelo ({exc.__class__.__name__})"
            )
    else:
        reasons["hf_classifier"] = "USE_LOCAL_CLASSIFIER=false"
    return Availability(llm=llm, hf_classifier=hf, reasons=reasons)


async def build_offline_fallback(settings: Settings | None = None) -> TriagePipeline:
    """Offline system: keyword classifier, heuristic priority, template reply."""
    cfg = settings or Settings(anthropic_api_key=None, use_local_classifier=False)
    return await build_offline_pipeline(cfg, classifier=KeywordHeuristicClassifier())


async def build_zero_shot_claude(settings: Settings, client: ClaudeClient) -> TriagePipeline:
    """Zero-shot baseline: Claude everywhere, empty retrieval store."""
    empty: TicketStore = InMemoryTicketStore(DeterministicEncoder())
    return await build_offline_pipeline(
        settings, classifier=ClaudeZeroShotClassifier(client), client=client, store=empty
    )


async def build_distilbert_only(settings: Settings) -> TriagePipeline:
    """Classifier-only baseline: fine-tuned model, no LLM calls."""
    cfg = settings.model_copy(update={"anthropic_api_key": None})
    return await build_offline_pipeline(
        cfg, classifier=HFTransformerClassifier(settings.classifier_model)
    )


async def build_hybrid(settings: Settings, client: ClaudeClient) -> TriagePipeline:
    """The production configuration: DistilBERT + Claude + retrieval."""
    return await build_offline_pipeline(
        settings, classifier=HFTransformerClassifier(settings.classifier_model), client=client
    )


@dataclass(frozen=True)
class AblationRow:
    """Routing metrics of one confidence-score variant."""

    variant: str
    decision_agreement: float
    auto_resolve_rate: float
    false_escalation_rate: float
    false_auto_resolve_rate: float


def ablate_confidence(
    report: EvalReport, cases: list[EvalCase], settings: Settings
) -> list[AblationRow]:
    """Re-run the decision rule over recorded signals with each term switched off.

    The classifier, retriever and analyzer outputs are taken from ``report``
    (no re-execution), so the rows isolate the contribution of the similarity
    term and of the sentiment/urgency penalty to the routing decision.
    """
    expected = {c.id: c.expected_decision for c in cases}
    variants = {
        "full": (True, True),
        "no_similarity": (False, True),
        "no_sentiment_penalty": (True, False),
        "intent_only": (False, False),
    }
    rows: list[AblationRow] = []
    for name, (use_sim, use_pen) in variants.items():
        got: list[Decision] = []
        exp: list[Decision] = []
        for r in report.cases:
            d = _redecide(r, settings, use_sim, use_pen)
            got.append(d)
            exp.append(expected[r.id])
        rates = triage_rates(exp, got)
        rows.append(
            AblationRow(
                variant=name,
                decision_agreement=rates.decision_agreement,
                auto_resolve_rate=rates.auto_resolve_rate,
                false_escalation_rate=rates.false_escalation_rate,
                false_auto_resolve_rate=rates.false_auto_resolve_rate,
            )
        )
    return rows


def _redecide(r: CaseResult, settings: Settings, use_sim: bool, use_pen: bool) -> Decision:
    return decide(
        intent=r.intents,
        similar=r.similar,
        sentiment=r.sentiment,
        urgency_score=r.urgency_score,
        auto_threshold=settings.auto_resolve_threshold,
        escalate_threshold=settings.escalate_threshold,
        use_similarity=use_sim,
        use_sentiment_penalty=use_pen,
    ).decision
