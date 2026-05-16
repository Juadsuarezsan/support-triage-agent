"""Zero-shot Claude classifier heuristic fallback (no API key needed)."""
from __future__ import annotations

import pytest

from src.classifier.intent_classifier import ClaudeZeroShotClassifier


@pytest.fixture
def classifier():
    return ClaudeZeroShotClassifier(model="claude-sonnet-4-5", api_key=None)


@pytest.mark.asyncio
async def test_refund_keyword(classifier):
    [top] = await classifier.classify("I want my money back, the product is broken")
    assert top.intent == "get_refund"


@pytest.mark.asyncio
async def test_human_keyword(classifier):
    [top] = await classifier.classify("Please connect me with a human agent")
    assert top.intent == "contact_human_agent"
    assert top.score >= 0.8


@pytest.mark.asyncio
async def test_default_general(classifier):
    [top] = await classifier.classify("Just wondering")
    assert top.intent == "general_question"
