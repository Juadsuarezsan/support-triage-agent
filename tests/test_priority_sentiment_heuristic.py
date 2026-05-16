"""Heuristic fallback for PrioritySentimentAnalyzer."""
from __future__ import annotations

import pytest

from src.agents.priority_sentiment import PrioritySentimentAnalyzer


@pytest.fixture
def analyzer():
    return PrioritySentimentAnalyzer(model="claude-sonnet-4-5", api_key=None)


@pytest.mark.asyncio
async def test_urgent_keywords_yield_p0(analyzer):
    priority, sentiment, urgency, _ = await analyzer.analyze("PRODUCTION IS DOWN, URGENT")
    assert priority == "P0"
    assert urgency >= 0.8


@pytest.mark.asyncio
async def test_negative_yields_p1(analyzer):
    priority, sentiment, urgency, _ = await analyzer.analyze("This is the worst service ever")
    assert priority == "P1"
    assert sentiment == "neg"


@pytest.mark.asyncio
async def test_neutral_defaults_p3(analyzer):
    priority, sentiment, urgency, _ = await analyzer.analyze("How does shipping work?")
    assert priority == "P3"
    assert sentiment == "neu"
