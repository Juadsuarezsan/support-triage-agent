"""End-to-end eval (uses deterministic encoder so it runs in CI without model download)."""
from __future__ import annotations

import pytest

from src.eval.runner import run_eval


@pytest.mark.asyncio
async def test_eval_meets_minimum_quality():
    report = await run_eval(use_fast_encoder=True)
    assert report["n"] > 0
    # With heuristics + deterministic encoder, top-1 accuracy should clear ~30%.
    assert report["intent_top1_accuracy"] >= 0.30, f"got {report['intent_top1_accuracy']:.2%}"
    # The complaints / human-contact cases MUST escalate even on heuristics.
    assert report["decision_agreement"] >= 0.40
