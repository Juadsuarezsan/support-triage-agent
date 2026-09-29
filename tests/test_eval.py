"""Metrics, runner, baselines/ablation, report writer, judge and the CLI."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from eval import run as eval_run
from src.agents.orchestrator import TriagePipeline
from src.api.schemas import BITEXT_INTENTS
from src.config import Settings
from src.eval.baselines import SYSTEM_OFFLINE, ablate_confidence, check_availability
from src.eval.judge import RUBRIC, JudgeScore, ResponseJudge, judge_batch, render_rubric
from src.eval.metrics import classification_metrics, confusion_counts, percentile, triage_rates
from src.eval.report import load_latest_runs, render_results_md, update_results, write_run
from src.eval.runner import format_report, load_cases, run_eval, run_pipeline_eval
from src.llm.claude import ClaudeClient

# --- metrics ------------------------------------------------------------------


def test_percentile() -> None:
    assert percentile([], 50) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5


def test_classification_metrics_perfect_and_partial() -> None:
    labels = ["a", "b", "c"]
    perfect = classification_metrics(["a", "b", "c"], ["a", "b", "c"], labels)
    assert perfect.macro_f1 == 1.0 and perfect.accuracy == 1.0
    partial = classification_metrics(["a", "a", "b"], ["a", "b", "zzz"], labels)
    assert partial.accuracy == pytest.approx(1 / 3)
    assert partial.per_class_f1["a"] == pytest.approx(2 / 3)
    assert partial.per_class_f1["c"] == 0.0
    assert partial.support == {"a": 2, "b": 1, "c": 0}
    with pytest.raises(ValueError):
        classification_metrics(["a"], [], labels)


def test_confusion_counts_buckets_unknown_predictions() -> None:
    m = confusion_counts(["a", "a", "b"], ["a", "b", "x"], ["a", "b"])
    assert m["a"] == {"a": 1, "b": 1, "other": 0}
    assert m["b"] == {"a": 0, "b": 0, "other": 1}


def test_triage_rates() -> None:
    exp = ["auto_resolve", "suggest", "escalate", "suggest"]
    got = ["auto_resolve", "escalate", "escalate", "auto_resolve"]
    r = triage_rates(exp, got)
    assert r.auto_resolve_rate == 0.5
    assert r.false_escalation_rate == pytest.approx(1 / 3)  # 1 of 3 non-escalate expected
    assert r.false_auto_resolve_rate == pytest.approx(1 / 3)  # 1 of 3 non-auto expected
    assert r.decision_agreement == 0.5
    assert triage_rates([], []).decision_agreement == 0.0
    with pytest.raises(ValueError):
        triage_rates(["a"], [])


# --- eval set + runner ---------------------------------------------------------


def test_eval_set_covers_all_intents_and_is_labelled_synthetic() -> None:
    cases = load_cases()
    assert len(cases) >= 100
    assert {c.expected_intent for c in cases} == set(BITEXT_INTENTS)
    assert all(c.synthetic for c in cases)
    assert len({c.id for c in cases}) == len(cases)


async def test_offline_eval_meets_minimum_quality(offline_pipeline: TriagePipeline) -> None:
    report = await run_pipeline_eval(offline_pipeline)
    assert report.n >= 100 and report.system == "offline_fallback"
    assert report.llm_used is False and report.cost_per_ticket_usd == 0.0
    # Deterministic keyword fallback: top-1 accuracy must clear 30 %.
    assert report.intent_top1_accuracy >= 0.30, f"got {report.intent_top1_accuracy:.2%}"
    # Complaints / human-contact cases MUST escalate even on heuristics.
    assert report.decision_agreement >= 0.40
    assert all(
        c.got_decision == "escalate"
        for c in report.cases
        if c.expected_intent in ("complaint", "contact_human_agent") and c.intent_ok
    )
    assert set(report.per_class_f1) == set(BITEXT_INTENTS)
    text = format_report(report)
    assert "Macro-F1" in text and "[OK]" in text


async def test_run_eval_legacy_dict() -> None:
    d = await run_eval(use_fast_encoder=False)
    assert d["n"] >= 100 and isinstance(d["cases"], list)


async def test_ablation_rows(offline_pipeline: TriagePipeline, offline_settings: Settings) -> None:
    cases = load_cases()[:20]
    report = await run_pipeline_eval(offline_pipeline, cases)
    rows = ablate_confidence(report, cases, offline_settings)
    assert [r.variant for r in rows] == [
        "full",
        "no_similarity",
        "no_sentiment_penalty",
        "intent_only",
    ]
    assert all(0.0 <= r.decision_agreement <= 1.0 for r in rows)


def test_check_availability_offline(offline_settings: Settings) -> None:
    a = check_availability(offline_settings)
    assert not a.llm and not a.hf_classifier
    assert "ANTHROPIC_API_KEY" in a.reasons["llm"]
    a2 = check_availability(offline_settings.model_copy(update={"use_local_classifier": True}))
    assert "ml" in a2.reasons["hf_classifier"]


# --- report -------------------------------------------------------------------


def test_write_run_and_render_pending_cells(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    md = render_results_md(load_latest_runs(runs)) if runs.exists() else render_results_md({})
    assert md.count("pendiente (requiere ANTHROPIC_API_KEY)") >= 4
    report = {
        "macro_f1": 0.5,
        "auto_resolve_rate": 0.25,
        "latency_p95_ms": 12.0,
        "cost_per_ticket_usd": 0.0,
        "n": 3,
        "classifier_backend": "keyword_heuristic",
        "encoder": "det",
        "intent_top1_accuracy": 0.5,
        "intent_top3_accuracy": 0.5,
        "decision_agreement": 0.5,
        "false_escalation_rate": 0.1,
        "false_auto_resolve_rate": 0.0,
        "priority_band_match": 0.9,
        "latency_p50_ms": 5.0,
    }
    path = write_run(
        {
            "system": SYSTEM_OFFLINE,
            "report": report,
            "ablation": [
                {
                    "variant": "full",
                    "decision_agreement": 0.5,
                    "auto_resolve_rate": 0.2,
                    "false_escalation_rate": 0.1,
                    "false_auto_resolve_rate": 0.0,
                }
            ],
        },
        "offline_fallback",
        runs,
    )
    assert path.exists() and json.loads(path.read_text())["system"] == SYSTEM_OFFLINE
    out = update_results(runs, tmp_path / "RESULTS.md")
    text = out.read_text(encoding="utf-8")
    assert "| Zero-shot Claude | pendiente (requiere ANTHROPIC_API_KEY)" in text
    assert "NO es el sistema en producción | 0.500 | 25.0% | 12 ms | $0.0000 |" in text
    assert "no_similarity" not in text and "| full |" in text
    assert "DistilBERT+LoRA" in text and "449" in text


# --- judge --------------------------------------------------------------------


def test_rubric_has_numbered_criteria_and_anchors() -> None:
    text = render_rubric()
    for n in range(1, 10):
        assert f"{n}. " in text
    for dim in ("helpfulness", "tone", "correctness"):
        assert set(RUBRIC[dim]["anchors"]) == {1, 2, 3, 4, 5}
        assert dim.upper() in text


async def test_judge_scores_with_mock(
    make_client: Callable[..., ClaudeClient], mock_json_reply: Callable[[dict[str, Any]], Any]
) -> None:
    mock_json_reply({"helpfulness": 4, "tone": 5, "correctness": 3, "justification": "fine"})
    judge = ResponseJudge(make_client())
    result = await judge.judge("ticket", "draft", ["evidence"])
    assert result.score == JudgeScore(helpfulness=4, tone=5, correctness=3, justification="fine")
    assert result.score.mean == 4.0
    summary = await judge_batch(judge, [("t", "d", ["e"])] * 3, concurrency=2)
    assert summary.n == 3 and summary.mean == 4.0 and summary.usage.calls == 3
    empty = await judge_batch(judge, [])
    assert empty.n == 0


# --- CLI ----------------------------------------------------------------------


def test_eval_run_cli_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("USE_LOCAL_CLASSIFIER", "false")
    runs, results = tmp_path / "runs", tmp_path / "RESULTS.md"
    code = eval_run.main(
        [
            "--runs-dir",
            str(runs),
            "--results",
            str(results),
            "--judge",
            "--systems",
            "offline_fallback",
            "zero_shot_claude",
            "hybrid",
            "distilbert_only",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert (
        "[skip] zero_shot_claude" in out
        and "[skip] hybrid" in out
        and "[skip] distilbert_only" in out
    )
    assert "[skip] judge" in out
    files = list(runs.glob("*-offline_fallback.json"))
    assert len(files) == 1
    assert "fallback determinista" in results.read_text(encoding="utf-8")
