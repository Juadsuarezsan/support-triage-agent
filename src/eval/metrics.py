"""Pure metric helpers shared by the runner, the baselines and the report.

Everything here is dependency-free (no scikit-learn) so the evaluation runs
in the lightweight CI environment.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


def percentile(xs: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile ``p`` (0-100) of ``xs``; 0.0 when empty."""
    if not xs:
        return 0.0
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Macro-F1 and per-class F1 over a fixed label set.

    Attributes:
        macro_f1: Unweighted mean of the per-class F1 over ``labels``.
        accuracy: Top-1 accuracy.
        per_class_f1: F1 per label (0.0 for labels never predicted nor present).
        support: Number of gold examples per label.
    """

    macro_f1: float
    accuracy: float
    per_class_f1: dict[str, float]
    support: dict[str, int]


def classification_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> ClassificationMetrics:
    """Compute macro-F1, accuracy and per-class F1.

    Args:
        y_true: Gold labels.
        y_pred: Predicted labels (any string; unknown labels count as errors).
        labels: The label set the macro average runs over.

    Returns:
        A :class:`ClassificationMetrics`.

    Raises:
        ValueError: When ``y_true`` and ``y_pred`` differ in length.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    support: Counter[str] = Counter()
    for gold, pred in zip(y_true, y_pred, strict=True):
        support[gold] += 1
        if gold == pred:
            tp[gold] += 1
        else:
            fp[pred] += 1
            fn[gold] += 1
    per_class: dict[str, float] = {}
    for label in labels:
        denom = 2 * tp[label] + fp[label] + fn[label]
        per_class[label] = (2 * tp[label] / denom) if denom else 0.0
    macro = sum(per_class.values()) / len(labels) if labels else 0.0
    acc = sum(tp.values()) / len(y_true) if y_true else 0.0
    return ClassificationMetrics(
        macro_f1=round(macro, 6),
        accuracy=round(acc, 6),
        per_class_f1={k: round(v, 6) for k, v in per_class.items()},
        support={label: support[label] for label in labels},
    )


def confusion_counts(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Nested ``{gold: {pred: count}}`` confusion matrix over ``labels``.

    Predictions outside ``labels`` are bucketed under ``"other"``.
    """
    matrix: dict[str, dict[str, int]] = {g: dict.fromkeys([*labels, "other"], 0) for g in labels}
    for gold, pred in zip(y_true, y_pred, strict=True):
        if gold not in matrix:
            continue
        matrix[gold][pred if pred in matrix[gold] else "other"] += 1
    return matrix


@dataclass(frozen=True)
class TriageRates:
    """Routing quality of the triage decisions.

    Attributes:
        auto_resolve_rate: Share of tickets the system closed without a human.
        false_escalation_rate: Escalated tickets among those that did not need it.
        false_auto_resolve_rate: Auto-resolved tickets among those that needed a human.
        decision_agreement: Exact agreement with the expected decision.
    """

    auto_resolve_rate: float
    false_escalation_rate: float
    false_auto_resolve_rate: float
    decision_agreement: float


def triage_rates(expected: Sequence[str], got: Sequence[str]) -> TriageRates:
    """Compute :class:`TriageRates` from expected and produced decisions."""
    n = len(expected)
    if n != len(got):
        raise ValueError("expected and got must have the same length")
    if n == 0:
        return TriageRates(0.0, 0.0, 0.0, 0.0)
    auto = sum(1 for g in got if g == "auto_resolve")
    not_escalate = [(e, g) for e, g in zip(expected, got, strict=True) if e != "escalate"]
    not_auto = [(e, g) for e, g in zip(expected, got, strict=True) if e != "auto_resolve"]
    false_esc = sum(1 for _, g in not_escalate if g == "escalate")
    false_auto = sum(1 for _, g in not_auto if g == "auto_resolve")
    agree = sum(1 for e, g in zip(expected, got, strict=True) if e == g)
    return TriageRates(
        auto_resolve_rate=round(auto / n, 6),
        false_escalation_rate=round(false_esc / len(not_escalate), 6) if not_escalate else 0.0,
        false_auto_resolve_rate=round(false_auto / len(not_auto), 6) if not_auto else 0.0,
        decision_agreement=round(agree / n, 6),
    )
