"""Render the per-class F1 chart of the fine-tuned classifier as a static SVG.

Reads ``models/intent-classifier-lora/training_metrics.json`` (versioned
output of the training script) and writes ``docs/classifier_per_class_f1.svg``
with pure Python, so no plotting library is needed. When the training script
has also saved ``confusion_matrix.json`` next to the metrics, a heatmap is
rendered to ``docs/classifier_confusion_matrix.svg`` as well.

Usage::

    python scripts/plot_classifier_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "models" / "intent-classifier-lora" / "training_metrics.json"
CONFUSION = ROOT / "models" / "intent-classifier-lora" / "confusion_matrix.json"
OUT_F1 = ROOT / "docs" / "classifier_per_class_f1.svg"
OUT_CM = ROOT / "docs" / "classifier_confusion_matrix.svg"

FONT = "font-family='system-ui, -apple-system, Segoe UI, Roboto, sans-serif'"


def _color(f1: float) -> str:
    if f1 >= 0.99:
        return "#1d9a5a"
    if f1 >= 0.95:
        return "#4f9ae0"
    return "#d98a00"


def render_f1_bars(per_class: dict[str, float], macro_f1: float, n_test: int) -> str:
    """Horizontal bar chart, sorted ascending so the weakest classes are on top."""
    rows = sorted(per_class.items(), key=lambda kv: kv[1])
    left, top, row_h, bar_w = 210, 64, 20, 420
    height = top + row_h * len(rows) + 40
    width = left + bar_w + 90
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' {FONT} font-size='12'>",
        f"<rect width='{width}' height='{height}' fill='white'/>",
        "<text x='16' y='24' font-size='16' font-weight='600' fill='#17202b'>"
        "DistilBERT + LoRA · F1 por intent (27 clases)</text>",
        f"<text x='16' y='44' fill='#5b6675'>Split de test de {n_test} ejemplos "
        f"(subconjunto estratificado de Bitext) · macro-F1 = {macro_f1:.4f} · "
        "fuente: models/intent-classifier-lora/training_metrics.json</text>",
    ]
    for tick in (0.90, 0.95, 1.0):
        x = left + bar_w * (tick - 0.85) / 0.15
        parts.append(
            f"<line x1='{x:.1f}' y1='{top - 6}' x2='{x:.1f}' y2='{height - 30}' "
            "stroke='#d9dee7' stroke-dasharray='3 3'/>"
        )
        parts.append(
            f"<text x='{x:.1f}' y='{height - 14}' text-anchor='middle' fill='#5b6675'>"
            f"{tick:.2f}</text>"
        )
    for i, (label, f1) in enumerate(rows):
        y = top + i * row_h
        w = max(0.0, bar_w * (f1 - 0.85) / 0.15)
        parts.append(
            f"<text x='{left - 8}' y='{y + 14}' text-anchor='end' fill='#17202b'>{label}</text>"
        )
        parts.append(
            f"<rect x='{left}' y='{y + 3}' width='{w:.1f}' height='{row_h - 8}' rx='3' "
            f"fill='{_color(f1)}'/>"
        )
        parts.append(f"<text x='{left + w + 6:.1f}' y='{y + 14}' fill='#17202b'>{f1:.3f}</text>")
    parts.append(f"<text x='{left}' y='{height - 14}' fill='#5b6675'>eje x truncado en 0.85</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def render_confusion(labels: list[str], matrix: list[list[int]]) -> str:
    """Heatmap of a confusion matrix (rows = gold, columns = predicted)."""
    cell, left, top = 22, 190, 190
    n = len(labels)
    width, height = left + cell * n + 20, top + cell * n + 20
    total_max = max((v for row in matrix for v in row), default=1) or 1
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
        f"viewBox='0 0 {width} {height}' {FONT} font-size='10'>",
        f"<rect width='{width}' height='{height}' fill='white'/>",
        "<text x='12' y='20' font-size='14' font-weight='600' fill='#17202b'>"
        "Matriz de confusión (filas = real, columnas = predicho)</text>",
    ]
    for j, label in enumerate(labels):
        x = left + j * cell + cell / 2
        parts.append(
            f"<text transform='translate({x:.1f},{top - 6}) rotate(-60)' fill='#17202b'>"
            f"{label}</text>"
        )
    for i, label in enumerate(labels):
        y = top + i * cell
        parts.append(
            f"<text x='{left - 6}' y='{y + 15}' text-anchor='end' fill='#17202b'>{label}</text>"
        )
        for j, value in enumerate(matrix[i]):
            x = left + j * cell
            alpha = 0.08 + 0.92 * (value / total_max) if value else 0.0
            fill = f"rgba(47,109,246,{alpha:.2f})" if i != j else f"rgba(29,154,90,{alpha:.2f})"
            parts.append(
                f"<rect x='{x}' y='{y}' width='{cell}' height='{cell}' fill='{fill}' "
                "stroke='#e6e9ef'/>"
            )
            if value:
                parts.append(
                    f"<text x='{x + cell / 2:.1f}' y='{y + 15}' text-anchor='middle' "
                    f"fill='#17202b'>{value}</text>"
                )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    """CLI entry point."""
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    OUT_F1.parent.mkdir(parents=True, exist_ok=True)
    OUT_F1.write_text(
        render_f1_bars(metrics["per_class_f1"], metrics["test_macro_f1"], metrics["n_test"]),
        encoding="utf-8",
    )
    print(f"wrote {OUT_F1}")
    if CONFUSION.exists():
        cm = json.loads(CONFUSION.read_text(encoding="utf-8"))
        OUT_CM.write_text(render_confusion(cm["labels"], cm["matrix"]), encoding="utf-8")
        print(f"wrote {OUT_CM}")
    else:
        print("confusion_matrix.json not found: re-run the training script to produce it")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
