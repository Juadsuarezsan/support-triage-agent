"""Persist evaluation runs and regenerate ``eval/RESULTS.md``.

Every run is written to ``eval/runs/<YYYY-MM-DD>-<name>.json``. The Markdown
report is rendered only from those files: a cell is filled when a run exists
for that system and left as ``pendiente (...)`` otherwise.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.eval.baselines import SYSTEM_DISTILBERT, SYSTEM_HYBRID, SYSTEM_OFFLINE, SYSTEM_ZERO_SHOT

ROOT = Path(__file__).resolve().parent.parent.parent
RUNS_DIR = ROOT / "eval" / "runs"
RESULTS_MD = ROOT / "eval" / "RESULTS.md"
TRAINING_METRICS = ROOT / "models" / "intent-classifier-lora" / "training_metrics.json"

PENDING_KEY = "pendiente (requiere ANTHROPIC_API_KEY)"
PENDING_ML = "pendiente (requiere extra `ml` + pesos del modelo)"
PENDING_BOTH = "pendiente (requiere ANTHROPIC_API_KEY + extra `ml`)"


def write_run(payload: dict[str, Any], name: str, runs_dir: Path = RUNS_DIR) -> Path:
    """Write ``payload`` to ``<runs_dir>/<today>-<name>.json`` and return the path."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    path = runs_dir / f"{stamp}-{name}.json"
    payload = {"generated_at": datetime.now(UTC).isoformat(), **payload}
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def load_latest_runs(runs_dir: Path = RUNS_DIR) -> dict[str, dict[str, Any]]:
    """Return the newest run per ``system`` found in ``runs_dir``."""
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        system = str(data.get("system", path.stem))
        data["_path"] = path.name
        latest[system] = data  # sorted by name => newest date wins
    return latest


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _row(label: str, run: dict[str, Any] | None, pending: str, *, auto_resolve: bool = True) -> str:
    if run is None:
        auto_cell = pending if auto_resolve else "—"
        return f"| {label} | {pending} | {auto_cell} | {pending} | {pending} |"
    rep = run["report"]
    auto_cell = _pct(rep["auto_resolve_rate"]) if auto_resolve else "—"
    return (
        f"| {label} | {rep['macro_f1']:.3f} | {auto_cell} | {rep['latency_p95_ms']:.0f} ms | "
        f"${rep['cost_per_ticket_usd']:.4f} |"
    )


def render_results_md(
    runs: dict[str, dict[str, Any]], training_metrics: Path = TRAINING_METRICS
) -> str:
    """Render the full ``RESULTS.md`` from the latest runs."""
    offline = runs.get(SYSTEM_OFFLINE)
    lines: list[str] = [
        "# Resultados de evaluación",
        "",
        "Generado por `python -m eval.run`. Cada número proviene de un archivo en "
        "`eval/runs/`; las celdas sin corrida quedan como `pendiente (...)`.",
        "",
        "## Tabla comparativa obligatoria",
        "",
        "Eval set: `data/eval/queries.jsonl` (tickets sintéticos escritos a mano, ground truth "
        "manual, 27 intents). Macro-F1 sobre las 27 clases; latencia p95 end-to-end en el "
        "proceso de evaluación; costo estimado con los precios configurados por token.",
        "",
        "| Sistema | Macro-F1 | Auto-resolve | Latencia p95 | Costo/ticket |",
        "|---|---|---|---|---|",
        _row("Zero-shot Claude", runs.get(SYSTEM_ZERO_SHOT), PENDING_KEY),
        _row(
            "DistilBERT fine-tuneado solo",
            runs.get(SYSTEM_DISTILBERT),
            PENDING_ML,
            auto_resolve=False,
        ),
        _row("Híbrido (este sistema)", runs.get(SYSTEM_HYBRID), PENDING_BOTH),
        _row(
            "Fallback determinista, sin LLM (heurística + plantilla + encoder hash) — "
            "NO es el sistema en producción",
            offline,
            "pendiente",
        ),
        "",
    ]
    if offline is not None:
        rep = offline["report"]
        lines += [
            "## Corrida offline (fallback determinista, sin LLM)",
            "",
            f"Archivo: `eval/runs/{offline['_path']}` · n = {rep['n']} · "
            f"clasificador = `{rep['classifier_backend']}` · encoder = `{rep['encoder']}`",
            "",
            "| Métrica | Valor |",
            "|---|---|",
            f"| Macro-F1 (27 intents) | {rep['macro_f1']:.3f} |",
            "| Intent top-1 / top-3 | "
            f"{_pct(rep['intent_top1_accuracy'])} / {_pct(rep['intent_top3_accuracy'])} |",
            f"| Decision agreement | {_pct(rep['decision_agreement'])} |",
            f"| Auto-resolution rate | {_pct(rep['auto_resolve_rate'])} |",
            f"| False-escalation rate | {_pct(rep['false_escalation_rate'])} |",
            f"| False auto-resolve rate | {_pct(rep['false_auto_resolve_rate'])} |",
            f"| Priority band match | {_pct(rep['priority_band_match'])} |",
            "| Latencia p50 / p95 | "
            f"{rep['latency_p50_ms']:.0f} ms / {rep['latency_p95_ms']:.0f} ms |",
            f"| Costo por ticket | ${rep['cost_per_ticket_usd']:.4f} (sin llamadas a LLM) |",
            "",
        ]
        ablation = offline.get("ablation", [])
        if ablation:
            lines += [
                "### Ablación del `confidence_score` (misma corrida, decisiones recalculadas)",
                "",
                "| Variante | Decision agreement | Auto-resolve | False escalation "
                "| False auto-resolve |",
                "|---|---|---|---|---|",
            ]
            for row in ablation:
                lines.append(
                    f"| {row['variant']} | {_pct(row['decision_agreement'])} | "
                    f"{_pct(row['auto_resolve_rate'])} | {_pct(row['false_escalation_rate'])} | "
                    f"{_pct(row['false_auto_resolve_rate'])} |"
                )
            lines.append("")
    judge = (runs.get(SYSTEM_HYBRID) or {}).get("judge")
    lines += [
        "## LLM-as-judge (helpfulness / tone / correctness, rúbrica en `src/eval/judge.py`)",
        "",
    ]
    if judge:
        lines += [
            f"n = {judge['n']} · helpfulness {judge['helpfulness']:.2f} · "
            f"tone {judge['tone']:.2f} · correctness {judge['correctness']:.2f} · "
            f"media {judge['mean']:.2f}",
            "",
        ]
    else:
        lines += [f"{PENDING_KEY} (100 respuestas generadas por el sistema híbrido).", ""]
    if training_metrics.exists():
        tm = json.loads(training_metrics.read_text(encoding="utf-8"))
        lines += [
            "## Clasificador DistilBERT+LoRA (métricas de entrenamiento versionadas)",
            "",
            f"Fuente: `models/intent-classifier-lora/training_metrics.json`. Alcance exacto: "
            f"split de test de **{tm['n_test']} ejemplos** de un subconjunto estratificado de "
            f"{tm['n_train'] + tm['n_val'] + tm['n_test']} filas de Bitext (no el 10 % completo de "
            "2.687). Re-entrenar sobre el split completo requiere red a HF Hub y el extra `ml`.",
            "",
            f"- Macro-F1 (test, {tm['n_test']} ej.): **{tm['test_macro_f1']:.4f}**",
            f"- Épocas: {tm['epochs']} · batch {tm['batch_size']} · lr {tm['lr']} · "
            f"{tm['train_seconds']:.0f} s en CPU",
            "- F1 por clase: ver `docs/classifier_per_class_f1.svg` y `docs/error_analysis.md`",
            "",
        ]
    lines += [
        "## Cómo regenerar",
        "",
        "```bash",
        "python -m eval.run              # offline: fallback determinista + ablación",
        "ANTHROPIC_API_KEY=... python -m eval.run --systems zero_shot_claude hybrid --judge",
        "pip install -e '.[ml]' && python -m eval.run --systems distilbert_only",
        "```",
        "",
    ]
    return "\n".join(lines)


def update_results(runs_dir: Path = RUNS_DIR, results_path: Path = RESULTS_MD) -> Path:
    """Rewrite ``RESULTS.md`` from the runs on disk and return its path."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(render_results_md(load_latest_runs(runs_dir)), encoding="utf-8")
    return results_path
