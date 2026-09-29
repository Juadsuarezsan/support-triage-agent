.PHONY: install install-ml test lint format typecheck eval eval-full serve demo-bake notebook data gitleaks clean

PY ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-ml:
	$(PIP) install -e ".[dev,ml,viz]"

test:
	$(PY) -m pytest --cov=src --cov-report=term-missing --cov-fail-under=70

lint:
	.venv/bin/ruff check .
	.venv/bin/black --check .

format:
	.venv/bin/black .
	.venv/bin/ruff check --fix .

typecheck:
	.venv/bin/mypy --strict src/ eval/

eval:
	$(PY) -m eval.run

eval-full:
	$(PY) -m eval.run --systems offline_fallback zero_shot_claude distilbert_only hybrid --judge

serve:
	.venv/bin/uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

demo-bake:
	$(PY) scripts/bake_demo_predictions.py

notebook:
	$(PY) scripts/build_notebook.py

data:
	$(PY) scripts/download_data.py --all

gitleaks:
	gitleaks detect --no-banner --redact --source .

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
