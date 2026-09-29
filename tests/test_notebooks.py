"""The notebooks are built from code that runs offline and carry real outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat

from scripts import build_notebook as bn


def test_build_notebooks_execute_and_store_outputs(tmp_path: Path) -> None:
    assert bn.main(["--out", str(tmp_path)]) == 0
    for name in ("demo.ipynb", "02_eval_pipeline.ipynb", "03_baselines_comparison.ipynb"):
        nb = nbformat.read(str(tmp_path / name), as_version=4)
        code = [c for c in nb.cells if c.cell_type == "code"]
        assert code and all(c.execution_count for c in code)
        outputs = "".join(o.get("text", "") for c in code for o in c.outputs)
        assert "keyword_heuristic" in outputs
    demo = nbformat.read(str(tmp_path / "demo.ipynb"), as_version=4)
    text = "".join(o["text"] for c in demo.cells if c.cell_type == "code" for o in c.outputs)
    assert "t-2: intent=complaint" in text and "escalate" in text
