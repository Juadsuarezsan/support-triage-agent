"""download_data.py and make_splits.py with mocked network / synthetic CSV."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest
import respx

from scripts import download_data as dl
from scripts import make_splits as ms


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    return tmp_path / "raw"


def test_download_bitext_streams_and_hashes(raw_dir: Path, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    body = b"instruction,intent,category\nhi,get_refund,REFUND\n"
    with respx.mock() as router:
        router.get(dl.BITEXT_URL).mock(return_value=httpx.Response(200, content=body))
        with httpx.Client() as client:
            d = dl.download_bitext(client, raw_dir)
    assert d.path.read_bytes() == body
    assert d.sha256 == hashlib.sha256(body).hexdigest() and d.bytes == len(body)
    assert d.license == "CC BY 4.0"


def test_download_twitter_uses_kaggle_basic_auth(raw_dir: Path, monkeypatch) -> None:
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    with respx.mock() as router:
        route = router.get(dl.TWITTER_URL).mock(return_value=httpx.Response(200, content=b"PK.."))
        with httpx.Client() as client:
            d = dl.download_twitter(client, raw_dir)
    assert d.path.suffix == ".zip" and d.bytes == 4
    assert route.calls[0].request.headers["authorization"].startswith("Basic ")


def test_download_twitter_without_credentials_raises(
    raw_dir: Path, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(RuntimeError, match="Kaggle credentials"), httpx.Client() as client:
        dl.download_twitter(client, raw_dir)


def test_kaggle_json_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".kaggle").mkdir()
    (tmp_path / ".kaggle" / "kaggle.json").write_text('{"username": "a", "key": "b"}')
    assert dl.kaggle_credentials() == ("a", "b")


def test_http_error_propagates(raw_dir: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with respx.mock() as router:
        router.get(dl.BITEXT_URL).mock(return_value=httpx.Response(404))
        with pytest.raises(httpx.HTTPStatusError), httpx.Client() as client:
            dl.download_bitext(client, raw_dir)


def test_manifest_is_rewritten_per_name(tmp_path: Path) -> None:
    f = tmp_path / "data" / "x.bin"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"abc")
    item = dl.Downloaded("x", f, dl.sha256_of(f), 3, "src", "MIT")
    manifest = tmp_path / "MANIFEST.txt"
    dl.update_manifest([item], manifest, root=tmp_path)
    dl.update_manifest([item], manifest, root=tmp_path)  # idempotent
    lines = [ln for ln in manifest.read_text().splitlines() if not ln.startswith("#")]
    assert len(lines) == 1 and lines[0].startswith("x\t" + hashlib.sha256(b"abc").hexdigest())


def test_cli_local_records_versioned_eval_files(tmp_path: Path) -> None:
    manifest = tmp_path / "MANIFEST.txt"
    assert dl.main(["--local", "--manifest", str(manifest)]) == 0
    text = manifest.read_text(encoding="utf-8")
    assert "eval_queries" in text and "data/eval/queries.jsonl" in text


def test_cli_requires_a_target() -> None:
    with pytest.raises(SystemExit):
        dl.main([])


def _write_csv(path: Path, per_intent: dict[str, int]) -> list[str]:
    labels: list[str] = []
    with path.open("w", encoding="utf-8") as fh:
        fh.write("instruction,intent,category\n")
        for intent, n in per_intent.items():
            for i in range(n):
                fh.write(f"text {i},{intent},CAT\n")
                labels.append(intent)
    return labels


def test_stratified_split_is_deterministic_and_stratified(tmp_path: Path) -> None:
    csv_path = tmp_path / "bitext.csv"
    labels = _write_csv(csv_path, {"a": 100, "b": 50, "c": 10})
    assert ms.read_intents(csv_path) == labels
    s1 = ms.stratified_split(labels)
    s2 = ms.stratified_split(labels)
    assert s1 == s2
    assert sorted(s1["train"] + s1["val"] + s1["test"]) == list(range(160))
    assert len(s1["test"]) == 16 and len(s1["val"]) == 16 and len(s1["train"]) == 128
    assert sum(1 for i in s1["test"] if labels[i] == "c") == 1  # small class kept in test
    assert ms.stratified_split(labels, seed=1) != s1
    with pytest.raises(ValueError):
        ms.stratified_split(labels, fractions=(0.5, 0.5, 0.5))


def test_make_splits_cli_writes_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "bitext.csv"
    _write_csv(csv_path, {"a": 20, "b": 20})
    out = tmp_path / "splits"
    assert ms.main(["--csv", str(csv_path), "--out", str(out)]) == 0
    info = json.loads((out / "SPLIT_INFO.json").read_text())
    assert info["seed"] == 20260516 and info["counts"] == {"train": 32, "val": 4, "test": 4}
    assert (out / "train.txt").read_text().splitlines()[0].isdigit()
    assert ms.main(["--csv", str(tmp_path / "missing.csv"), "--out", str(out)]) == 1
    with pytest.raises(ValueError):
        ms.read_intents(csv_path, column="nope")
