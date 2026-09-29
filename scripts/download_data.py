"""Download the raw datasets and record their SHA-256 in ``data/MANIFEST.txt``.

Datasets
--------
* **Bitext Customer Support** (Hugging Face, CC BY 4.0) - the CSV is fetched
  straight from the dataset repository, no ``datasets`` library needed.
* **Customer Support on Twitter** (Kaggle, CC BY-NC-SA 4.0) - fetched through
  the Kaggle API with ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (or
  ``~/.kaggle/kaggle.json``).

Usage::

    python scripts/download_data.py --all
    python scripts/download_data.py --bitext
    python scripts/download_data.py --twitter

Files land in ``data/raw/`` (gitignored). Every download is streamed to disk,
hashed and appended to the manifest. Network access to huggingface.co and
kaggle.com is required; the module is unit-tested with mocked HTTP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "MANIFEST.txt"

BITEXT_REPO = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
BITEXT_FILE = "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
BITEXT_URL = f"https://huggingface.co/datasets/{BITEXT_REPO}/resolve/main/{BITEXT_FILE}"
BITEXT_LICENSE = "CC BY 4.0"

TWITTER_SLUG = "thoughtvector/customer-support-on-twitter"
TWITTER_URL = f"https://www.kaggle.com/api/v1/datasets/download/{TWITTER_SLUG}"
TWITTER_LICENSE = "CC BY-NC-SA 4.0"

CHUNK = 1 << 20


@dataclass(frozen=True)
class Downloaded:
    """Result of one download."""

    name: str
    path: Path
    sha256: str
    bytes: int
    source: str
    license: str


def sha256_of(path: Path) -> str:
    """Stream-hash a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_download(
    client: httpx.Client, url: str, dest: Path, *, auth: tuple[str, str] | None = None
) -> int:
    """Stream ``url`` to ``dest`` and return the number of bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with client.stream("GET", url, auth=auth, follow_redirects=True, timeout=120.0) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_bytes(CHUNK):
                fh.write(chunk)
                written += len(chunk)
    return written


def kaggle_credentials() -> tuple[str, str]:
    """Read Kaggle credentials from the environment or ``~/.kaggle/kaggle.json``."""
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if cfg.exists():
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return str(data["username"]), str(data["key"])
    raise RuntimeError("Kaggle credentials missing: set KAGGLE_USERNAME/KAGGLE_KEY")


def download_bitext(client: httpx.Client, raw_dir: Path = RAW_DIR) -> Downloaded:
    """Fetch the Bitext CSV."""
    dest = raw_dir / "bitext" / BITEXT_FILE
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with client.stream(
        "GET", BITEXT_URL, headers=headers, follow_redirects=True, timeout=120.0
    ) as r:
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        with dest.open("wb") as fh:
            for chunk in r.iter_bytes(CHUNK):
                fh.write(chunk)
                size += len(chunk)
    return Downloaded("bitext", dest, sha256_of(dest), size, BITEXT_URL, BITEXT_LICENSE)


def download_twitter(client: httpx.Client, raw_dir: Path = RAW_DIR) -> Downloaded:
    """Fetch the Kaggle archive (zip) of Customer Support on Twitter."""
    dest = raw_dir / "twitter" / "customer-support-on-twitter.zip"
    size = stream_download(client, TWITTER_URL, dest, auth=kaggle_credentials())
    return Downloaded("twitter", dest, sha256_of(dest), size, TWITTER_URL, TWITTER_LICENSE)


def update_manifest(items: list[Downloaded], manifest: Path = MANIFEST, root: Path = ROOT) -> None:
    """Append one line per download to the manifest (creating the header if needed)."""
    manifest.parent.mkdir(parents=True, exist_ok=True)
    header = "# name\tsha256\tbytes\tpath\tlicense\tsource\tdownloaded_at\n"
    existing = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
    lines = [ln for ln in existing.splitlines() if ln and not ln.startswith("#")]
    kept = [ln for ln in lines if ln.split("\t")[0] not in {i.name for i in items}]
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    for i in items:
        rel = i.path.relative_to(root).as_posix()
        kept.append(f"{i.name}\t{i.sha256}\t{i.bytes}\t{rel}\t{i.license}\t{i.source}\t{stamp}")
    manifest.write_text(header + "\n".join(sorted(kept)) + "\n", encoding="utf-8")


LOCAL_FILES: dict[str, tuple[str, str]] = {
    "eval_queries": ("data/eval/queries.jsonl", "hand-written synthetic, MIT"),
    "eval_seed_tickets": ("data/eval/seed_tickets.jsonl", "hand-written synthetic, MIT"),
}


def record_local(
    root: Path = ROOT, files: dict[str, tuple[str, str]] | None = None
) -> list[Downloaded]:
    """Hash the versioned evaluation files so the manifest also covers them."""
    out: list[Downloaded] = []
    for name, (rel, license_) in (files or LOCAL_FILES).items():
        path = root / rel
        out.append(Downloaded(name, path, sha256_of(path), path.stat().st_size, "repo", license_))
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--bitext", action="store_true")
    parser.add_argument("--twitter", action="store_true")
    parser.add_argument("--local", action="store_true", help="hash the versioned eval files")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)
    if not (args.all or args.bitext or args.twitter or args.local):
        parser.error("choose --all, --bitext, --twitter and/or --local")
    done: list[Downloaded] = []
    if args.local:
        done.extend(record_local())
    if args.all or args.bitext or args.twitter:
        with httpx.Client() as client:
            if args.all or args.bitext:
                done.append(download_bitext(client, args.raw_dir))
            if args.all or args.twitter:
                done.append(download_twitter(client, args.raw_dir))
    update_manifest(done, args.manifest)
    for d in done:
        print(f"[ok] {d.name}: {d.bytes} bytes sha256={d.sha256[:12]}... -> {d.path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
