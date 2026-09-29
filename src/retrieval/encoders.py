"""Text encoders for similar-ticket retrieval.

* :class:`DeterministicEncoder` - hashed bag-of-tokens; no model download.
  Used in tests, CI and offline evaluation runs.
* :class:`HFEncoder` - ``sentence-transformers/all-mpnet-base-v2`` (``ml``
  extra). Imported lazily so the API boots without torch.
* :class:`CachedEncoder` - wraps any encoder with a SHA-256 keyed cache and
  only sends cache misses to the inner encoder, in one batch.
"""

from __future__ import annotations

import hashlib
import re
from collections import OrderedDict
from typing import Any, Protocol

from src.observability import trace_logger

_TOKEN = re.compile(r"[a-zA-Z]+")


class Encoder(Protocol):
    """Anything that turns texts into fixed-size float vectors."""

    name: str

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode ``texts`` in one batch, preserving order."""
        ...


class DeterministicEncoder:
    """Hashed bag-of-tokens embedding (dimension 256, L2-normalised).

    Semantically weak by design: it measures lexical overlap only. It is
    reproducible, dependency-free and fast, which is what tests need.
    """

    name = "deterministic_hash_256"
    DIM = 256

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode ``texts``; an empty text maps to the ``<empty>`` token."""
        out: list[list[float]] = []
        for text in texts:
            tokens = _TOKEN.findall(text.lower()) or ["<empty>"]
            acc = [0.0] * self.DIM
            for tok in tokens:
                digest = hashlib.sha256(tok.encode()).digest()
                raw = (digest * ((self.DIM // len(digest)) + 1))[: self.DIM]
                for i, b in enumerate(raw):
                    acc[i] += (b - 128) / 128.0
            norm = sum(x * x for x in acc) ** 0.5 or 1.0
            out.append([x / norm for x in acc])
        return out


class HFEncoder:
    """``all-mpnet-base-v2`` sentence embeddings (768-d, normalised).

    Args:
        model_name: Sentence-transformers model ID.
        model_factory: Injectable replacement for ``SentenceTransformer``.
    """

    name = "sentence-transformers/all-mpnet-base-v2"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        model_factory: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self._factory = model_factory
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            factory = self._factory
            if factory is None:
                from sentence_transformers import SentenceTransformer  # heavy, deferred

                factory = SentenceTransformer
            trace_logger().info("loading sentence-transformer {} (~420 MB)", self.model_name)
            self._model = factory(self.model_name)
        return self._model

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode ``texts`` with normalised embeddings."""
        vectors = self._load().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in row] for row in vectors]


class CachedEncoder:
    """LRU cache in front of another encoder, keyed by SHA-256 of the text.

    Args:
        inner: The encoder that computes cache misses.
        max_items: Cache capacity (least-recently-used entries are evicted).
    """

    def __init__(self, inner: Encoder, max_items: int = 4096) -> None:
        self.inner = inner
        self.name = f"cached({inner.name})"
        self.max_items = max_items
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def encode(self, texts: list[str]) -> list[list[float]]:
        """Return cached vectors and encode only the misses, in one batch."""
        keys = [self._key(t) for t in texts]
        missing: dict[str, str] = {}
        for key, text in zip(keys, texts, strict=True):
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
            else:
                self.misses += 1
                missing.setdefault(key, text)
        if missing:
            fresh = await self.inner.encode(list(missing.values()))
            for key, vector in zip(missing.keys(), fresh, strict=True):
                self._cache[key] = vector
                if len(self._cache) > self.max_items:
                    self._cache.popitem(last=False)
        return [self._cache[k] for k in keys]


def build_encoder(backend: str) -> Encoder:
    """Build the encoder named by ``ENCODER_BACKEND`` wrapped in a cache.

    Args:
        backend: ``"deterministic"`` or ``"hf"``.

    Returns:
        A :class:`CachedEncoder` around the requested backend.

    Raises:
        ValueError: For an unknown backend name.
    """
    if backend == "deterministic":
        return CachedEncoder(DeterministicEncoder())
    if backend == "hf":
        return CachedEncoder(HFEncoder())
    raise ValueError(f"unknown encoder backend: {backend!r}")
