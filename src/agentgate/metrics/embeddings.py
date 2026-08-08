"""Local text embeddings (A3.1: no embedding API dependency).

Two implementations behind one protocol:

* :class:`HashingEmbedder` — the default. A deterministic hashed bag-of-words with sublinear
  term weighting. It is **lexical, not semantic**, and is named honestly in every report: it
  catches paraphrase-by-word-choice, not paraphrase-by-meaning. It is the default because it
  requires no download, no model file, and produces byte-identical vectors on every machine,
  which is what CI reproducibility needs.
* :class:`SentenceTransformerEmbedder` — a genuine sentence encoder, used when the optional
  ``embeddings`` extra is installed. Better semantics, at the cost of a model download and
  platform-dependent floating point.

``docs/limitations.md`` states which one produced any published number.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from itertools import pairwise
from typing import Any

# Unicode-aware: an ASCII-only class silently mangles ordinary words ("café" -> "caf",
# "naïve" -> "na" + "ve"), which would make similarity depend on the writer's keyboard.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

DEFAULT_DIMENSIONS = 512


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class HashingEmbedder:
    """A deterministic hashed bag-of-words encoder.

    Args:
        dimensions: Vector width. Larger reduces hash collisions at linear memory cost.
        use_bigrams: Include adjacent token pairs, which recovers a little word-order
            sensitivity — enough to stop "A refunded B" and "B refunded A" scoring identically.
    """

    name = "hashing-bow"
    semantic = False

    def __init__(self, dimensions: int = DEFAULT_DIMENSIONS, *, use_bigrams: bool = True) -> None:
        self.dimensions = dimensions
        self.use_bigrams = use_bigrams

    def _features(self, text: str) -> Counter[str]:
        tokens = _tokens(text)
        features = Counter(tokens)
        if self.use_bigrams:
            features.update(f"{first}_{second}" for first, second in pairwise(tokens))
        return features

    def encode_one(self, text: str) -> list[float]:
        """Embed a single string into an L2-normalised vector."""
        vector = [0.0] * self.dimensions
        for feature, count in self._features(text).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        return [self.encode_one(text) for text in texts]


class SentenceTransformerEmbedder:
    """A local sentence-transformers encoder (optional ``embeddings`` extra).

    Args:
        model_id: Hugging Face model id. Small models keep the download inside a free tier.
    """

    semantic = True

    def __init__(self, model_id: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.name = f"st:{model_id}"
        self.model_id = model_id
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of strings."""
        vectors = self._load().encode(list(texts), normalize_embeddings=True)
        return [[float(value) for value in vector] for vector in vectors]


@lru_cache(maxsize=4)
def get_embedder(name: str = "hashing-bow") -> HashingEmbedder | SentenceTransformerEmbedder:
    """Build (and cache) an embedder by name.

    Args:
        name: ``"hashing-bow"`` for the deterministic default, or a ``sentence-transformers``
            model id for the optional encoder.

    Returns:
        An object satisfying :class:`~agentgate.metrics.base.Embedder`.
    """
    if name in ("hashing-bow", "", "default"):
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name)
