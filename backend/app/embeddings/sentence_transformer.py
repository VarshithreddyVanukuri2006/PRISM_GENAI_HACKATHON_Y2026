from __future__ import annotations

from typing import Sequence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SentenceTransformerEmbedder:
    """Lazy, CPU-first adapter around Sentence Transformers."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL,
                 batch_size: int = 32):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Semantic retrieval requires Sentence Transformers. Install the optional "
                    "dependencies with: pip install -r backend/requirements-semantic.txt"
                ) from exc
            try:
                self._model = SentenceTransformer(self.model_name, device="cpu")
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load embedding model '{self.model_name}'. Check the model name, "
                    "network access, and the local Hugging Face model cache."
                ) from exc
        return self._model

    def encode(self, texts: Sequence[str]):
        if not texts:
            return []
        model = self._load_model()
        return model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
