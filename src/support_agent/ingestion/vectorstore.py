"""Index vectoriel : embeddings + FAISS (recherche par similarite cosinus).

Deux backends d'embeddings, choisis automatiquement :

1. sentence-transformers (recommande) : embeddings semantiques de qualite,
   locaux, gratuits. Necessite un telechargement de modele la 1re fois.
2. TF-IDF (scikit-learn) : repli automatique si sentence-transformers n'est
   pas installe ou si le modele ne peut pas etre telecharge. Aucun
   telechargement, tourne partout — parfait pour une demo hors-ligne.

Dans les deux cas les vecteurs sont normalises (L2) et indexes dans un
`IndexFlatIP` FAISS : le produit scalaire = similarite cosinus dans [0, 1].
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


class Embedder:
    """Encapsule le backend d'embeddings choisi."""

    def __init__(self, backend: str, model_name: str):
        self.backend = backend
        self.model_name = model_name
        self._model = None          # sentence-transformers
        self._vectorizer = None     # tfidf

        if backend == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(model_name)
            except Exception as exc:
                print(
                    f"[embeddings] sentence-transformers indisponible ({exc}).\n"
                    f"[embeddings] Repli sur TF-IDF (local, sans telechargement)."
                )
                self.backend = "tfidf"

        if self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

    # -- construction (indexation) --
    def fit_transform(self, texts: list[str]) -> np.ndarray:
        if self.backend == "tfidf":
            matrix = self._vectorizer.fit_transform(texts).toarray().astype("float32")
            return _l2_normalize(matrix)
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")

    # -- requete --
    def transform(self, texts: list[str]) -> np.ndarray:
        if self.backend == "tfidf":
            matrix = self._vectorizer.transform(texts).toarray().astype("float32")
            return _l2_normalize(matrix)
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype="float32")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    def __init__(self, embedder: Embedder, chunks: list[dict], index):
        self.embedder = embedder
        self.chunks = chunks
        self.index = index

    # ------------------------------------------------------------------ #
    @classmethod
    def build(cls, chunks: list[dict], backend: str, model_name: str) -> "VectorStore":
        import faiss

        if not chunks:
            raise ValueError("Aucun chunk a indexer : lance d'abord l'ingestion.")
        embedder = Embedder(backend, model_name)
        # On indexe `embed_text` quand il existe (la question, pour une FAQ) et
        # `text` sinon. Le contexte rendu au LLM reste toujours `text` complet.
        vectors = embedder.fit_transform(
            [c.get("embed_text") or c["text"] for c in chunks]
        )
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return cls(embedder, chunks, index)

    # ------------------------------------------------------------------ #
    def search(self, query: str, k: int = 4) -> list[tuple[dict, float]]:
        vec = self.embedder.transform([query])
        scores, ids = self.index.search(vec, min(k, len(self.chunks)))
        results: list[tuple[dict, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    # ------------------------------------------------------------------ #
    def save(self, index_dir: Path) -> None:
        import faiss

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "index.faiss"))
        (index_dir / "chunks.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (index_dir / "meta.json").write_text(
            json.dumps(
                {"backend": self.embedder.backend, "model": self.embedder.model_name},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        if self.embedder.backend == "tfidf":
            with open(index_dir / "vectorizer.pkl", "wb") as fh:
                pickle.dump(self.embedder._vectorizer, fh)

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        import faiss

        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_dir / "index.faiss"))

        embedder = Embedder.__new__(Embedder)
        embedder.backend = meta["backend"]
        embedder.model_name = meta["model"]
        embedder._model = None
        embedder._vectorizer = None
        if meta["backend"] == "tfidf":
            with open(index_dir / "vectorizer.pkl", "rb") as fh:
                embedder._vectorizer = pickle.load(fh)
        else:
            from sentence_transformers import SentenceTransformer

            embedder._model = SentenceTransformer(meta["model"])
        return cls(embedder, chunks, index)

    @staticmethod
    def exists(index_dir: Path) -> bool:
        index_dir = Path(index_dir)
        return (index_dir / "index.faiss").exists() and (index_dir / "chunks.json").exists()
