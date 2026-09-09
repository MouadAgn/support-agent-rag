"""Construit l'index vectoriel a partir de la FAQ et des PDF.

    python scripts/ingest.py

Etapes : chargement (FAQ + PDF + OCR) -> chunking -> embeddings -> index FAISS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from support_agent.config import get_settings                     # noqa: E402
from support_agent.ingestion.chunking import chunk_documents      # noqa: E402
from support_agent.ingestion.loaders import load_all              # noqa: E402
from support_agent.ingestion.vectorstore import VectorStore       # noqa: E402


def main() -> None:
    s = get_settings()
    print("1) Chargement des sources (FAQ + PDF)...")
    docs = load_all(s.faq_dir, s.pdf_dir)
    print(f"   -> {len(docs)} document(s) charge(s)")

    print(f"2) Chunking (taille={s.chunk_size}, overlap={s.chunk_overlap})...")
    chunks = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
    print(f"   -> {len(chunks)} chunk(s)")

    print(f"3) Embeddings + index FAISS (backend demande : {s.embeddings_backend})...")
    store = VectorStore.build(chunks, s.embeddings_backend, s.embeddings_model)
    store.save(s.index_dir)
    print(f"   -> backend reel utilise : {store.embedder.backend}")
    print(f"   -> index enregistre dans : {s.index_dir}")
    print("\nOK. Tu peux maintenant lancer :  python app.py")


if __name__ == "__main__":
    main()
