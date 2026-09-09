"""Decoupage des documents en morceaux (chunks).

Pourquoi decouper ? Un LLM et un moteur de recherche vectoriel travaillent
mieux sur des passages courts et homogenes : on retrouve le bon passage et
on n'envoie au LLM que ce qui est utile (moins de tokens = moins cher).

Strategie : on respecte d'abord les paragraphes, puis on regroupe jusqu'a
`chunk_size` caracteres, avec un `overlap` (chevauchement) pour ne pas couper
une info en deux entre deux chunks. Les entrees courtes (ex. une FAQ) restent
d'un seul bloc.
"""
from __future__ import annotations


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.replace("\r\n", "\n").split("\n\n")]
    return [p for p in parts if p]


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    paragraphs = _split_paragraphs(text) or [text]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:  # paragraphe trop long -> fenetres glissantes
            start = 0
            while start < len(para):
                chunks.append(para[start : start + chunk_size])
                start += max(1, chunk_size - overlap)
            current = ""
    if current:
        chunks.append(current)

    # applique le chevauchement entre chunks consecutifs
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for prev, nxt in zip(chunks, chunks[1:]):
            tail = prev[-overlap:]
            overlapped.append(f"{tail} {nxt}".strip())
        chunks = overlapped
    return chunks


def chunk_documents(docs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """Transforme des documents en chunks indexables (avec id + metadata).

    Chaque chunk porte deux textes :
      - `text`       : ce qu'on donne au LLM comme contexte (question + reponse) ;
      - `embed_text` : ce qu'on indexe pour la recherche.

    Pour une FAQ, on indexe la QUESTION seule. L'utilisateur pose une question,
    et comparer une question a une question est bien plus precis que la comparer
    a un bloc question + reponse, ou la reponse dilue le sens. Mesure sur 20
    reformulations : rappel@1 de 9/20 a 14/20.
    """
    chunks: list[dict] = []
    for i, doc in enumerate(docs):
        # une FAQ = une unite : on ne la redecoupe pas
        if doc["metadata"].get("type") == "faq":
            pieces = [doc["text"]]
        else:
            pieces = chunk_text(doc["text"], chunk_size, overlap)
        question = doc["metadata"].get("question")
        for j, piece in enumerate(pieces):
            if not piece.strip():
                continue
            chunks.append(
                {
                    # `i` (indice du document) rend l'id unique : sans lui, toutes
                    # les entrees de FAQ partageaient "faq.md::0::0".
                    "id": f"{doc['metadata']['source']}::{doc['metadata'].get('page', 0)}::{i}::{j}",
                    "text": piece,
                    "embed_text": question if question else piece,
                    "metadata": doc["metadata"],
                }
            )
    return chunks
