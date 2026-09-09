"""Chargement des sources : FAQ (Markdown) et PDF (texte + OCR si besoin).

Chaque source est transformee en une liste de "documents" :
    {"text": str, "metadata": {"source": str, "type": str, ...}}
"""
from __future__ import annotations

import re
from pathlib import Path


# --------------------------------------------------------------------------- #
#  FAQ Markdown                                                                #
# --------------------------------------------------------------------------- #
def load_faq(faq_dir: Path) -> list[dict]:
    """Parse des fichiers FAQ au format :

        ### Q: Comment suivre ma commande ?
        Reponse sur plusieurs lignes...

    Chaque paire question/reponse devient un document (bon pour le RAG :
    une unite semantique claire, on ne la decoupe pas davantage).
    """
    docs: list[dict] = []
    for path in sorted(Path(faq_dir).glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        # decoupe sur les entetes de question "### Q:"
        blocks = re.split(r"\n(?=###\s*Q\s*:)", raw)
        for block in blocks:
            m = re.match(r"###\s*Q\s*:\s*(.+)", block.strip())
            if not m:
                continue
            question = m.group(1).strip()
            answer = block.strip()[m.end():].strip()
            if not answer:
                continue
            docs.append(
                {
                    "text": f"Question : {question}\nReponse : {answer}",
                    "metadata": {
                        "source": path.name,
                        "type": "faq",
                        "question": question,
                    },
                }
            )
    return docs


# --------------------------------------------------------------------------- #
#  PDF (texte natif + OCR en repli)                                            #
# --------------------------------------------------------------------------- #
def load_pdfs(pdf_dir: Path) -> list[dict]:
    """Extrait le texte de chaque PDF, page par page.

    - PDF "texte" : extraction directe via pypdf (rapide, fiable).
    - PDF scanne / image : si le texte est vide, on tente l'OCR
      (pytesseract), s'il est installe. Sinon on previent et on ignore.
    """
    from pypdf import PdfReader

    docs: list[dict] = []
    for path in sorted(Path(pdf_dir).glob("*.pdf")):
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            source_type = "pdf"
            if not text:  # page image -> OCR
                text = _ocr_page(path, i)
                source_type = "pdf-ocr"
            if not text:
                continue
            docs.append(
                {
                    "text": text,
                    "metadata": {"source": path.name, "type": source_type, "page": i},
                }
            )
    return docs


def _ocr_page(pdf_path: Path, page_number: int) -> str:
    """OCR d'une page (optionnel). Renvoie "" si l'OCR n'est pas disponible."""
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except Exception:
        print(
            f"  [OCR ignore] {pdf_path.name} p.{page_number} : "
            "installe pytesseract + pdf2image + tesseract pour l'OCR."
        )
        return ""
    try:
        images = convert_from_path(
            str(pdf_path), first_page=page_number, last_page=page_number, dpi=200
        )
        if not images:
            return ""
        return pytesseract.image_to_string(images[0], lang="fra+eng").strip()
    except Exception as exc:  # pragma: no cover
        print(f"  [OCR echec] {pdf_path.name} p.{page_number} : {exc}")
        return ""


def load_all(faq_dir: Path, pdf_dir: Path) -> list[dict]:
    docs = load_faq(faq_dir) + load_pdfs(pdf_dir)
    return docs
