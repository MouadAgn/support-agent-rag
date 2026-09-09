from support_agent.ingestion.chunking import chunk_documents, chunk_text


def test_short_text_reste_un_seul_chunk():
    assert chunk_text("phrase courte", 800, 120) == ["phrase courte"]


def test_texte_long_est_decoupe():
    texte = "\n\n".join(f"Paragraphe {i} " + "mot " * 60 for i in range(6))
    chunks = chunk_text(texte, 300, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 50 + 5 for c in chunks)


def test_faq_non_redecoupee():
    docs = [{"text": "Q x " * 400, "metadata": {"source": "faq.md", "type": "faq"}}]
    chunks = chunk_documents(docs, 100, 20)
    assert len(chunks) == 1  # une FAQ = une unite


def test_chunk_a_un_id_et_metadata():
    docs = [{"text": "bonjour", "metadata": {"source": "doc.pdf", "type": "pdf", "page": 2}}]
    chunks = chunk_documents(docs, 800, 120)
    assert chunks[0]["id"].startswith("doc.pdf::2::")
    assert chunks[0]["metadata"]["source"] == "doc.pdf"
