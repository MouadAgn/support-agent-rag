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


def test_les_ids_sont_uniques_entre_entrees_de_faq():
    """Regression : toutes les entrees de FAQ partageaient "faq.md::0::0",
    car l'id ne contenait ni la page (absente) ni l'indice du document."""
    docs = [
        {"text": f"Question : Q{i}\nReponse : R{i}",
         "metadata": {"source": "faq.md", "type": "faq", "question": f"Q{i}"}}
        for i in range(5)
    ]
    ids = [c["id"] for c in chunk_documents(docs, 800, 120)]
    assert len(set(ids)) == len(ids) == 5


def test_faq_indexee_sur_la_question_mais_contexte_complet():
    """On indexe la question seule (recherche plus precise) tout en gardant
    la reponse complete comme contexte pour le LLM."""
    docs = [{
        "text": "Question : Quels sont les delais ?\nReponse : 3 a 5 jours ouvres.",
        "metadata": {"source": "faq.md", "type": "faq",
                     "question": "Quels sont les delais ?"},
    }]
    chunk = chunk_documents(docs, 800, 120)[0]
    assert chunk["embed_text"] == "Quels sont les delais ?"
    assert "3 a 5 jours ouvres" in chunk["text"]


def test_pdf_indexe_sur_son_propre_texte():
    """Hors FAQ, il n'y a pas de question : on indexe le texte lui-meme."""
    docs = [{"text": "Article 4 : le retour est gratuit.",
             "metadata": {"source": "cgv.pdf", "type": "pdf", "page": 1}}]
    chunk = chunk_documents(docs, 800, 120)[0]
    assert chunk["embed_text"] == chunk["text"]
