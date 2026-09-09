from support_agent.ingestion.vectorstore import VectorStore


def _chunks():
    return [
        {"id": "1", "text": "Les delais de livraison sont de 3 a 5 jours ouvres.",
         "metadata": {"source": "faq.md"}},
        {"id": "2", "text": "Vous avez 30 jours pour retourner un article.",
         "metadata": {"source": "faq.md"}},
        {"id": "3", "text": "Nous acceptons Visa, Mastercard et PayPal.",
         "metadata": {"source": "faq.md"}},
    ]


def test_search_retourne_le_bon_chunk():
    store = VectorStore.build(_chunks(), "tfidf", "n/a")
    hits = store.search("combien de temps pour la livraison", k=2)
    assert hits
    top_chunk, top_score = hits[0]
    assert "livraison" in top_chunk["text"].lower()
    assert 0.0 <= top_score <= 1.0001


def test_save_load_roundtrip(tmp_path):
    store = VectorStore.build(_chunks(), "tfidf", "n/a")
    store.save(tmp_path)
    assert VectorStore.exists(tmp_path)
    reloaded = VectorStore.load(tmp_path)
    # backend TF-IDF (lexical) : la requete partage le vocabulaire de la cible
    hits = reloaded.search("payer avec Visa Mastercard ou PayPal", k=1)
    assert "paypal" in hits[0][0]["text"].lower()
