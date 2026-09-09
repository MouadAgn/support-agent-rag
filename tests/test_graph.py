"""Tests bout-en-bout du graphe LangGraph (en mode demo)."""
import pytest

from support_agent.config import get_settings
from support_agent.ingestion.chunking import chunk_documents
from support_agent.ingestion.loaders import load_all
from support_agent.ingestion.vectorstore import VectorStore
from support_agent.agent.graph import SupportAgent


@pytest.fixture(scope="module")
def agent(tmp_path_factory):
    s = get_settings()
    docs = load_all(s.faq_dir, s.pdf_dir)
    chunks = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
    index_dir = tmp_path_factory.mktemp("index")
    VectorStore.build(chunks, "tfidf", "n/a").save(index_dir)
    s.index_dir = index_dir
    return SupportAgent(s)


def test_question_faq_est_repondue(agent):
    r = agent.answer("Quels sont les delais de livraison ?")
    assert r["cost"]["resolution"] == "answered"
    assert r["cost"]["llm_cost_eur"] >= 0
    assert r["documents"]


def test_hors_sujet_ouvre_un_ticket(agent):
    r = agent.answer("Quelle est la capitale de l'Australie ?")
    assert r["cost"]["resolution"] == "ticket"
    assert r["ticket"]["type"] == "ticket"


def test_urgence_escalade_vers_appel(agent):
    r = agent.answer("Paiement bloque, c'est urgent, je veux un remboursement, litige")
    assert r["cost"]["resolution"] == "escalate"
    assert r["ticket"]["channel"] == "centre-appel"


def test_la_trace_couvre_les_etapes(agent):
    r = agent.answer("Comment retourner un article ?")
    trace = " ".join(r["trace"])
    assert "retrieve" in trace and "grade" in trace and "finalize" in trace
