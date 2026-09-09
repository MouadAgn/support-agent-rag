"""Etat partage qui circule entre les noeuds du graphe LangGraph."""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    question: str                     # question de l'utilisateur
    documents: list[dict]             # chunks recuperes (RAG)
    retrieval_score: float            # meilleure similarite trouvee
    relevant: bool                    # le contexte est-il pertinent ? (RAG quality)
    answer: str                       # reponse generee
    can_answer: bool                  # le LLM a-t-il pu repondre depuis le contexte ?
    decision: str                     # "answered" | "ticket" | "escalate" | "smalltalk"
    ticket: Optional[dict]            # ticket cree / routage appel
    cost: dict                        # rapport de cout
    trace: list[str]                  # journal des etapes (pour l'affichage)
    _tracker: Any                     # CostTracker (interne)
    _llm: Any                         # LLM (interne)
    _store: Any                       # VectorStore (interne)
    _settings: Any                    # Settings (interne)
