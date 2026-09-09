"""Assemblage du graphe LangGraph et point d'entree de l'agent.

Flux :

    START -> retrieve -> grade -> [pertinent ?]
                                      |-- oui --> generate -> [a repondu ?]
                                      |                          |-- oui --> finalize -> END
                                      |                          '-- non --> handle_fallback
                                      '-- non ------------------------------> handle_fallback -> finalize -> END
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..config import Settings, get_settings
from ..cost.tracker import CostTracker
from ..llm import get_llm
from ..ingestion.vectorstore import VectorStore
from . import nodes
from .state import AgentState


def build_graph():
    """Construit et compile le graphe (structure pure, sans dependances)."""
    g = StateGraph(AgentState)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade)
    g.add_node("generate", nodes.generate)
    g.add_node("handle_fallback", nodes.handle_fallback)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade", nodes.route_after_grade,
        {"generate": "generate", "handle_fallback": "handle_fallback"},
    )
    g.add_conditional_edges(
        "generate", nodes.route_after_generate,
        {"finalize": "finalize", "handle_fallback": "handle_fallback"},
    )
    g.add_edge("handle_fallback", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


class SupportAgent:
    """Interface haut niveau : charge l'index + le LLM, et repond aux questions."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not VectorStore.exists(self.settings.index_dir):
            raise FileNotFoundError(
                "Index introuvable. Lance d'abord l'ingestion :\n"
                "  python scripts/ingest.py"
            )
        self.store = VectorStore.load(self.settings.index_dir)
        self.llm = get_llm(self.settings)
        self.graph = build_graph()

    def answer(self, question: str) -> AgentState:
        state: AgentState = {
            "question": question,
            "trace": [],
            "_settings": self.settings,
            "_store": self.store,
            "_llm": self.llm,
            "_tracker": CostTracker(self.settings),
        }
        return self.graph.invoke(state)
