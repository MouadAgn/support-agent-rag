"""Assemblage du graphe LangGraph et point d'entree de l'agent.

Flux :

    START -> triage -> [vraie demande ?]
                          |-- non (bonjour, merci...) ----------> finalize -> END
                          '-- oui --> retrieve -> grade -> [pertinent ?]
                                                   |-- oui --> generate -> [a repondu ?]
                                                   |              |-- oui --> finalize -> END
                                                   |              '-- non --> handle_fallback
                                                   '-- non ---------------> handle_fallback -> finalize -> END
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
    g.add_node("triage", nodes.triage)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("grade", nodes.grade)
    g.add_node("generate", nodes.generate)
    g.add_node("handle_fallback", nodes.handle_fallback)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "triage")
    g.add_conditional_edges(
        "triage", nodes.route_after_triage,
        {"retrieve": "retrieve", "finalize": "finalize"},
    )
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

    def _initial_state(self, question: str) -> AgentState:
        return {
            "question": question,
            "trace": [],
            "_settings": self.settings,
            "_store": self.store,
            "_llm": self.llm,
            "_tracker": CostTracker(self.settings),
        }

    def answer(self, question: str) -> AgentState:
        return self.graph.invoke(self._initial_state(question))

    def stream(self, question: str):
        """Execute le graphe noeud par noeud et emet la mise a jour de chaque
        etape des qu'elle est calculee. Utilise par l'interface web pour
        afficher le pipeline en direct."""
        for update in self.graph.stream(self._initial_state(question),
                                        stream_mode="updates"):
            for node, partial in update.items():
                yield node, (partial or {})
