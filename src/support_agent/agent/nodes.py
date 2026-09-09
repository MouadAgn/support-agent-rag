"""Les noeuds du graphe. Chaque noeud fait UNE chose et met a jour l'etat."""
from __future__ import annotations

from .prompts import SYSTEM_PROMPT, build_user_prompt
from .state import AgentState
from .tools import create_ticket, escalate_to_agent, looks_urgent


def retrieve(state: AgentState) -> dict:
    """RAG - etape 1 : retrouver les passages les plus proches de la question."""
    store = state["_store"]
    settings = state["_settings"]
    hits = store.search(state["question"], k=settings.top_k)
    documents = [{**chunk, "score": score} for chunk, score in hits]
    best = documents[0]["score"] if documents else 0.0
    trace = state.get("trace", []) + [
        f"retrieve : {len(documents)} passage(s), meilleure similarite = {best:.3f}"
    ]
    return {"documents": documents, "retrieval_score": best, "trace": trace}


def grade(state: AgentState) -> dict:
    """RAG quality : le contexte est-il assez pertinent pour tenter une reponse ?

    Garde-fou base sur le score de similarite. En dessous du seuil, on ne
    laisse pas le LLM halluciner : on bascule vers un humain.
    """
    settings = state["_settings"]
    relevant = bool(state["documents"]) and state["retrieval_score"] >= settings.score_threshold
    trace = state.get("trace", []) + [
        f"grade : contexte {'PERTINENT' if relevant else 'INSUFFISANT'} "
        f"(seuil = {settings.score_threshold})"
    ]
    return {"relevant": relevant, "trace": trace}


def generate(state: AgentState) -> dict:
    """Genere la reponse a partir du contexte, via le LLM (DeepSeek ou simule)."""
    llm = state["_llm"]
    tracker = state["_tracker"]

    context = "\n\n---\n\n".join(
        f"[{d['metadata'].get('source')}] {d['text']}" for d in state["documents"]
    )
    user_prompt = build_user_prompt(state["question"], context)
    result = llm.chat(SYSTEM_PROMPT, user_prompt)
    tracker.add("generate", result.input_tokens, result.output_tokens)

    can_answer = "INSUFFISANT" not in result.text.upper()
    decision = "answered" if can_answer else None
    trace = state.get("trace", []) + [
        f"generate : {result.input_tokens} tok in / {result.output_tokens} tok out "
        f"-> {'reponse' if can_answer else 'contexte insuffisant'}"
    ]
    out = {"answer": result.text, "can_answer": can_answer, "trace": trace}
    if decision:
        out["decision"] = decision
    return out


def handle_fallback(state: AgentState) -> dict:
    """L'agent ne peut pas repondre : il choisit un OUTIL (ticket ou appel)."""
    question = state["question"]
    if looks_urgent(question):
        ticket = escalate_to_agent.invoke({"question": question, "reason": "urgence/litige"})
        decision = "escalate"
        msg = (
            "Je transmets votre demande a un teleconseiller qui va vous "
            f"rappeler (reference {ticket['id']})."
        )
    else:
        ticket = create_ticket.invoke({"question": question, "category": "general"})
        decision = "ticket"
        msg = (
            "Je n'ai pas la reponse exacte, j'ouvre un ticket pour un "
            f"conseiller (reference {ticket['id']}). Vous recevrez un retour."
        )
    trace = state.get("trace", []) + [f"fallback : outil '{decision}' -> {ticket['id']}"]
    return {"decision": decision, "ticket": ticket, "answer": msg, "trace": trace}


def finalize(state: AgentState) -> dict:
    """Calcule le rapport de cout de la reponse."""
    tracker = state["_tracker"]
    report = tracker.report(state.get("decision", "answered"))
    trace = state.get("trace", []) + [
        f"finalize : cout LLM = {report.llm_cost:.5f} EUR, "
        f"resolution = {report.resolution}, economie = {report.saving:.3f} EUR"
    ]
    return {"cost": report.as_dict(), "trace": trace}


# --------------------------------------------------------------------------- #
#  Fonctions d'aiguillage (conditional edges)                                 #
# --------------------------------------------------------------------------- #
def route_after_grade(state: AgentState) -> str:
    return "generate" if state.get("relevant") else "handle_fallback"


def route_after_generate(state: AgentState) -> str:
    return "finalize" if state.get("can_answer") else "handle_fallback"
