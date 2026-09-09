"""Outils (tools) que l'agent peut declencher quand il ne repond pas lui-meme.

Ce sont de vrais outils LangChain (@tool) : dans un vrai SI on brancherait ici
l'API du helpdesk (Zendesk, Jira Service Management...) ou la telephonie du
centre d'appel. Ici on simule en journalisant l'action et en renvoyant un
identifiant, ce qui suffit a demontrer le routage.
"""
from __future__ import annotations

import random
import re
from datetime import datetime

from langchain_core.tools import tool

# Mots qui signalent l'urgence / un litige -> plutot un appel qu'un ticket
URGENT_PATTERNS = [
    r"\burgent\b", r"\blitige\b", r"\breclamation\b", r"\brembours",
    r"\bbloqu", r"\bfraud", r"\bpiratage\b", r"\bimpossible de me connecter\b",
    r"\bje n'arrive plus\b", r"\bplainte\b", r"\bavocat\b",
]


@tool
def create_ticket(question: str, category: str = "general") -> dict:
    """Cree un ticket de support pour un conseiller (traitement asynchrone).

    A utiliser quand l'agent ne peut pas repondre mais que la demande n'est
    pas urgente.
    """
    ticket_id = f"TCK-{datetime.now():%Y%m%d}-{random.randint(1000, 9999)}"
    return {
        "type": "ticket",
        "id": ticket_id,
        "category": category,
        "question": question,
        "channel": "helpdesk",
    }


@tool
def escalate_to_agent(question: str, reason: str = "urgence") -> dict:
    """Escalade vers un teleconseiller (centre d'appel) pour un traitement
    immediat. A utiliser pour les demandes urgentes ou sensibles."""
    call_id = f"CALL-{datetime.now():%Y%m%d}-{random.randint(1000, 9999)}"
    return {
        "type": "escalate",
        "id": call_id,
        "reason": reason,
        "question": question,
        "channel": "centre-appel",
    }


def looks_urgent(question: str) -> bool:
    q = question.lower()
    return any(re.search(p, q) for p in URGENT_PATTERNS)
