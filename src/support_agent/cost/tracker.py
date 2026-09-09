"""Suivi des couts par reponse.

Objectif metier : montrer combien coute une reponse de l'agent (tokens LLM)
et la comparer au cout d'un traitement humain (ticket ou appel), pour
chiffrer l'economie realisee en deviant la sollicitation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Settings


@dataclass
class Usage:
    """Un appel LLM : combien de tokens en entree / sortie."""

    step: str
    input_tokens: int
    output_tokens: int


@dataclass
class CostReport:
    input_tokens: int
    output_tokens: int
    llm_cost: float          # cout des tokens (EUR)
    resolution: str          # "answered" | "ticket" | "escalate"
    human_cost: float        # cout humain evite (ou engage) selon resolution
    saving: float            # economie nette (EUR)
    calls: list[Usage] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "llm_cost_eur": round(self.llm_cost, 6),
            "resolution": self.resolution,
            "human_cost_eur": round(self.human_cost, 2),
            "saving_eur": round(self.saving, 4),
        }


class CostTracker:
    def __init__(self, settings: Settings):
        self.s = settings
        self.calls: list[Usage] = []

    def add(self, step: str, input_tokens: int, output_tokens: int) -> None:
        self.calls.append(Usage(step, input_tokens, output_tokens))

    def _llm_cost(self, in_tok: int, out_tok: int) -> float:
        return (
            in_tok / 1_000_000 * self.s.price_input_per_m
            + out_tok / 1_000_000 * self.s.price_output_per_m
        )

    def report(self, resolution: str) -> CostReport:
        in_tok = sum(c.input_tokens for c in self.calls)
        out_tok = sum(c.output_tokens for c in self.calls)
        llm_cost = self._llm_cost(in_tok, out_tok)

        if resolution == "answered":
            # on a evite un ticket humain -> economie = cout ticket - cout LLM
            human_cost = self.s.cost_per_ticket
            saving = human_cost - llm_cost
        elif resolution == "ticket":
            # l'agent a pre-qualifie puis ouvert un ticket : cout LLM + ticket
            human_cost = self.s.cost_per_ticket
            saving = -llm_cost
        else:  # escalate -> appel
            human_cost = self.s.cost_per_call
            saving = -llm_cost

        return CostReport(
            input_tokens=in_tok,
            output_tokens=out_tok,
            llm_cost=llm_cost,
            resolution=resolution,
            human_cost=human_cost,
            saving=saving,
            calls=list(self.calls),
        )
