from support_agent.config import get_settings
from support_agent.cost.tracker import CostTracker


def test_cout_answered_genere_une_economie():
    s = get_settings()
    t = CostTracker(s)
    t.add("generate", 500, 50)
    r = t.report("answered")
    assert r.input_tokens == 500 and r.output_tokens == 50
    assert r.llm_cost > 0
    # une reponse coute bien moins cher qu'un ticket humain
    assert r.llm_cost < s.cost_per_ticket
    assert r.saving > 0


def test_cout_escalade_utilise_le_prix_appel():
    s = get_settings()
    r = CostTracker(s).report("escalate")
    assert r.human_cost == s.cost_per_call


def test_prix_proportionnel_aux_tokens():
    s = get_settings()
    petit = CostTracker(s)
    petit.add("g", 100, 10)
    gros = CostTracker(s)
    gros.add("g", 10000, 1000)
    assert gros.report("answered").llm_cost > petit.report("answered").llm_cost
