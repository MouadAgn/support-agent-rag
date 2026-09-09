"""Le triage doit distinguer la politesse d'une vraie demande de support."""
import pytest

from support_agent.agent.triage import detecter_conversation
from support_agent.config import get_settings
from support_agent.cost.tracker import CostTracker


@pytest.mark.parametrize(
    "message, famille",
    [
        ("Salut", "salutation"),
        ("bonjour", "salutation"),
        ("Bonjour !", "salutation"),
        ("Bjr", "salutation"),
        ("Bonjour, comment ca va ?", "salutation"),
        ("merci", "remerciement"),
        ("Merci beaucoup !", "remerciement"),
        ("au revoir", "adieu"),
        ("Merci, bonne journee", "adieu"),
    ],
)
def test_politesse_detectee(message, famille):
    assert detecter_conversation(message) == famille


@pytest.mark.parametrize(
    "message",
    [
        "Quels sont les delais de livraison ?",
        "Bonjour, quels sont vos delais de livraison ?",  # politesse + demande
        "Merci de me dire comment retourner un article",  # "merci" mais vraie demande
        "Salut, mon colis est bloque",
        "",
    ],
)
def test_vraie_demande_non_confondue(message):
    assert detecter_conversation(message) is None


def test_une_salutation_ne_compte_aucune_economie():
    """Un "bonjour" n'aurait jamais genere de ticket : le compter comme une
    economie de 5 EUR gonflerait artificiellement le ROI."""
    r = CostTracker(get_settings()).report("smalltalk")
    assert r.human_cost == 0.0
    assert r.saving == 0.0
    assert r.llm_cost == 0.0  # aucun appel LLM n'a eu lieu
