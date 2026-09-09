"""Le triage doit distinguer la conversation d'une vraie demande de support.

Les deux sens comptent, mais pas de la meme facon. Rater un "bonjour" coute
un appel LLM inutile. Classer "mon colis est bloque" comme de la politesse
coute un client jamais traite. Le second jeu de tests est donc le plus
important : il verifie qu'un seul mot metier suffit a basculer dans le RAG.
"""
import pytest

from support_agent.agent.triage import (
    IDENTITE,
    MOTS_OUTILS,
    POLITESSE,
    detecter_conversation,
    reponse_conversation,
)
from support_agent.config import get_settings
from support_agent.cost.tracker import CostTracker


# --------------------------------------------------------------------------- #
#  1. Conversation reconnue                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message",
    [
        "Bonjour", "bonjour !", "Bonsoir", "Salut", "salut !", "slt", "Bjr",
        "bjour", "cc", "Coucou", "Hello", "hey", "Hi", "yo", "wesh", "hola",
        "Allo ?", "re", "rebonjour", "Enchante",
        "cv", "cv ?", "ca va", "Ca va ?", "ca va bien ?",
        "Bonjour Madame", "bonjour, comment ca va ?",
        "Bonjour, j'ai une petite question",
        "bonjour, j'aurais besoin d'un renseignement",
    ],
)
def test_salutations(message):
    assert detecter_conversation(message) == "salutation"


@pytest.mark.parametrize(
    "message",
    [
        "merci", "Merci !", "mercii", "merciiii", "Merci beaucoup",
        "merci beaucoup !", "Mille mercis", "merci infiniment",
        "Je vous remercie", "thanks", "thank you", "thx", "mrc",
    ],
)
def test_remerciements(message):
    assert detecter_conversation(message) == "remerciement"


@pytest.mark.parametrize(
    "message",
    [
        "ok", "OK !", "okay", "kk", "d'accord", "dac", "daccord",
        "super", "Nickel", "parfait", "top", "cool", "excellent",
        "tres bien", "ca marche", "ca roule", "Vu", "entendu", "compris",
        "c'est note", "impeccable",
    ],
)
def test_acquiescements(message):
    assert detecter_conversation(message) == "acquiescement"


@pytest.mark.parametrize(
    "message",
    ["pardon", "Desole", "desolee", "Excusez-moi", "excuse moi", "oups",
     "sorry", "au temps pour moi"],
)
def test_excuses(message):
    assert detecter_conversation(message) == "excuse"


@pytest.mark.parametrize(
    "message",
    [
        "au revoir", "Au revoir !", "bye", "byebye", "ciao", "adieu",
        "bonne journee", "Bonne soiree", "bonne nuit", "bonne continuation",
        "a bientot", "a demain", "a plus", "a la prochaine",
        "Merci, bonne journee", "merci beaucoup et bonne journee a vous",
    ],
)
def test_adieux(message):
    assert detecter_conversation(message) == "adieu"


@pytest.mark.parametrize(
    "message",
    [
        "qui es-tu ?", "Qui etes-vous ?", "tu es qui ?",
        "t'es un robot ?", "Tu es un robot ?", "es-tu un robot ?",
        "tu es une IA ?", "tu es humain ?", "je parle a un humain ?",
        "comment tu t'appelles ?", "quel est ton nom ?",
        "que peux-tu faire ?", "tu peux faire quoi ?", "tu sers a quoi ?",
        "tu peux m'aider ?", "tu parles francais ?", "comment tu fonctionnes ?",
        # la politesse en tete ou en queue ne doit pas empecher la detection
        "Bonjour, qui es-tu ?", "salut, tu peux m'aider ?",
        "qui es-tu ? merci",
    ],
)
def test_identite(message):
    assert detecter_conversation(message) == "identite"


@pytest.mark.parametrize("message", ["\U0001F44D", "???", "!!!", "...", ":)"])
def test_message_sans_mot_demande_une_reformulation(message):
    assert detecter_conversation(message) == "indetermine"


@pytest.mark.parametrize("message", ["", "   ", "\n"])
def test_message_vide_nest_pas_de_la_conversation(message):
    assert detecter_conversation(message) is None


# --------------------------------------------------------------------------- #
#  2. Vraies demandes : JAMAIS classees comme conversation                    #
#     C'est le jeu de tests critique.                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "message",
    [
        # demandes nues
        "Quels sont les delais de livraison ?",
        "Comment retourner un article ?",
        "ou est mon colis ?",
        "mon compte est bloque",
        "je veux un remboursement",
        "c'est combien les frais de port ?",
        # politesse + demande : la demande doit gagner
        "Bonjour, quels sont vos delais de livraison ?",
        "Salut, mon colis est bloque",
        "bonjour j'ai besoin d'aide pour ma commande",
        "Merci de me dire comment retourner un article",
        "ok mais mon colis est ou ?",
        "d'accord et pour le remboursement ?",
        "pardon, je me suis trompe de taille",
        "au revoir, mais avant : ou est ma facture ?",
        # pieges lexicaux : contiennent une formule de conversation
        "ca va me couter combien ?",
        "est-ce que ca va etre livre demain ?",
        "ca marche pas votre site",
        "comment ca marche le paiement en 3 fois ?",
        "tu peux m'aider a retourner un article ?",
        "qui est le PDG de la societe ?",
        "c'est bon pour la garantie apres 2 ans ?",
        # hors sujet : doit aller au RAG puis en ticket, pas en accueil
        "quel est le chiffre d'affaires de votre entreprise ?",
        "est-ce que vous recrutez des developpeurs ?",
    ],
)
def test_vraie_demande_part_dans_le_rag(message):
    assert detecter_conversation(message) is None


def test_un_message_long_est_toujours_une_demande():
    """Au-dela de la limite de mots, on ne prend pas le risque de classer
    un message comme de la politesse."""
    message = "bonjour " * 20
    assert detecter_conversation(message) is None


# --------------------------------------------------------------------------- #
#  3. Coherence du vocabulaire                                                #
# --------------------------------------------------------------------------- #
def test_aucun_mot_metier_dans_les_mots_outils():
    """Un mot metier parmi les mots outils rendrait une vraie demande
    invisible : c'est le pire scenario possible pour ce module."""
    metier = {
        "colis", "commande", "commandes", "livraison", "livrer", "retour",
        "retourner", "facture", "remboursement", "rembourser", "paiement",
        "payer", "carte", "compte", "article", "taille", "stock", "garantie",
        "promo", "code", "adresse", "probleme", "souci", "bloque", "casse",
        "endommage", "manque", "urgent", "litige", "plainte",
    }
    assert not (MOTS_OUTILS & metier)


def test_les_familles_ne_se_chevauchent_pas_avec_les_mots_outils():
    """Un mot de politesse aussi present dans les mots outils serait ignore
    par la boucle de detection, sauf a compter sur l'ordre des tests."""
    chevauchement = POLITESSE & MOTS_OUTILS
    assert chevauchement == set(), chevauchement


def test_chaque_famille_a_une_reponse():
    for message, famille in [
        ("bonjour", "salutation"), ("merci", "remerciement"),
        ("ok", "acquiescement"), ("pardon", "excuse"),
        ("qui es-tu ?", "identite"), ("au revoir", "adieu"),
        ("???", "indetermine"),
    ]:
        assert detecter_conversation(message) == famille
        assert len(reponse_conversation(famille)) > 20


def test_les_formulations_didentite_sont_normalisees():
    """Les entrees d'IDENTITE sont comparees a du texte normalise : elles
    doivent donc etre en minuscules, sans accent ni ponctuation."""
    for phrase in IDENTITE:
        assert phrase == phrase.lower()
        assert all(c.isalnum() or c == " " for c in phrase), phrase


# --------------------------------------------------------------------------- #
#  4. Impact sur les couts                                                    #
# --------------------------------------------------------------------------- #
def test_une_salutation_ne_compte_aucune_economie():
    """Un bonjour n'aurait jamais genere de ticket : le compter comme une
    economie de 5 EUR gonflerait artificiellement le ROI."""
    r = CostTracker(get_settings()).report("smalltalk")
    assert r.human_cost == 0.0
    assert r.saving == 0.0
    assert r.llm_cost == 0.0  # aucun appel LLM n'a eu lieu
