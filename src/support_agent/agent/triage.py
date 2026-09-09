"""Triage de l'intention, avant toute recherche documentaire.

Pourquoi ce module. Un agent de support ne recoit pas que des questions :
il recoit aussi des "bonjour", des "merci", des "au revoir". Sans triage,
ces messages traversent tout le pipeline RAG, declenchent un appel LLM
facture, et finissent en ticket ouvert pour rien.

On les traite donc en amont, de facon deterministe : aucun appel LLM,
aucune latence, et surtout aucun ticket parasite dans la file du support.

La regle : on retire du message les mots de politesse et les mots vides.
S'il ne reste rien, c'est de la conversation. S'il reste quelque chose,
c'est une demande — et elle part dans le RAG.

    "salut"                                  -> conversation
    "bonjour, comment ca va ?"               -> conversation
    "bonjour, quels sont vos delais ?"       -> demande (RAG)
"""
from __future__ import annotations

import re
import unicodedata

# Mots de politesse, par famille : chacune a sa reponse.
SALUTATIONS = {
    "bonjour", "salut", "bonsoir", "coucou", "hello", "hey", "hi", "yo",
    "slt", "bjr", "bsr", "cc", "allo", "re", "salu", "wesh", "bonjr",
    # "ca va" et son abreviation SMS : une prise de contact, pas une demande
    "cava", "cv",
}
REMERCIEMENTS = {"merci", "mercis", "thanks", "thank", "thx", "mci"}

# Acquiescements : le client valide, il n'attend pas de recherche documentaire.
ACQUIESCEMENTS = {
    "ok", "oki", "okay", "dac", "dacord", "daccord", "cool", "super", "top",
    "genial", "parfait", "nickel", "impeccable", "impec", "tresbien",
    "camarche", "bien", "compris", "entendu",
}
ADIEUX = {
    "aurevoir", "revoir", "bye", "ciao", "adieu",
    "bonnejournee", "bonnesoiree", "bonnenuit", "abientot", "aplus", "ademain",
}

# Locutions en plusieurs mots : le decoupage les separerait et on perdrait
# le sens ("bonne journee" deviendrait "bonne" + "journee", deux mots vides).
LOCUTIONS = {
    ("bonne", "journee"): "bonnejournee",
    ("bonne", "soiree"): "bonnesoiree",
    ("bonne", "nuit"): "bonnenuit",
    ("au", "revoir"): "aurevoir",
    ("a", "bientot"): "abientot",
    ("a", "plus"): "aplus",
    ("a", "demain"): "ademain",
    ("ca", "va"): "cava",
    ("ca", "marche"): "camarche",
    ("tres", "bien"): "tresbien",
    ("d", "accord"): "daccord",
}

POLITESSE = SALUTATIONS | REMERCIEMENTS | ACQUIESCEMENTS | ADIEUX

# Mots vides : sans valeur informative, ils ne suffisent pas a faire une demande.
MOTS_VIDES = {
    "a", "au", "aux", "beaucoup", "bien", "bonne", "ca", "cava", "ce", "comment",
    "d", "de", "des", "du", "en", "est", "et", "il", "j", "je", "journee", "l",
    "la", "le", "les", "m", "madame", "mademoiselle", "moi", "monsieur", "n",
    "ok", "on", "ou", "plait", "pour", "qui", "s", "sa", "se", "soiree", "sont",
    "stp", "suis", "sva", "svp", "t", "toi", "tous", "tout", "tres", "tu", "un",
    "une", "va", "vas", "vous", "y", "ete", "allez", "vais", "ai",
}

# Au-dela de cette longueur, on considere qu'il y a forcement une demande.
MAX_MOTS_CONVERSATION = 8

REPONSES = {
    "salutation": (
        "Bonjour ! Je suis l'assistant de support. Je peux vous renseigner sur "
        "les commandes, la livraison, les retours, les paiements et votre compte. "
        "Quelle est votre question ?"
    ),
    "remerciement": (
        "Avec plaisir. N'hesitez pas si vous avez une autre question."
    ),
    "acquiescement": (
        "Tres bien. Je reste a votre disposition si vous avez une question."
    ),
    "adieu": (
        "Bonne journee ! Je reste disponible si vous avez besoin d'aide."
    ),
}


def _normaliser(texte: str) -> list[str]:
    """Minuscules, sans accents, sans ponctuation -> liste de mots."""
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    mots = [m for m in re.split(r"[^a-z0-9]+", texte) if m]

    # regroupe les locutions ("bonne journee" -> "bonnejournee")
    fusionnes: list[str] = []
    i = 0
    while i < len(mots):
        paire = (mots[i], mots[i + 1]) if i + 1 < len(mots) else None
        if paire in LOCUTIONS:
            fusionnes.append(LOCUTIONS[paire])
            i += 2
        else:
            fusionnes.append(mots[i])
            i += 1
    return fusionnes


def detecter_conversation(question: str) -> str | None:
    """Renvoie la famille de politesse ("salutation", "remerciement",
    "acquiescement", "adieu") si le message n'est que de la conversation,
    sinon None.
    """
    mots = _normaliser(question)
    if not mots or len(mots) > MAX_MOTS_CONVERSATION:
        return None

    # Un mot de politesse au moins, et rien d'autre qui porte du sens.
    familles: list[str] = []
    for mot in mots:
        if mot in SALUTATIONS:
            familles.append("salutation")
        elif mot in REMERCIEMENTS:
            familles.append("remerciement")
        elif mot in ACQUIESCEMENTS:
            familles.append("acquiescement")
        elif mot in ADIEUX:
            familles.append("adieu")
        elif mot not in MOTS_VIDES:
            return None  # un mot porteur de sens -> c'est une vraie demande

    if not familles:
        return None
    # "merci, au revoir" -> on repond sur la derniere intention exprimee
    return familles[-1]


def reponse_conversation(famille: str) -> str:
    return REPONSES.get(famille, REPONSES["salutation"])
