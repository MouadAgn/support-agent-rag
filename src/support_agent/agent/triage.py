"""Triage de l'intention, avant toute recherche documentaire.

Pourquoi ce module. Un agent de support ne recoit pas que des questions :
il recoit aussi des "bonjour", des "merci", des "ok", des "cv ?", des
"pardon", et des "t'es un robot ?". Sans triage, ces messages traversent
tout le pipeline RAG, declenchent un appel LLM facture, et finissent en
ticket ouvert pour rien — ce qui pollue la file du support.

On les traite donc en amont, de facon deterministe : aucun appel LLM,
aucune latence, aucun ticket parasite.

LA REGLE. On retire du message les mots de politesse et les mots outils.
S'il ne reste rien, c'est de la conversation. S'il reste un seul mot
porteur de sens, c'est une demande — et elle part dans le RAG.

    "salut"                                  -> conversation (salutation)
    "cv ?"                                   -> conversation (salutation)
    "ok nickel"                              -> conversation (acquiescement)
    "t'es un robot ?"                        -> conversation (identite)
    "bonjour, comment ca va ?"               -> conversation (salutation)
    "bonjour, quels sont vos delais ?"       -> demande      (RAG)
    "ok mais mon colis est ou ?"             -> demande      (RAG)

Ce sens de lecture est important : le risque a eviter n'est pas de rater
un "bonjour", c'est de classer une vraie demande comme de la politesse et
de ne jamais la traiter. D'ou le fait qu'un seul mot porteur de sens
suffise a basculer dans le RAG.
"""
from __future__ import annotations

import re
import unicodedata

# --------------------------------------------------------------------------- #
#  Familles de politesse : chacune a sa reponse                               #
# --------------------------------------------------------------------------- #
SALUTATIONS = {
    # formes standard
    "bonjour", "bonjours", "bonsoir", "salut", "salutations", "coucou",
    "enchante", "enchantee",
    # abreviations et SMS
    "bjr", "bjour", "bonjr", "bonj", "bsr", "bsoir", "slt", "salu", "cc",
    "kikou", "kikoo",
    # emprunts a l'anglais, courants dans les messages clients
    "hello", "helo", "hallo", "hey", "hei", "hi", "hii", "yo", "yop", "hola",
    # registre familier
    "wesh", "wsh",
    # prise de contact
    "allo", "alo", "re", "rebonjour", "rebjr",
    # "ca va" et son abreviation : une prise de contact, pas une demande
    "cava", "cv", "cvt",
}

REMERCIEMENTS = {
    "merci", "mercis", "mercii", "mrc", "mci", "remercie", "remerciements",
    "thanks", "thank", "thx", "tks", "gracias", "grazie",
}

# Le client valide ce qui vient d'etre dit : il n'attend pas de recherche.
ACQUIESCEMENTS = {
    "ok", "oki", "okay", "okey", "kk", "dac", "dacc", "dacord", "daccord",
    "compris", "entendu", "note", "notee", "vu", "recu",
    "cool", "super", "top", "genial", "geniale", "parfait", "parfaite",
    "nickel", "impeccable", "impec", "excellent", "excellente", "formidable",
    "chouette", "bien", "bon", "tresbien", "camarche", "cestbon", "cameva",
}

ADIEUX = {
    "aurevoir", "revoir", "adieu", "bye", "byebye", "ciao", "tchao", "tchuss",
    "bonnejournee", "bonnesoiree", "bonnenuit", "bonneapresmidi",
    "abientot", "aplus", "ademain", "aplustard", "alaprochaine",
    "bonnecontinuation",
}

EXCUSES = {
    "pardon", "desole", "desolee", "excuse", "excuses", "excusez", "sorry",
    "oups", "autant", "mautant",
}

POLITESSE = SALUTATIONS | REMERCIEMENTS | ACQUIESCEMENTS | ADIEUX | EXCUSES

# --------------------------------------------------------------------------- #
#  Locutions : le decoupage les separerait et on perdrait le sens             #
#  ("bonne journee" deviendrait "bonne" + "journee", deux mots outils)        #
# --------------------------------------------------------------------------- #
LOCUTIONS = {
    ("bonne", "journee"): "bonnejournee",
    ("bonne", "soiree"): "bonnesoiree",
    ("bonne", "nuit"): "bonnenuit",
    ("bonne", "apres", "midi"): "bonneapresmidi",
    ("bonne", "fin", "de"): "bonnejournee",
    ("a", "la", "prochaine"): "alaprochaine",
    ("bonne", "continuation"): "bonnecontinuation",
    ("au", "revoir"): "aurevoir",
    ("a", "bientot"): "abientot",
    ("a", "plus"): "aplus",
    ("a", "demain"): "ademain",
    ("a", "tantot"): "abientot",
    ("ca", "va"): "cava",
    ("ca", "marche"): "camarche",
    ("ca", "roule"): "camarche",
    ("tres", "bien"): "tresbien",
    ("d", "accord"): "daccord",
    ("au", "temps"): "autant",
    ("mille", "mercis"): "merci",
    ("merci", "infiniment"): "merci",
}

# --------------------------------------------------------------------------- #
#  Questions sur l'agent lui-meme                                             #
#                                                                              #
#  "t'es un robot ?" n'est pas une demande de support, mais ce n'est pas non   #
#  plus de la politesse : ces phrases sont faites uniquement de mots outils,   #
#  donc la regle generale ne peut pas les attraper. On les liste donc          #
#  explicitement, comparees a la phrase entiere pour ne pas confondre avec     #
#  "tu peux m'aider a retourner un article ?".                                 #
# --------------------------------------------------------------------------- #
IDENTITE = {
    "qui es tu", "qui etes vous", "tu es qui", "vous etes qui", "c est qui",
    "tu es un robot", "es tu un robot", "t es un robot", "vous etes un robot",
    "tu es une ia", "es tu une ia", "t es une ia", "tu es une machine",
    "tu es un bot", "tu es un chatbot", "tu es humain", "es tu humain",
    "es tu une personne", "je parle a un humain", "je parle a un robot",
    "tu es une intelligence artificielle", "tu es un programme",
    "comment tu t appelles", "comment t appelles tu", "quel est ton nom",
    "tu t appelles comment", "c est quoi ton nom",
    "que sais tu faire", "que peux tu faire", "tu peux faire quoi",
    "tu sais faire quoi", "tu sers a quoi", "a quoi tu sers",
    "tu peux m aider", "tu peux maider", "vous pouvez m aider",
    "tu parles francais", "tu comprends le francais",
    "comment ca marche", "comment tu fonctionnes",
}

# --------------------------------------------------------------------------- #
#  Mots outils : sans valeur informative, ils ne suffisent pas a faire une     #
#  demande. On n'y met JAMAIS de mot metier (colis, commande, livraison,       #
#  facture, remboursement...) : ce sont eux qui doivent declencher le RAG.     #
# --------------------------------------------------------------------------- #
MOTS_OUTILS = {
    # articles, pronoms, prepositions
    "a", "au", "aux", "c", "ce", "ces", "cet", "cette", "d", "de", "des", "du",
    "en", "et", "il", "ils", "j", "je", "l", "la", "le", "les", "leur", "lui",
    "m", "ma", "me", "mes", "moi", "mon", "n", "ne", "nos", "notre", "nous",
    "on", "ou", "par", "pour", "que", "qui", "quoi", "s", "sa", "se", "ses",
    "son", "t", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos",
    "votre", "vous", "y",
    # auxiliaires et verbes vides
    "ai", "aie", "as", "est", "es", "etais", "etait", "ete", "etes", "sont",
    "suis", "sera", "serait", "va", "vas", "vais", "vont", "avez", "avons",
    "allez", "allons", "peux", "peut", "pouvez", "veux", "veut", "voudrais",
    "aimerais", "souhaite", "souhaiterais", "fait", "faire", "dire",
    "aurais", "aurai", "aura", "auriez", "avais", "avait", "serais",
    "you", "your", "me",
    # formules de politesse et remplissage
    "beaucoup", "bonne", "cher", "chere", "chers", "encore", "hein", "juste",
    "madame", "mademoiselle", "messieurs", "mesdames", "monsieur", "petit",
    "petite", "plait", "please", "simplement", "stp", "sil", "svp", "toujours",
    "tous", "tout", "toute", "tres", "vraiment", "alors", "donc", "voila",
    "comment", "journee", "soiree", "nuit", "midi", "apres", "prochaine",
    "continuation", "temps", "infiniment", "mille",
    # demande d'aide non qualifiee : le client n'a pas encore dit son sujet
    "aide", "aider", "aides", "besoin", "question", "questions", "demande",
    "renseignement", "renseignements", "renseigner", "information",
    "informations", "infos", "info", "savoir", "possible",
    # interlocuteur / canal, sans objet precis
    "service", "client", "clientele", "support", "equipe", "team", "assistance",
}

# Au-dela de cette longueur, on considere qu'il y a forcement une demande.
MAX_MOTS_CONVERSATION = 12

REPONSES = {
    "salutation": (
        "Bonjour ! Je suis l'assistant de support. Je peux vous renseigner sur "
        "les commandes, la livraison, les retours, les paiements et votre "
        "compte. Quelle est votre question ?"
    ),
    "remerciement": (
        "Avec plaisir. N'hesitez pas si vous avez une autre question."
    ),
    "acquiescement": (
        "Tres bien. Je reste a votre disposition si vous avez une question."
    ),
    "excuse": (
        "Il n'y a aucun souci. Dites-moi simplement comment je peux vous aider."
    ),
    "identite": (
        "Je suis l'assistant virtuel du support. Je reponds a partir de la FAQ "
        "et des documents de la boutique : commandes, livraison, retours, "
        "paiements et compte client. Si je n'ai pas la reponse, je transmets "
        "votre demande a un conseiller."
    ),
    "adieu": (
        "Bonne journee ! Je reste disponible si vous avez besoin d'aide."
    ),
    "indetermine": (
        "Je n'ai pas compris votre message. Pouvez-vous reformuler votre "
        "question ?"
    ),
}

# Quand plusieurs familles apparaissent dans un meme message, on repond sur
# la plus engageante. "merci, bonne journee" est un adieu ; "ca va bien ?"
# est une salutation, pas un acquiescement. L'acquiescement est le signal le
# plus faible, il ne gagne que s'il est seul.
PRIORITE = ("identite", "adieu", "salutation", "excuse", "remerciement",
            "acquiescement")

# Vocabulaires parcourus pour chaque mot.
_FAMILLES = (
    ("salutation", SALUTATIONS),
    ("remerciement", REMERCIEMENTS),
    ("acquiescement", ACQUIESCEMENTS),
    ("excuse", EXCUSES),
    ("adieu", ADIEUX),
)


def _mots(texte: str) -> list[str]:
    """Minuscules, sans accents, sans ponctuation -> liste de mots.

    Les lettres repetees par emphase sont ramenees a une seule ("merciii"
    -> "merci"), a partir de trois occurrences seulement : deux lettres
    identiques sont courantes en francais ("bonne", "aurevoir").
    """
    texte = unicodedata.normalize("NFD", texte.lower())
    texte = "".join(c for c in texte if unicodedata.category(c) != "Mn")
    texte = re.sub(r"(.)\1{2,}", r"\1", texte)
    return [m for m in re.split(r"[^a-z0-9]+", texte) if m]


def _fusionner_locutions(mots: list[str]) -> list[str]:
    """Regroupe les locutions, en essayant d'abord les plus longues."""
    fusionnes: list[str] = []
    i = 0
    while i < len(mots):
        for taille in (3, 2):
            groupe = tuple(mots[i : i + taille])
            if len(groupe) == taille and groupe in LOCUTIONS:
                fusionnes.append(LOCUTIONS[groupe])
                i += taille
                break
        else:
            fusionnes.append(mots[i])
            i += 1
    return fusionnes


def _noyau(mots: list[str]) -> list[str]:
    """Retire la politesse aux deux extremites, pour comparer le coeur de la
    phrase aux formulations d'IDENTITE ("bonjour qui es tu" -> "qui es tu")."""
    debut, fin = 0, len(mots)
    while debut < fin and mots[debut] in POLITESSE:
        debut += 1
    while fin > debut and mots[fin - 1] in POLITESSE:
        fin -= 1
    return mots[debut:fin]


def detecter_conversation(question: str) -> str | None:
    """Renvoie la famille de conversation si le message n'est pas une demande
    de support, sinon None.

    Familles : "salutation", "remerciement", "acquiescement", "excuse",
    "identite", "adieu", "indetermine".
    """
    bruts = _mots(question)

    # Message sans aucun mot (emoji seul, ponctuation seule) : on demande une
    # reformulation plutot que d'envoyer une requete vide dans le RAG.
    if not bruts:
        return "indetermine" if question.strip() else None

    if len(bruts) > MAX_MOTS_CONVERSATION:
        return None

    # Question sur l'agent lui-meme, comparee a la phrase entiere.
    if " ".join(_noyau(bruts)) in IDENTITE:
        return "identite"

    familles: list[str] = []
    for mot in _fusionner_locutions(bruts):
        for nom, vocabulaire in _FAMILLES:
            if mot in vocabulaire:
                familles.append(nom)
                break
        else:
            if mot not in MOTS_OUTILS:
                return None  # un mot porteur de sens -> vraie demande

    if not familles:
        return None
    return next(f for f in PRIORITE if f in familles)


def reponse_conversation(famille: str) -> str:
    return REPONSES.get(famille, REPONSES["salutation"])
