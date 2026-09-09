# Les concepts, expliques simplement

Ce document explique chaque notion utilisee dans le projet. Pour chacune :
ce que c'est, a quoi ca sert, et comment c'est utilise ici. Objectif : que tu
puisses en parler avec tes mots en entretien, sans reciter.

---

## LLM (Large Language Model)

**C'est quoi.** Un grand modele de langage. Un reseau de neurones entraine sur
d'enormes quantites de texte, qui predit le mot suivant et sait ainsi rediger,
resumer, repondre, reformuler. DeepSeek, GPT, Llama, Mistral en sont des
exemples.

**A quoi ca sert.** A generer du texte en langage naturel. Dans un support
client, c'est lui qui redige la reponse finale, propre et polie.

**Dans le projet.** On appelle DeepSeek (`deepseek-chat`) via le SDK compatible
OpenAI. Le LLM n'intervient qu'a l'etape de generation, et seulement a partir
des documents qu'on lui donne. Voir `src/support_agent/llm.py`.

**Le piege a connaitre : l'hallucination.** Un LLM peut inventer une reponse
fausse avec aplomb. C'est LE probleme du support automatise. La parade dans ce
projet : on lui interdit de repondre en dehors du contexte fourni, et on ajoute
un garde-fou (voir "RAG quality").

---

## Prompt

**C'est quoi.** Le texte d'instruction qu'on envoie au LLM. Il y a en general un
"system prompt" (le role et les regles) et un "user prompt" (la demande).

**A quoi ca sert.** A cadrer le comportement du modele. Un bon prompt fait la
moitie de la qualite du resultat.

**Dans le projet.** Le system prompt (dans `agent/prompts.py`) dit : "tu es un
assistant de support, tu reponds uniquement a partir du contexte, si tu ne sais
pas tu ecris INSUFFISANT, tu n'inventes jamais". Le user prompt assemble le
contexte trouve + la question. Ce mot "INSUFFISANT" est un signal qu'on
detecte ensuite pour basculer vers un humain.

---

## RAG (Retrieval-Augmented Generation)

**C'est quoi.** "Generation augmentee par la recherche". Au lieu de demander au
LLM de repondre de memoire, on va d'abord **chercher** les passages utiles dans
une base documentaire, puis on les **donne** au LLM pour qu'il redige sa reponse
a partir de ces passages.

**A quoi ca sert.** Deux benefices majeurs. D'abord, les reponses collent a TES
documents (a jour, specifiques a l'entreprise), pas aux connaissances generales
du modele. Ensuite, ca reduit fortement les hallucinations, car le modele
s'appuie sur des sources reelles.

**Dans le projet.** C'est toute la chaine : la question part chercher les bons
extraits de la FAQ et des PDF (etape retrieve), puis ces extraits sont injectes
dans le prompt (etape generate). Sans RAG, l'agent ne connaitrait pas tes
delais de livraison ni ta politique de retour.

**La phrase a retenir en entretien :** "Le RAG, c'est un examen a livre ouvert
pour le LLM : il ne recite pas, il lit la doc avant de repondre."

---

## Embeddings

**C'est quoi.** Transformer un texte en un vecteur de nombres qui capture son
**sens**. Deux phrases proches par le sens donnent deux vecteurs proches dans
l'espace, meme si elles n'utilisent pas les memes mots. "delais de livraison" et
"combien de temps pour recevoir mon colis" seront proches.

**A quoi ca sert.** A faire de la recherche **semantique** plutot que par mots
cles. C'est ce qui permet de retrouver le bon passage meme quand le client
formule sa question autrement que la FAQ.

**Dans le projet.** On calcule les embeddings avec `sentence-transformers`
(modele `all-MiniLM-L6-v2`), en local et gratuitement. Un repli en TF-IDF
(scikit-learn) existe pour tourner sans aucun telechargement. Voir
`ingestion/vectorstore.py`.

**A savoir :** DeepSeek n'a pas d'API d'embeddings. Faire les embeddings en local
est donc un vrai choix d'archi : gratuit, rapide, et confidentiel (les documents
ne partent pas chez un tiers).

---

## Vector store (base vectorielle) et FAISS

**C'est quoi.** Une base de donnees specialisee dans le stockage de vecteurs et
la recherche des "plus proches voisins". FAISS (de Meta) en est une, tres
utilisee, qui tourne en memoire.

**A quoi ca sert.** A retrouver tres vite, parmi des milliers de passages, ceux
dont le vecteur ressemble le plus au vecteur de la question.

**Dans le projet.** On indexe tous les chunks dans un `IndexFlatIP` FAISS. Les
vecteurs sont normalises, donc le produit scalaire donne directement la
**similarite cosinus** (un score entre 0 et 1). La recherche renvoie les
`top_k` passages avec leur score.

---

## Chunking (decoupage)

**C'est quoi.** Couper les documents en morceaux (chunks) de taille raisonnable
avant de les indexer.

**A quoi ca sert.** Deux raisons. Pour la recherche : un petit passage cible est
plus facile a retrouver qu'un document entier. Pour le cout : on n'envoie au LLM
que les quelques passages utiles, pas tout le document, donc moins de tokens.

**Le compromis.** Trop gros, on noie l'info et on paie des tokens inutiles. Trop
petit, on coupe une idee en deux et on perd le contexte. On ajoute un
**overlap** (chevauchement) entre chunks pour ne pas perdre une info a la
frontiere.

**Dans le projet.** `CHUNK_SIZE=800` caracteres, `CHUNK_OVERLAP=120`. Astuce
metier : une entree de FAQ (une question + sa reponse) est deja une unite de
sens, donc on ne la redecoupe pas. Voir `ingestion/chunking.py`.

---

## OCR (Reconnaissance optique de caracteres)

**C'est quoi.** Extraire le texte d'une image ou d'un document scanne. "Lire"
une image de texte pour la transformer en texte editable.

**A quoi ca sert.** Beaucoup de documents (factures, contrats, notices scannees)
sont des PDF image, sans texte selectionnable. Sans OCR, le RAG ne peut rien en
tirer.

**Dans le projet.** On extrait d'abord le texte natif des PDF avec `pypdf`
(rapide, fiable). Si une page ne contient pas de texte (page scannee), on tente
l'OCR avec Tesseract (`pytesseract`). C'est un repli optionnel, active si
Tesseract est installe. Voir `ingestion/loaders.py`.

---

## VLM (Vision-Language Model)

**C'est quoi.** Un modele qui comprend a la fois l'image et le texte. On peut lui
montrer une photo ou un schema et lui poser une question dessus.

**A quoi ca sert.** La ou l'OCR se contente de lire les caracteres, un VLM
**comprend** le visuel : decrire une capture d'ecran, lire un tableau complexe,
interpreter un schema, comprendre une photo de produit defectueux envoyee par un
client.

**Dans le projet.** L'OCR couvre le cas courant (PDF scanne = du texte). Le VLM
est la piste d'evolution pour les documents riches en visuels et pour les images
envoyees par les clients. C'est un bon point a mentionner comme "prochaine
etape" en entretien : on montre qu'on connait la difference OCR / VLM.

**La distinction a retenir :** OCR = lire les lettres d'une image. VLM =
comprendre le contenu d'une image.

---

## RAG quality (qualite du RAG)

**C'est quoi.** L'ensemble des mesures qui garantissent que le RAG repond bien :
est-ce qu'on a retrouve les bons passages ? est-ce que la reponse est fidele aux
sources ? est-ce qu'on sait dire "je ne sais pas" ?

**A quoi ca sert.** Sans controle qualite, un RAG repond a tout, y compris a
cote. Le pire pour un support, c'est une reponse fausse donnee avec assurance.

**Dans le projet, deux garde-fous.**
1. Un seuil de pertinence sur le score de recherche (`SCORE_THRESHOLD`). Si le
   meilleur passage est trop loin de la question, on ne genere meme pas : on
   passe la main a un humain (etape grade).
2. Le LLM doit repondre "INSUFFISANT" quand le contexte ne contient pas la
   reponse. On detecte ce mot et on bascule aussi vers un humain.

Resultat : l'agent prefere dire "je passe la main" plutot que d'inventer. C'est
exactement ce qu'on veut d'un support automatise.

---

## Agent

**C'est quoi.** Un systeme qui ne se contente pas de repondre, mais qui **decide
d'actions** : chercher, appeler un outil, choisir un chemin selon la situation.
Un chatbot repond ; un agent agit.

**A quoi ca sert.** A automatiser un processus complet, pas juste une reponse.
Ici : chercher, evaluer, repondre OU ouvrir un ticket OU escalader.

**Dans le projet.** Notre agent enchaine des etapes et prend des decisions
(repondre / ticket / appel) en fonction de la pertinence du contexte et de
l'urgence de la demande. Il utilise des outils pour agir sur le monde exterieur.

---

## Tools (outils)

**C'est quoi.** Des fonctions que l'agent peut declencher pour agir au-dela du
texte : creer un ticket, envoyer un mail, interroger une API, router un appel.

**A quoi ca sert.** A connecter l'agent au systeme d'information reel. Un agent
sans outils ne fait que parler ; avec des outils, il fait des choses.

**Dans le projet.** Deux outils LangChain (`@tool`) dans `agent/tools.py` :
`create_ticket` (ouvre un ticket pour un conseiller) et `escalate_to_agent`
(route vers le centre d'appel). Ici ils simulent l'action en renvoyant un
identifiant ; dans un vrai SI on brancherait l'API du helpdesk (Zendesk, Jira)
ou la telephonie.

---

## LangChain

**C'est quoi.** Une bibliotheque Python qui fournit les briques standard pour
construire des applications a base de LLM : appels aux modeles, outils,
connexions aux bases vectorielles, gestion des prompts.

**A quoi ca sert.** A ne pas reinventer la roue. On utilise des composants
eprouves et interchangeables (changer de LLM ou de vector store sans tout
reecrire).

**Dans le projet.** On utilise `langchain-core` (les outils `@tool`) et
`langchain-openai` (le client de chat, pointe vers DeepSeek).

---

## LangGraph

**C'est quoi.** Une extension de LangChain pour orchestrer un agent sous forme de
**graphe** : des noeuds (les etapes) relies par des aretes, avec des branchements
conditionnels et un etat partage qui circule.

**A quoi ca sert.** Quand la logique n'est pas une simple ligne droite mais un
arbre de decisions (si le contexte est bon alors repondre, sinon escalader),
LangGraph rend ce flux **lisible, testable et controlable**. On voit le chemin
exact suivi pour chaque question.

**Dans le projet.** Le graphe (dans `agent/graph.py`) a cinq noeuds : retrieve,
grade, generate, handle_fallback, finalize. Deux branchements conditionnels
decident du chemin. C'est le squelette de tout l'agent.

**LangChain vs LangGraph, la phrase qui clarifie :** "LangChain donne les
briques, LangGraph donne le plan de montage avec les embranchements."

---

## Tokens et couts

**C'est quoi.** Un token est un morceau de mot (environ 4 caracteres en
francais). Les LLM facturent au token : un prix pour les tokens envoyes (entree)
et un prix pour les tokens generes (sortie).

**A quoi ca sert de les suivre.** Pour piloter le cout d'un support automatise et
prouver le retour sur investissement. C'est un argument cle : combien coute une
reponse, et combien on economise par rapport a un humain.

**Dans le projet.** A chaque appel LLM, on recupere le nombre de tokens
(entree/sortie) et on calcule le cout en euros. On le compare au cout d'un
ticket (~5 EUR) ou d'un appel (~6,50 EUR). Une reponse automatique coute de
l'ordre de 0,0002 EUR : chaque question deviee vers l'agent economise donc
presque le cout complet d'un traitement humain. Voir `cost/tracker.py`.

---

## Recapitulatif du flux (a savoir raconter)

Une question arrive. On la transforme en vecteur (embedding) et on cherche dans
l'index FAISS les passages les plus proches (RAG, retrieve). On verifie leur
pertinence (grade) : si c'est trop faible, on ne prend pas de risque et on
escalade. Sinon, on donne ces passages au LLM avec un prompt strict qui lui
interdit d'inventer (generate). S'il ne trouve pas la reponse dans le contexte,
il le dit et on bascule vers un humain. Selon l'urgence, on ouvre un ticket ou
on route un appel (tools). Enfin, on chiffre le cout de la reponse et l'economie
realisee (finalize). Tout ce flux est un graphe LangGraph.

---

## Questions d'entretien probables

**"Comment tu evites les hallucinations ?"**
Trois niveaux : le RAG (le modele repond sur des sources reelles), un prompt
strict (interdiction de repondre hors contexte, obligation de dire INSUFFISANT),
et un seuil de pertinence qui escalade vers un humain quand la recherche est
trop faible.

**"Pourquoi LangGraph et pas juste des if/else ?"**
Parce que le flux a des branchements et un etat partage. LangGraph rend ce
graphe explicite, testable et observable (on trace le chemin exact), la ou des
if/else deviennent vite illisibles quand l'agent grandit.

**"Comment tu maitrises les couts ?"**
Embeddings en local (gratuits), chunking pour n'envoyer au LLM que l'utile, et
un suivi des tokens par reponse qui chiffre l'economie face a un ticket ou un
appel.

**"Et si le document est un scan ou une image ?"**
OCR (Tesseract) pour extraire le texte des PDF scannes. Pour les visuels
complexes ou les images clients, l'evolution naturelle est un VLM qui comprend
l'image, pas seulement les caracteres.

**"Comment tu mesures que le RAG est bon ?"**
On regarde le score de recuperation, le taux de reponses vs escalades, et la
fidelite des reponses aux sources. On peut ajouter un jeu de questions/reponses
de reference pour mesurer la precision du retrieval et la qualite des reponses.
