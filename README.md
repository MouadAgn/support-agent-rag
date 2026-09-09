# Agent de support utilisateur (LangGraph + RAG)

Un agent qui repond aux questions des clients a partir de la FAQ et des
documents (PDF) de l'entreprise. Son but : reduire les sollicitations du
support humain. Il ne repond que lorsqu'il est sur de lui ; sinon il ouvre un
ticket ou escalade vers un teleconseiller. Chaque reponse affiche son cout en
tokens et le compare au cout d'un traitement humain.

Projet concu comme une preuve de concept realiste, orientee production :
architecture claire, garde-fous anti-hallucination, tests, et suivi des couts.

## Le parcours

```
Utilisateur
    |
    v
 Plateforme / FAQ
    |
    v
 AGENT (LangGraph + RAG)
    |
    |-- repond directement            (question couverte par la doc)
    |-- ouvre un TICKET               (question hors perimetre, non urgente)
    '-- escalade vers un TELECONSEILLER (urgence / litige, appel)
```

## Comment ca marche

Le coeur est un graphe LangGraph a cinq etapes :

1. **retrieve** : on cherche dans l'index les passages les plus proches de la
   question (recherche vectorielle, RAG).
2. **grade** : garde-fou qualite. Si les passages ne sont pas assez pertinents
   (score sous le seuil), on ne laisse pas le LLM improviser : on passe la main
   a un humain.
3. **generate** : le LLM redige une reponse **uniquement** a partir du contexte
   trouve. S'il n'a pas l'info, il doit repondre "INSUFFISANT".
4. **handle_fallback** : quand l'agent ne peut pas repondre, il declenche un
   outil : `create_ticket` (asynchrone) ou `escalate_to_agent` (appel), selon
   l'urgence detectee dans la question.
5. **finalize** : calcul du cout de la reponse (tokens en entree/sortie) et de
   l'economie realisee par rapport a un ticket ou un appel.

La recherche (embeddings) tourne **en local**, gratuitement. Seule la
generation appelle DeepSeek. C'est un choix d'architecture : moins cher, plus
rapide, et les documents ne quittent pas l'infrastructure.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env      # puis colle ta cle DeepSeek dans .env
```

## Utilisation

```bash
# 1. generer le PDF d'exemple (une fois)
python scripts/build_sample_pdf.py

# 2. construire l'index a partir de la FAQ + des PDF
python scripts/ingest.py

# 3. poser des questions
python app.py                                   # mode interactif
python app.py "Quels sont les delais de livraison ?"
python app.py --trace "Comment retourner un article ?"   # detail des etapes
```

### Essayer sans cle (mode demo)

Tout le pipeline (RAG, routage, couts) fonctionne sans cle API grace a un LLM
simule et a un repli d'embeddings en TF-IDF :

```bash
DEMO_MODE=true EMBEDDINGS_BACKEND=tfidf python scripts/ingest.py
DEMO_MODE=true EMBEDDINGS_BACKEND=tfidf python app.py --trace "delais de livraison"
```

## Le modele de cout

Pour chaque question, l'agent chiffre :

- le cout des tokens LLM (prix entree + prix sortie, configurables dans `.env`) ;
- le cout humain evite : un ticket (~5 EUR) ou un appel (~6,50 EUR).

Une reponse automatique coute de l'ordre de 0,0002 EUR. Chaque question deviee
economise donc quasiment le cout complet d'un ticket. Les prix sont des
estimations a ajuster a la grille DeepSeek et aux couts reels du support.

## Configuration (.env)

Les reglages importants : le fournisseur et le modele (`LLM_PROVIDER`,
`LLM_MODEL`), le seuil de pertinence du RAG (`SCORE_THRESHOLD`), la taille des
chunks (`CHUNK_SIZE`, `CHUNK_OVERLAP`), le backend d'embeddings
(`EMBEDDINGS_BACKEND`), et les prix pour le calcul des couts. Voir
`.env.example` pour le detail.

## Structure du projet

```
support-agent-rag/
├── app.py                      # CLI (interactif + une-question)
├── scripts/
│   ├── build_sample_pdf.py     # genere un PDF d'exemple
│   └── ingest.py               # construit l'index vectoriel
├── data/
│   ├── faq/faq.md              # base de connaissance FAQ
│   ├── pdf/                     # documents PDF a ingerer
│   └── index/                  # index FAISS genere (non versionne)
├── src/support_agent/
│   ├── config.py               # configuration (.env)
│   ├── llm.py                  # client DeepSeek/OpenAI + LLM simule
│   ├── ingestion/
│   │   ├── loaders.py          # FAQ + PDF (+ OCR en repli)
│   │   ├── chunking.py         # decoupage en morceaux
│   │   └── vectorstore.py      # embeddings + FAISS (+ repli TF-IDF)
│   ├── agent/
│   │   ├── graph.py            # graphe LangGraph
│   │   ├── nodes.py            # les 5 etapes
│   │   ├── tools.py            # create_ticket / escalate_to_agent
│   │   ├── prompts.py          # prompts systeme
│   │   └── state.py            # etat partage
│   └── cost/tracker.py         # suivi des couts
├── docs/CONCEPTS.md            # les notions expliquees (RAG, LLM, VLM, OCR...)
└── tests/                      # pytest (chunking, RAG, couts, graphe)
```

## Tests

```bash
pytest -q
```

## Pile technique

Python, LangGraph, LangChain, DeepSeek (compatible OpenAI), FAISS,
sentence-transformers (repli TF-IDF via scikit-learn), pypdf, OCR Tesseract
optionnel, tiktoken, pytest.

Les notions employees sont expliquees en detail dans
[`docs/CONCEPTS.md`](docs/CONCEPTS.md).
