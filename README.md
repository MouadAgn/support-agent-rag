# Agent de support client — RAG + LangGraph

**Un agent qui répond aux clients à partir de la documentation de l'entreprise, et qui refuse de répondre quand il n'est pas sûr.**

---

## Le problème

Un service client passe l'essentiel de son temps sur des questions déjà écrites quelque part : délais de livraison, retours, moyens de paiement. C'est répétitif, ça coûte cher, et ça noie les demandes qui méritent vraiment un humain.

La solution évidente — brancher un chatbot — crée un problème pire. **Un LLM interrogé sans garde-fou invente une réponse plausible et fausse.** Sur une politique de remboursement, une réponse inventée avec assurance engage l'entreprise. Mieux vaut pas de chatbot du tout.

Ce projet traite les deux problèmes ensemble : automatiser les questions courantes **et** garantir que l'agent se tait quand il ne sait pas.

## Ce que fait l'agent

Chaque question sort par une des trois portes :

```
                   ┌─ RÉPOND          la doc contient la réponse
Question client ───┼─ OUVRE UN TICKET  hors périmètre, pas urgent
                   └─ ESCALADE         urgence ou litige → téléconseiller rappelle
```

Il n'y a pas de quatrième porte. L'agent ne peut pas répondre à côté : soit il s'appuie sur un passage de la doc, soit il passe la main.

## Comment c'est construit

Un graphe [LangGraph](src/support_agent/agent/graph.py) à cinq étapes, chacune isolée et testable :

| Étape | Rôle |
|---|---|
| **retrieve** | Cherche les 4 passages les plus proches de la question dans un index vectoriel FAISS |
| **grade** | Garde-fou n°1 : si le meilleur passage est trop loin, on n'appelle même pas le LLM |
| **generate** | Le LLM rédige **uniquement** à partir des passages trouvés. S'il n'a pas l'info, il doit écrire `INSUFFISANT` |
| **handle_fallback** | Déclenche l'outil adapté : `create_ticket` ou `escalate_to_agent` |
| **finalize** | Chiffre le coût en tokens de la réponse et le compare au coût humain |

**Trois choix d'architecture qui comptent :**

- **La recherche tourne en local, seule la rédaction part chez DeepSeek.** Les embeddings (`sentence-transformers`) sont calculés sur la machine : gratuits, rapides, et les documents ne quittent pas l'infrastructure. DeepSeek ne voit que les 4 extraits nécessaires.
- **Le routage ticket/escalade est déterministe, pas confié au LLM.** Sur une décision d'escalade, un client fraudé mis en file asynchrone est un incident. Une règle explicite est auditable et gratuite. Les actions sont déjà encapsulées en outils LangChain (`@tool`) pour basculer en tool-calling le jour où la taxonomie l'exige.
- **Quand `grade` rejette, zéro token est dépensé.** Le garde-fou est aussi une optimisation de coût.

## Résultats mesurés

13 questions réelles, reformulées différemment de la FAQ, sur `deepseek-v4-pro`.

| | Résultat |
|---|---|
| **Questions hors périmètre déviées vers un humain** | **5 / 5** |
| **Réponses inventées (hallucinations)** | **0** |
| Routage urgence correct (litige, fraude → appel) | 2 / 2 |
| Questions du périmètre auxquelles l'agent répond | 5 / 8 |
| Bon passage retrouvé par la recherche | 8 / 8 (score 0,38 – 0,63) |
| Coût moyen d'une réponse | **0,0017 €** |
| Latence médiane bout en bout | 2,8 s *(recherche seule : 141 ms)* |
| Tests automatisés | 13 ✅ |

**Ce que ces chiffres disent vraiment.** Le résultat qui compte est le premier : sur « quel est votre chiffre d'affaires ? » ou « quelle est la composition chimique de vos t-shirts ? », l'agent n'a jamais bluffé. Il a ouvert un ticket.

Le prix de cette prudence est visible ligne 5 : **l'agent refuse aussi 3 questions qu'il aurait pu traiter.** Exemple, « vous expédiez en Belgique ? » — la FAQ dit « la plupart des pays de l'UE » sans nommer la Belgique, et l'agent a préféré passer la main. C'est un arbitrage assumé : pour un support client, un faux refus coûte un ticket, une fausse réponse coûte un litige.

> ⚠️ Corpus de démonstration : 10 entrées de FAQ + 1 PDF (13 passages indexés). Les chiffres valident le comportement du pipeline, pas une performance à l'échelle d'une vraie base documentaire.

## Le modèle économique

Une réponse coûte **0,0017 €**. Un ticket traité par un humain est estimé à **5 €**.

> Environ **2 900 réponses automatiques** pour le prix d'un seul ticket humain.

Le coût est calculé sur les tokens réellement facturés par l'API (`prompt_tokens` / `completion_tokens`), aux tarifs officiels DeepSeek convertis en euros — pas sur une estimation. Voir [`cost/tracker.py`](src/support_agent/cost/tracker.py).

**Ce qui reste une hypothèse**, et qu'il faut assumer comme tel : les 5 € par ticket et 6,50 € par appel sont des ordres de grandeur métier, pas des coûts mesurés. Et l'économie affichée suppose que chaque question traitée serait devenue un ticket — en pratique, une partie n'aurait jamais atteint le support.

## Limites connues

- **Le seuil de pertinence ne filtre rien en pratique.** À `0.20`, il n'a rejeté aucune des 13 questions (scores mesurés : 0,33 à 0,63). C'est le prompt strict qui fait tout le travail de refus. Le seuil doit être recalibré sur un jeu de questions étiquetées.
- **`deepseek-v4-pro` est un modèle à raisonnement** : sur 6 appels mesurés, **83 % des tokens de sortie facturés** (71 % à 98 %) sont des tokens de raisonnement que personne ne voit. La sortie étant le poste le plus cher, on paie du raisonnement pour reformuler une FAQ — `deepseek-v4-flash` serait ~3× moins cher.
- **Le cache de prompt DeepSeek n'est pas exploité** (input mis en cache facturé 30× moins). Le prompt système est identique à chaque requête : gain immédiat disponible.
- **`all-MiniLM-L6-v2` est entraîné principalement sur de l'anglais** alors que le corpus est en français. Un modèle multilingue améliorerait la recherche.
- **Le routage d'urgence est lexical** : il ne détecte pas l'implicite (« on m'a prélevé deux fois » sans mot-clé d'urgence) ni les fautes de frappe.
- **Pas de mémoire conversationnelle** : chaque question est traitée isolément.
- **Détection de refus fragile** : `"INSUFFISANT" in texte` rejette aussi une réponse valide qui citerait le mot. Une sortie structurée serait plus robuste.

## Démarrer

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # puis colle ta clé DeepSeek
```

```bash
python scripts/build_sample_pdf.py    # génère le PDF d'exemple (une fois)
python scripts/ingest.py              # construit l'index vectoriel
```

**En ligne de commande :**
```bash
python app.py                                          # mode interactif
python app.py "Quels sont les délais de livraison ?"
python app.py --trace "Comment retourner un article ?" # détail de chaque étape
```

**Interface web** — visualise le pipeline en direct, étape par étape (FastAPI + SSE) :
```bash
python webapp/server.py     # puis http://localhost:8000
```

**Sans clé API** — tout le pipeline (RAG, routage, coûts) tourne avec un LLM simulé :
```bash
DEMO_MODE=true EMBEDDINGS_BACKEND=tfidf python scripts/ingest.py
DEMO_MODE=true EMBEDDINGS_BACKEND=tfidf python app.py --trace "délais de livraison"
```

**Tests :**
```bash
pytest -q
```

## Structure

```
support-agent-rag/
├── app.py                      # CLI
├── webapp/server.py            # interface web live (FastAPI + SSE)
├── scripts/ingest.py           # construction de l'index vectoriel
├── data/                       # faq/ · pdf/ · index/ (généré, non versionné)
├── src/support_agent/
│   ├── config.py               # configuration (.env)
│   ├── llm.py                  # client DeepSeek/OpenAI + LLM simulé
│   ├── ingestion/              # loaders (PDF + OCR) · chunking · vectorstore
│   ├── agent/                  # graph · nodes · tools · prompts · state
│   └── cost/tracker.py         # suivi des coûts
├── docs/CONCEPTS.md            # les notions expliquées (RAG, LLM, OCR, VLM…)
└── tests/                      # pytest
```

## Pile technique

Python · LangGraph · LangChain · DeepSeek (API compatible OpenAI) · FAISS · sentence-transformers (repli TF-IDF) · FastAPI + SSE · pypdf · Tesseract (OCR, optionnel) · tiktoken · pytest

Toutes les notions employées sont expliquées pas à pas dans [`docs/CONCEPTS.md`](docs/CONCEPTS.md).
