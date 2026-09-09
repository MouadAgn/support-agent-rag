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

En amont, un triage écarte ce qui n'est pas une demande. « Bonjour », « cv ? », « ok », « merci », « t'es un robot ? » reçoivent une réponse immédiate — pas de recherche documentaire, pas d'appel au modèle, et surtout pas de ticket ouvert pour rien.

La règle est asymétrique, et c'est volontaire : **un seul mot métier suffit à basculer dans le RAG.** Rater un « bonjour » coûte un appel LLM ; classer « mon colis est bloqué » comme de la politesse coûte un client jamais traité.

## Comment c'est construit

Un graphe [LangGraph](src/support_agent/agent/graph.py) à six étapes, chacune isolée et testable :

| Étape | Rôle |
|---|---|
| **triage** | Sépare une vraie demande de ce qui n'en est pas : salutation, remerciement, acquiescement, excuse, adieu, question sur l'agent (« t'es un robot ? »), message vide de mots (emoji seul). Chaque cas reçoit une réponse directe, sans recherche ni appel au modèle |
| **retrieve** | Cherche les 4 passages les plus proches de la question dans un index vectoriel FAISS |
| **grade** | Garde-fou n°1 : si le meilleur passage est trop loin, on n'appelle même pas le LLM |
| **generate** | Le LLM rédige **uniquement** à partir des passages trouvés. S'il n'a pas l'info, il doit écrire `INSUFFISANT` |
| **handle_fallback** | Déclenche l'outil adapté : `create_ticket` ou `escalate_to_agent` |
| **finalize** | Chiffre le coût en tokens de la réponse et le compare au coût humain |

**Quatre choix d'architecture qui comptent :**

- **La recherche tourne en local, seule la rédaction part chez DeepSeek.** Les embeddings (`sentence-transformers`) sont calculés sur la machine : gratuits, rapides, et les documents ne quittent pas l'infrastructure. DeepSeek ne voit que les 4 extraits nécessaires.
- **Le routage ticket/escalade est déterministe, pas confié au LLM.** Sur une décision d'escalade, un client fraudé mis en file asynchrone est un incident. Une règle explicite est auditable et gratuite. Les actions sont déjà encapsulées en outils LangChain (`@tool`) pour basculer en tool-calling le jour où la taxonomie l'exige.
- **Pour une FAQ, on indexe la question, pas la question + la réponse.** Le client pose une question : comparer une question à une question est bien plus précis que la comparer à un bloc où la réponse dilue le sens. Le LLM reçoit quand même la réponse complète comme contexte. Gain mesuré : rappel@1 de 9/20 à 14/20.
- **Quand `grade` rejette — ou quand le triage reconnaît un message conversationnel — zéro token est dépensé.** Les garde-fous sont aussi des optimisations de coût.

## Résultats mesurés

23 messages réels, tous reformulés différemment de la FAQ, sur `deepseek-v4-pro`.

| | Résultat |
|---|---|
| **Questions hors périmètre déviées vers un humain** | **4 / 4** |
| **Réponses inventées (hallucinations)** | **0** |
| Questions du périmètre traitées | **12 / 12** |
| Urgences et litiges escaladés vers un appel | 2 / 2 |
| Messages conversationnels traités sans appel au modèle | 5 / 5 *(0 token, 2 ms)* |
| Bon passage dans les 4 résultats de recherche | 17 / 20 |
| Coût moyen d'une réponse | **0,0014 €** |
| Latence médiane bout en bout | 3,3 s |
| Tests automatisés | 160 ✅ |

**Ce que ces chiffres disent vraiment.** Le résultat qui compte est le premier : sur « quel est votre chiffre d'affaires ? » ou « quelle est la composition chimique de vos t-shirts ? », l'agent n'a jamais bluffé — il a ouvert un ticket.

Le second est le plus instructif sur la démarche. La couverture était de **8/12**, et le coupable n'était ni le prompt ni la taille du corpus : c'était le modèle d'embeddings. `all-MiniLM-L6-v2`, entraîné sur de l'anglais, comparait des mots plutôt que du sens — « justificatif pour ma **compta** » remontait « Comment créer un **compte** ? ». Un jeu de 20 reformulations annotées a permis de comparer quatre configurations :

| Configuration | rappel@1 | rappel@4 |
|---|---|---|
| MiniLM anglais, question + réponse indexées | 9/20 | 11/20 |
| MiniLM anglais, question seule indexée | 9/20 | 14/20 |
| Multilingue, question + réponse indexées | 9/20 | 17/20 |
| **Multilingue, question seule indexée** | **14/20** | **17/20** |

Le passage au modèle multilingue et l'indexation sur la question portent la couverture à **12/12**, sans toucher au prompt.

> ⚠️ Corpus de démonstration : 33 entrées de FAQ + 1 PDF de CGV, soit 36 passages indexés. Les chiffres valident le comportement du pipeline, pas une performance à l'échelle d'une base documentaire réelle.

## Le modèle économique

Une réponse coûte **0,0014 €**. Un ticket traité par un humain est estimé à **5 €**.

> Environ **3 500 réponses automatiques** pour le prix d'un seul ticket humain.

Le coût est calculé sur les tokens réellement facturés par l'API (`prompt_tokens` / `completion_tokens`), aux tarifs officiels DeepSeek convertis en euros — pas sur une estimation. Voir [`cost/tracker.py`](src/support_agent/cost/tracker.py).

**Ce qui reste une hypothèse**, et qu'il faut assumer comme tel : les 5 € par ticket et 6,50 € par appel sont des ordres de grandeur métier, pas des coûts mesurés. Et l'économie affichée suppose que chaque question traitée serait devenue un ticket — en pratique, une partie n'aurait jamais atteint le support.

## Limites connues

- **Le seuil de pertinence ne filtre toujours rien.** À `0.20`, il ne rejette aucun message : les questions hors périmètre marquent 0,27 à 0,34, les questions couvertes 0,28 à 0,79. La séparation s'est améliorée avec le modèle multilingue, mais les deux plages se chevauchent encore — un seuil unique ne peut pas les trancher. C'est le prompt strict qui fait le travail de refus. Il faudrait recalibrer sur un jeu étiqueté plus large, ou remplacer le seuil par un classifieur.
- **`deepseek-v4-pro` est un modèle à raisonnement** : sur 6 appels mesurés, **83 % des tokens de sortie facturés** (71 % à 98 %) sont des tokens de raisonnement que personne ne voit. La sortie étant le poste le plus cher, on paie du raisonnement pour reformuler une FAQ — `deepseek-v4-flash` serait ~3× moins cher.
- **Le cache de prompt DeepSeek n'est pas exploité** (input mis en cache facturé 30× moins). Le prompt système est identique à chaque requête : gain immédiat disponible.
- **La recherche rate encore 3 reformulations sur 20**, malgré le modèle multilingue — par exemple « on m'a envoyé autre chose que ce que j'ai commandé ». Une recherche hybride (dense + lexicale) ou un re-ranking corrigeraient une partie de ces cas.
- **Indexer la question seule a un revers** : un terme qui n'existe que dans la réponse devient moins retrouvable. Ici « Belgique » n'apparaît que dans la réponse sur l'international, et la question fonctionne quand même — mais le compromis est réel.
- **Le routage d'urgence est lexical** : il ne détecte pas l'implicite (« on m'a prélevé deux fois » sans mot-clé d'urgence) ni les fautes de frappe.
- **Pas de mémoire conversationnelle** : chaque question est traitée isolément.
- **Détection de refus fragile** : `"INSUFFISANT" in texte` rejette aussi une réponse valide qui citerait le mot. Une sortie structurée serait plus robuste.

## Démarrer

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Crée ensuite un fichier `.env` à la racine :

```ini
DEEPSEEK_API_KEY=sk-...        # ta clé ; sans elle, le mode démo prend le relais
LLM_MODEL=deepseek-v4-pro      # deepseek-v4-flash est ~3x moins cher
LLM_BASE_URL=https://api.deepseek.com

TOP_K=4                        # passages envoyés au modèle
SCORE_THRESHOLD=0.20           # seuil de pertinence (garde-fou)
CHUNK_SIZE=800                 # taille des morceaux, en caractères
CHUNK_OVERLAP=120

# modèle d'embeddings, local et gratuit (multilingue : le corpus est en français)
EMBEDDINGS_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

PRICE_INPUT_PER_M=1.135        # € par million de tokens en entrée
PRICE_OUTPUT_PER_M=3.406       # € par million de tokens en sortie
COST_PER_TICKET=5.00           # coût moyen d'un ticket humain
COST_PER_CALL=6.50             # coût moyen d'un appel
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
│   ├── agent/                  # graph · nodes · triage · tools · prompts · state
│   └── cost/tracker.py         # suivi des coûts
├── docs/CONCEPTS.md            # les notions expliquées (RAG, LLM, OCR, VLM…)
└── tests/                      # pytest
```

## Pile technique

Python · LangGraph · LangChain · DeepSeek (API compatible OpenAI) · FAISS · sentence-transformers (repli TF-IDF) · FastAPI + SSE · pypdf · Tesseract (OCR, optionnel) · tiktoken · pytest

Toutes les notions employées sont expliquées pas à pas dans [`docs/CONCEPTS.md`](docs/CONCEPTS.md).
