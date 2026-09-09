"""Interface en ligne de commande de l'agent de support.

Usage :
    python app.py                 # mode interactif (REPL)
    python app.py "ma question"   # reponse unique
    python app.py --trace "..."   # affiche le detail des etapes du graphe
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from support_agent.agent.graph import SupportAgent   # noqa: E402
from support_agent.config import get_settings        # noqa: E402

DECISION_LABEL = {
    "answered": "REPONDU par l'agent",
    "ticket": "TICKET ouvert (conseiller)",
    "escalate": "ESCALADE vers teleconseiller",
}


def render(result: dict, show_trace: bool) -> None:
    cost = result.get("cost", {})
    print("\n" + "=" * 68)
    print("REPONSE :")
    print("  " + result.get("answer", "").replace("\n", "\n  "))

    docs = result.get("documents", [])
    if docs:
        srcs = sorted({d["metadata"].get("source", "?") for d in docs})
        print(f"\nSOURCES   : {', '.join(srcs)}  (similarite max "
              f"{result.get('retrieval_score', 0):.3f})")

    print(f"DECISION  : {DECISION_LABEL.get(cost.get('resolution'), '?')}")
    print(
        f"COUT      : {cost.get('input_tokens', 0)} tok in + "
        f"{cost.get('output_tokens', 0)} tok out "
        f"= {cost.get('llm_cost_eur', 0):.5f} EUR"
    )
    saving = cost.get("saving_eur", 0)
    if cost.get("resolution") == "answered":
        print(f"ECONOMIE  : ~{saving:.2f} EUR (ticket humain evite)")
    else:
        print(f"COUT HUMAIN : {cost.get('human_cost_eur', 0):.2f} EUR "
              f"(+ {cost.get('llm_cost_eur', 0):.5f} EUR de pre-qualification)")

    if show_trace:
        print("\nTRACE :")
        for step in result.get("trace", []):
            print(f"  - {step}")
    print("=" * 68)


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--trace"]
    show_trace = "--trace" in sys.argv

    s = get_settings()
    mode = "DEMO (LLM simule)" if s.use_fake_llm else f"{s.provider} / {s.model}"
    print(f"Agent de support pret — mode LLM : {mode}")

    agent = SupportAgent(s)

    if args:  # question unique
        render(agent.answer(" ".join(args)), show_trace)
        return

    print("Pose ta question (ou 'q' pour quitter).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if q:
            render(agent.answer(q), show_trace)


if __name__ == "__main__":
    main()
