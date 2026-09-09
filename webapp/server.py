"""Serveur web de demonstration (FastAPI + Server-Sent Events).

Lance l'agent et diffuse chaque etape du graphe en temps reel vers le
navigateur, pour visualiser le pipeline en direct.

    python webapp/server.py
    # puis ouvre http://localhost:8000
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi import FastAPI                       # noqa: E402
from fastapi.responses import FileResponse, StreamingResponse  # noqa: E402

from support_agent.agent.graph import SupportAgent  # noqa: E402
from support_agent.config import get_settings       # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Agent de support — demo")
settings = get_settings()
agent = SupportAgent(settings)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def config():
    mode = "demo" if settings.use_fake_llm else settings.model
    return {"mode": mode, "provider": settings.provider}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/ask")
def ask(question: str):
    """Diffuse les etapes du graphe en SSE, puis un evenement final."""

    def gen():
        acc: dict = {"question": question}
        try:
            for node, partial in agent.stream(question):
                for k, v in partial.items():
                    if not k.startswith("_"):
                        acc[k] = v
                detail = (partial.get("trace") or [""])[-1]
                yield _sse({
                    "type": "step",
                    "node": node,
                    "detail": detail,
                    "score": acc.get("retrieval_score"),
                })

            cost = acc.get("cost", {})
            sources = sorted({
                d["metadata"].get("source", "?") for d in acc.get("documents", [])
            })
            yield _sse({
                "type": "done",
                "answer": acc.get("answer", ""),
                "decision": cost.get("resolution"),
                "sources": sources,
                "score": acc.get("retrieval_score"),
                "cost": cost,
            })
        except Exception as exc:  # pragma: no cover
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    print("Interface : http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
