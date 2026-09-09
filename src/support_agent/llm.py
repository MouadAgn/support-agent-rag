"""Client LLM unifie.

- En production : appelle DeepSeek (ou OpenAI) via le SDK compatible OpenAI.
- En mode demo (sans cle) : un LLM simule et deterministe, pour pouvoir
  tester et demontrer tout le pipeline sans depenser de tokens.

Dans les deux cas, `chat()` renvoie (texte, usage) ou usage contient le
nombre de tokens en entree/sortie, ce qui alimente le calcul des couts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .config import Settings

try:  # comptage de tokens (approche cl100k, suffisant pour une estimation)
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:  # pragma: no cover - fallback grossier

    def count_tokens(text: str) -> int:
        return max(1, len((text or "").split()))


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int


class LLM(Protocol):
    def chat(self, system: str, user: str) -> LLMResult: ...


# --------------------------------------------------------------------------- #
#  LLM reel (DeepSeek / OpenAI)                                                #
# --------------------------------------------------------------------------- #
class RealLLM:
    """Appelle DeepSeek ou OpenAI. DeepSeek est compatible avec le SDK OpenAI :
    il suffit de pointer `base_url` vers https://api.deepseek.com."""

    def __init__(self, settings: Settings):
        from openai import OpenAI

        self.settings = settings
        self.model = settings.model
        base_url = settings.base_url if settings.provider == "deepseek" else None
        self.client = OpenAI(api_key=settings.api_key, base_url=base_url)

    def chat(self, system: str, user: str) -> LLMResult:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
        )
        text = resp.choices[0].message.content or ""
        usage = resp.usage
        # DeepSeek renvoie prompt_tokens / completion_tokens comme OpenAI
        in_tok = getattr(usage, "prompt_tokens", None) or count_tokens(system + user)
        out_tok = getattr(usage, "completion_tokens", None) or count_tokens(text)
        return LLMResult(text=text, input_tokens=in_tok, output_tokens=out_tok)


# --------------------------------------------------------------------------- #
#  LLM simule (mode demo, sans cle)                                           #
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Reproduit le comportement attendu du vrai LLM de facon deterministe.

    Il lit le CONTEXTE injecte dans le prompt (le meme que celui envoye a
    DeepSeek) et fabrique une reponse a partir de ce contexte, afin que la
    demo montre un vrai comportement de RAG : reponse ancree si le contexte
    contient l'info, aveu d'ignorance sinon.
    """

    def chat(self, system: str, user: str) -> LLMResult:
        context = self._extract(user, "CONTEXTE")
        question = self._extract(user, "QUESTION") or user

        if not context.strip():
            answer = "INSUFFISANT : aucun document pertinent pour repondre."
        else:
            best = self._best_passage(context, question)
            if best:
                answer = (
                    f"D'apres la documentation : {best} "
                    "(reponse generee en mode demo)."
                )
            else:
                answer = "INSUFFISANT : le contexte ne couvre pas cette question."

        in_tok = count_tokens(system + user)
        out_tok = count_tokens(answer)
        return LLMResult(text=answer, input_tokens=in_tok, output_tokens=out_tok)

    @staticmethod
    def _extract(text: str, tag: str) -> str:
        m = re.search(rf"{tag}\s*:?\s*\n?(.*?)(?:\n[A-ZÉÈ]{{4,}}\s*:|\Z)", text, re.S)
        return m.group(1).strip() if m else ""

    @staticmethod
    def _best_passage(context: str, question: str) -> str:
        q_words = {w.lower() for w in re.findall(r"\w+", question) if len(w) > 3}
        best, best_score = "", 0
        for line in re.split(r"(?<=[.!?])\s+|\n", context):
            line = line.strip(" -•\t")
            if len(line) < 15:
                continue
            score = sum(1 for w in q_words if w in line.lower())
            if score > best_score:
                best, best_score = line, score
        return best if best_score > 0 else ""


def get_llm(settings: Settings) -> LLM:
    """Retourne le bon LLM selon la config (reel ou simule)."""
    if settings.use_fake_llm:
        return FakeLLM()
    return RealLLM(settings)
