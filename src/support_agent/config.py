"""Configuration centrale, chargee depuis les variables d'environnement (.env)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Racine du projet (…/support-agent-rag)
ROOT = Path(__file__).resolve().parents[2]

# Charge le .env s'il existe (sinon on reste sur les valeurs par defaut / demo)
load_dotenv(ROOT / ".env")


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "oui"}


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    # --- LLM ---
    provider: str = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    model: str = os.getenv("LLM_MODEL", "deepseek-chat").strip()
    base_url: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip()
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "").strip()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    demo_mode: bool = _get_bool("DEMO_MODE", False)

    # --- RAG ---
    top_k: int = _get_int("TOP_K", 4)
    score_threshold: float = _get_float("SCORE_THRESHOLD", 0.20)
    chunk_size: int = _get_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 120)

    # --- Embeddings ---
    embeddings_backend: str = os.getenv("EMBEDDINGS_BACKEND", "sentence-transformers").strip()
    # Modele multilingue : le corpus est en francais. all-MiniLM-L6-v2, entraine
    # sur de l'anglais, matchait sur la ressemblance des mots ("compta" ->
    # "compte") plutot que sur le sens. Rappel@4 mesure : 11/20 -> 17/20.
    embeddings_model: str = os.getenv(
        "EMBEDDINGS_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ).strip()

    # --- Couts (EUR) ---
    price_input_per_m: float = _get_float("PRICE_INPUT_PER_M", 0.26)
    price_output_per_m: float = _get_float("PRICE_OUTPUT_PER_M", 1.02)
    cost_per_ticket: float = _get_float("COST_PER_TICKET", 5.00)
    cost_per_call: float = _get_float("COST_PER_CALL", 6.50)

    # --- Chemins ---
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    index_dir: Path = field(default_factory=lambda: ROOT / "data" / "index")
    faq_dir: Path = field(default_factory=lambda: ROOT / "data" / "faq")
    pdf_dir: Path = field(default_factory=lambda: ROOT / "data" / "pdf")

    @property
    def api_key(self) -> str:
        return self.deepseek_api_key if self.provider == "deepseek" else self.openai_api_key

    @property
    def use_fake_llm(self) -> bool:
        """Vrai si on tourne sans vraie cle -> LLM simule (mode demo)."""
        return self.demo_mode or not self.api_key


def get_settings() -> Settings:
    return Settings()
