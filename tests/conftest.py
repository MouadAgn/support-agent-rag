import sys
from pathlib import Path

# rend le package importable + force le mode demo pour les tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import os

os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("EMBEDDINGS_BACKEND", "tfidf")
