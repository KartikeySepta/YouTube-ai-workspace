"""
STEP 4: EMBEDDINGS

One function to turn text into vectors, using the same model already proven to work
in Step 11 (claim clustering). Kept as a thin wrapper so if you ever swap embedding
models, you change it in ONE place, not scattered across chunker/vector_store/chat.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import EMBEDDING_MODEL

_model = None


def get_model():
    """Load once, reuse — loading the model from disk/cache is the slow part."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_text(text: str) -> list[float]:
    """Embed a single string (used for CHUNKS/passages — no instruction prefix)."""
    model = get_model()
    vector = model.encode(text, show_progress_bar=False)
    return vector.tolist()


def embed_query(query: str) -> list[float]:
    """
    Embed a SEARCH QUERY. bge-small-en-v1.5 was trained asymmetrically: queries need
    this specific instruction prefix prepended to retrieve well, passages don't.
    Skipping this silently produces mediocre rankings with no error — it's the exact
    bug we caught by testing on a real query and noticing the wrong chunk won.
    """
    model = get_model()
    instruction = "Represent this sentence for searching relevant passages: "
    vector = model.encode(instruction + query, show_progress_bar=False)
    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many strings at once — much faster than calling embed_text() in a loop."""
    model = get_model()
    vectors = model.encode(texts, show_progress_bar=False)
    return vectors.tolist()


if __name__ == "__main__":
    # Quick manual check: embed 3 sentences, confirm similar ones score higher.
    import numpy as np

    texts = [
        "Don't rely only on Fiverr for freelance income",
        "Diversify your income sources away from Fiverr",
        "I had pizza for lunch today",
    ]
    vectors = embed_batch(texts)

    def cos_sim(a, b):
        a, b = np.array(a), np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    print("similarity(sentence 0, sentence 1) [should be HIGH - same idea]:", cos_sim(vectors[0], vectors[1]))
    print("similarity(sentence 0, sentence 2) [should be LOW - unrelated]:  ", cos_sim(vectors[0], vectors[2]))