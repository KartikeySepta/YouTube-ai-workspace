"""
STEP 8: RERANKING

Fusion (Step 7) just showed its real limitation on your data: it ranked an
irrelevant "Wikipedia reminders" chunk as high as the genuinely relevant "how RAG
works" chunk, because RRF only sees RANK POSITION, not how good a match actually is.

A cross-encoder fixes this differently: instead of comparing pre-computed embeddings,
it reads the query AND each candidate's actual text TOGETHER and scores relevance
directly. Slower (so we only run it on the top ~20-30 fused candidates, never the
whole corpus), but much better at exactly this "these look similar in rank but one
is actually irrelevant" problem.
"""

import sys
from pathlib import Path

from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import RERANKER_MODEL, RERANK_KEEP_TOP

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, candidates: list[dict], top_k: int = RERANK_KEEP_TOP) -> list[dict]:
    """
    Takes the fused candidate list (Step 7's output, ~20-30 items) and returns the
    real top `top_k`, reordered by actual cross-encoder relevance — not rank fusion.
    """
    if not candidates:
        return []

    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: -x[1])

    return [{**c, "rerank_score": float(s)} for c, s in scored[:top_k]]


if __name__ == "__main__":
    from retrieval.hybrid import hybrid_search
    from core.config import VECTOR_TOP_K

    workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"
    query = sys.argv[2] if len(sys.argv) > 2 else "how does RAG work"

    fused = hybrid_search(query, workspace_id=workspace_id, top_k=VECTOR_TOP_K)
    print(f"Query: '{query}'\n")
    print("--- Fused ranking (input to reranker) ---")
    for i, r in enumerate(fused[:5]):
        print(f"  #{i+1} chunk {r['chunk_id']} (rrf={r['rrf_score']:.4f}): {r['text'][:70]}...")

    reranked = rerank(query, fused, top_k=5)
    print("\n--- RERANKED (cross-encoder) ---")
    for i, r in enumerate(reranked):
        print(f"  #{i+1} chunk {r['chunk_id']} (rerank_score={r['rerank_score']:.3f}): {r['text'][:70]}...")