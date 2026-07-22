"""
STEP 7: FUSION (Reciprocal Rank Fusion)

You now have two different rankings for the same query:
  - vector search (Step 5): good at "similar meaning", weak at exact terms
  - BM25 (Step 6): good at exact terms, has no idea about meaning at all

Fusion combines both into one ranked list. An item scores well if it ranks well in
EITHER list, and especially well if it ranks well in BOTH.

Formula: Reciprocal Rank Fusion (RRF) — simple, standard, no tuning needed.
  score(item) = sum over each ranked list of  1 / (k + rank_in_that_list)
  (k=60 is the standard constant used in the original RRF paper — it just softens
  the impact of small rank differences near the top; you don't need to tune it.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import WORKSPACES_DIR, VECTOR_TOP_K, BM25_TOP_K
from retrieval.bm25 import load_workspace_chunks, build_bm25_index, bm25_search
from retrieval.vector_store import search as vector_search


def reciprocal_rank_fusion(*ranked_lists: list[dict], id_key: str = "chunk_id", k: int = 60) -> list[dict]:
    """
    Takes any number of ranked lists (each a list of dicts with a shared id_key),
    fuses them into one ranked list by RRF score.

    Each input list should already be in rank order (best first) — this function
    reads POSITION in the list as the rank, not any score field.
    """
    rrf_scores = {}
    item_lookup = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list):
            item_id = item[id_key]
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            item_lookup[item_id] = item   # keep one copy of the full item data

    fused_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])
    return [{**item_lookup[item_id], "rrf_score": rrf_scores[item_id]} for item_id in fused_ids]


def hybrid_search(query: str, workspace_id: str, top_k: int = 10) -> list[dict]:
    """
    Full hybrid retrieval: run vector + BM25 in parallel, fuse with RRF.
    Returns the fused, ranked list — this is what Step 8 (reranking) will take as input.
    """
    chunks = load_workspace_chunks(workspace_id)
    bm25_index, chunks = build_bm25_index(chunks)

    vector_results = vector_search(query, workspace_id=workspace_id, top_k=VECTOR_TOP_K)
    bm25_results = bm25_search(bm25_index, chunks, query, top_k=BM25_TOP_K)

    fused = reciprocal_rank_fusion(vector_results, bm25_results, id_key="chunk_id")
    return fused[:top_k]


if __name__ == "__main__":
    workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"
    query = sys.argv[2] if len(sys.argv) > 2 else "how does RAG work"

    chunks = load_workspace_chunks(workspace_id)
    bm25_index, chunks = build_bm25_index(chunks)

    vector_results = vector_search(query, workspace_id=workspace_id, top_k=VECTOR_TOP_K)
    bm25_results = bm25_search(bm25_index, chunks, query, top_k=BM25_TOP_K)

    print(f"Query: '{query}'\n")
    print("--- Vector search ranking ---")
    for i, r in enumerate(vector_results[:5]):
        print(f"  #{i+1} chunk {r['chunk_id']} (score={r['score']:.3f}): {r['text'][:70]}...")

    print("\n--- BM25 ranking ---")
    for i, r in enumerate(bm25_results[:5]):
        print(f"  #{i+1} chunk {r['chunk_id']} (score={r['bm25_score']:.3f}): {r['text'][:70]}...")

    fused = reciprocal_rank_fusion(vector_results, bm25_results, id_key="chunk_id")
    print("\n--- FUSED ranking (RRF) ---")
    for i, r in enumerate(fused[:5]):
        print(f"  #{i+1} chunk {r['chunk_id']} (rrf={r['rrf_score']:.4f}): {r['text'][:70]}...")