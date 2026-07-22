"""
STEP 6: BM25 KEYWORD SEARCH

Vector search just showed a real weakness: with 9 closely-related chunks, it couldn't
confidently tell "how RAG actually works" (dense with exact terms like tokenizer,
embedding, cosine similarity) apart from chunks that only mention RAG in passing.

BM25 is old-school keyword scoring — no embeddings, no ML, just clever word counting
that rewards exact/rare term matches. It's not smarter than vector search, it's
DIFFERENT — which is exactly why Step 7 (fusion) combines both instead of picking one.

Built per-workspace, in memory. Cheap to rebuild whenever a new video is added.
"""

import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import WORKSPACES_DIR


def tokenize(text: str) -> list[str]:
    """Simple lowercase word tokenizer — good enough for BM25's purposes."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(chunks: list[dict]):
    """Returns (bm25_index, chunks) — chunks kept alongside since BM25 only knows token lists."""
    tokenized_corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, chunks


def bm25_search(bm25: BM25Okapi, chunks: list[dict], query: str, top_k: int = 10) -> list[dict]:
    """Returns chunks ranked by keyword overlap, each with a bm25_score attached."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

    results = []
    for i in ranked_indices:
        if scores[i] <= 0:
            continue  # zero score = no keyword overlap at all, don't pretend it's a match
        results.append({**chunks[i], "bm25_score": float(scores[i])})
    return results


def load_workspace_chunks(workspace_id: str) -> list[dict]:
    chunks_path = Path(WORKSPACES_DIR) / workspace_id / "chunks.json"
    with open(chunks_path) as f:
        return json.load(f)


if __name__ == "__main__":
    workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"
    query = sys.argv[2] if len(sys.argv) > 2 else "how does RAG work"

    chunks = load_workspace_chunks(workspace_id)
    bm25, chunks = build_bm25_index(chunks)
    results = bm25_search(bm25, chunks, query, top_k=5)

    print(f"Query: '{query}'\n")
    for r in results:
        print(f"[chunk {r['chunk_index']}] bm25_score={r['bm25_score']:.3f}")
        print(f"  {r['text'][:120]}...\n")