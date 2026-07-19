"""
STEP 5: VECTOR STORE

Wraps Qdrant behind a small interface so nothing else in the project calls Qdrant
directly — if you ever swap vector stores, this is the only file that changes.

THE ONE RULE THAT MATTERS MOST IN THIS FILE:
Every chunk stored here carries a workspace_id in its payload, and every search
is FILTERED by workspace_id. This is what stops one research topic's videos from
leaking into another topic's answers. It's tested explicitly below — two fake
workspaces, and a query in workspace A must NEVER return workspace B's data.

Uses Qdrant's local mode by default (a file on disk, no server needed) — good for
solo/small-scale use. Switch to a server URL later only if you actually need it.
"""

import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieval.embeddings import embed_text, embed_batch, embed_query

COLLECTION_NAME = "transcript_chunks"
VECTOR_SIZE = 384   # bge-small-en-v1.5 output dimension — update if you change EMBEDDING_MODEL

_client = None


def get_client(path: str = "data/qdrant_db"):
    """Local on-disk Qdrant — no server process needed. Reused across calls."""
    global _client
    if _client is None:
        _client = QdrantClient(path=path)
        if not _client.collection_exists(COLLECTION_NAME):
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
    return _client


def upsert_chunks(chunks: list[dict], client: QdrantClient = None):
    """
    Store chunks with their embeddings + full metadata payload.
    Each chunk dict must have: chunk_id, workspace_id, video_id, text, start_seconds,
    end_seconds, channel, video_title (matches TranscriptChunk from Step 2).
    """
    client = client or get_client()
    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts)

    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(PointStruct(
            id=abs(hash(chunk["chunk_id"])) % (10 ** 15),  # Qdrant needs int/uuid ids; deterministic from chunk_id
            vector=vector,
            payload={
                "chunk_id": chunk["chunk_id"],
                "workspace_id": chunk["workspace_id"],
                "video_id": chunk["video_id"],
                "video_title": chunk.get("video_title", ""),
                "channel": chunk.get("channel", ""),
                "start_seconds": chunk.get("start_seconds"),
                "end_seconds": chunk.get("end_seconds"),
                "text": chunk["text"],
            },
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def search(query: str, workspace_id: str, top_k: int = 10, client: QdrantClient = None) -> list[dict]:
    """
    Search chunks by meaning, STRICTLY scoped to one workspace_id.
    Returns list of {chunk_id, video_id, text, start_seconds, end_seconds, score, ...}
    """
    client = client or get_client()
    query_vector = embed_query(query)   # NOT embed_text — queries need the BGE instruction prefix

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=Filter(
            must=[FieldCondition(key="workspace_id", match=MatchValue(value=workspace_id))]
        ),
        limit=top_k,
    )

    return [
        {**point.payload, "score": point.score}
        for point in results.points
    ]


if __name__ == "__main__":
    # ============================================================
    # Full test using Qdrant's in-memory mode — no disk, no server.
    # Proves workspace isolation actually works before you trust it.
    # ============================================================
    test_client = QdrantClient(":memory:")
    test_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    fake_chunks = [
        {"chunk_id": "wsA_c1", "workspace_id": "workspace_A", "video_id": "vidA",
         "video_title": "Fiverr video", "channel": "Channel A",
         "start_seconds": 0, "end_seconds": 60, "text": "Don't rely only on Fiverr for freelance income"},
        {"chunk_id": "wsB_c1", "workspace_id": "workspace_B", "video_id": "vidB",
         "video_title": "Cooking video", "channel": "Channel B",
         "start_seconds": 0, "end_seconds": 60, "text": "Here's how to make great pasta at home"},
    ]

    n = upsert_chunks(fake_chunks, client=test_client)
    print(f"Upserted {n} chunks across 2 fake workspaces\n")

    print("--- Query 'freelance income advice' scoped to workspace_A ---")
    results_a = search("freelance income advice", workspace_id="workspace_A", top_k=5, client=test_client)
    for r in results_a:
        print(f"  [{r['workspace_id']}] {r['text']} (score={r['score']:.3f})")

    print("\n--- Query 'freelance income advice' scoped to workspace_B (should find NOTHING relevant) ---")
    results_b = search("freelance income advice", workspace_id="workspace_B", top_k=5, client=test_client)
    for r in results_b:
        print(f"  [{r['workspace_id']}] {r['text']} (score={r['score']:.3f})")

    print("\n--- CHECK: did workspace_B leak workspace_A's data? ---")
    leaked = any(r["workspace_id"] == "workspace_A" for r in results_b)
    print("LEAK DETECTED - BUG!" if leaked else "NO LEAK - isolation working correctly")