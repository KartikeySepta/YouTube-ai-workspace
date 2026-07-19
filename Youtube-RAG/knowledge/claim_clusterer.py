"""
STEP 11: CLAIM CLUSTERING

Groups claims that express the same idea in different words, using the exact same
tool as Step 4 (embeddings + cosine similarity) — just pointed at claim text instead
of transcript chunk text.

This ALSO catches something we found by accident in Step 3: when chunks overlap
(by design, from Step 2), the same sentence can get extracted as a claim twice —
once from each overlapping chunk. Clustering merges those back into one claim
with multiple evidence references, instead of double-counting them as "2 sources agree".

Original claims are NEVER modified or deleted — clustering only ADDS a cluster_id
to each one, and separately records which claims belong to which cluster.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import WORKSPACES_DIR, EMBEDDING_MODEL, CLAIM_CLUSTER_SIMILARITY_THRESHOLD

_model = None


def get_model():
    """Load the embedding model once and reuse it (loading is the slow part)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cluster_claims(claims: list[dict], threshold: float = CLAIM_CLUSTER_SIMILARITY_THRESHOLD):
    """
    Simple greedy clustering:
    - Embed every claim's text.
    - Walk through claims in order. For each one, compare it to every existing
      cluster's running-average embedding (centroid).
    - If the best match is above `threshold`, join that cluster (and update its centroid).
    - Otherwise, start a new cluster.

    Returns:
      cluster_assignments: list of cluster_id, same order/length as `claims`
      clusters: list of {cluster_id, member_indices, size}
    """
    model = get_model()
    texts = [c["claim"] for c in claims]
    embeddings = model.encode(texts, show_progress_bar=False)

    clusters = []  # each: {"cluster_id": str, "member_indices": [int], "centroid": np.ndarray}
    assignments = [None] * len(claims)

    for i, emb in enumerate(embeddings):
        best_cluster = None
        best_sim = -1.0

        for cluster in clusters:
            sim = cosine_sim(emb, cluster["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster is not None and best_sim >= threshold:
            best_cluster["member_indices"].append(i)
            n = len(best_cluster["member_indices"])
            # running average centroid — keeps the cluster's "center of meaning" updated
            best_cluster["centroid"] = (best_cluster["centroid"] * (n - 1) + emb) / n
            assignments[i] = best_cluster["cluster_id"]
        else:
            cluster_id = f"cluster_{len(clusters):04d}"
            clusters.append({"cluster_id": cluster_id, "member_indices": [i], "centroid": emb})
            assignments[i] = cluster_id

    cluster_summaries = [
        {"cluster_id": c["cluster_id"], "size": len(c["member_indices"]),
         "member_claim_ids": [claims[idx]["claim_id"] for idx in c["member_indices"]]}
        for c in clusters
    ]

    return assignments, cluster_summaries


def run_clustering_for_workspace(workspace_id: str):
    claims_path = Path(WORKSPACES_DIR) / workspace_id / "claims.json"
    with open(claims_path) as f:
        claims = json.load(f)

    assignments, cluster_summaries = cluster_claims(claims)

    # Attach cluster_id to each claim WITHOUT touching anything else about it
    for claim, cluster_id in zip(claims, assignments):
        claim["cluster_id"] = cluster_id

    claims_out_path = Path(WORKSPACES_DIR) / workspace_id / "claims.json"
    with open(claims_out_path, "w") as f:
        json.dump(claims, f, indent=2)

    clusters_out_path = Path(WORKSPACES_DIR) / workspace_id / "clusters.json"
    with open(clusters_out_path, "w") as f:
        json.dump(cluster_summaries, f, indent=2)

    return claims, cluster_summaries


if __name__ == "__main__":
    workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"
    claims, clusters = run_clustering_for_workspace(workspace_id)

    multi_member_clusters = [c for c in clusters if c["size"] > 1]

    print(f"{len(claims)} claims -> {len(clusters)} clusters")
    print(f"({len(multi_member_clusters)} clusters have more than 1 member — real grouping happened)\n")

    for c in sorted(multi_member_clusters, key=lambda x: -x["size"]):
        print(f"--- {c['cluster_id']} ({c['size']} members) ---")
        for claim_id in c["member_claim_ids"]:
            claim_text = next(cl["claim"] for cl in claims if cl["claim_id"] == claim_id)
            print(f"  [{claim_id}] {claim_text}")
        print()