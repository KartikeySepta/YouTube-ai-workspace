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
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import (
    WORKSPACES_DIR, EMBEDDING_MODEL,
    WITHIN_VIDEO_MERGE_THRESHOLD, CLAIM_CLUSTER_GRAY_ZONE_LOW,
    CROSS_VIDEO_ADJUDICATION_FLOOR, GEMINI_MODEL,
)

_model = None


def get_model():
    """Load the embedding model once and reuse it (loading is the slow part)."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _call_gemini(prompt: str) -> str:
    """Delegates to the centralized wrapper with 429 retry/backoff (core/llm.py)."""
    from core.llm import generate_content
    return generate_content(prompt)


def adjudicate_merge(claim_a: str, claim_b: str, context_a: str = "", context_b: str = "") -> tuple[bool, str]:
    """
    Ask the LLM whether two claims assert the SAME checkable thing and should merge.

    The bar is deliberately strict: merge ONLY if same assertion, same subject, AND same
    scope/reference population/timeframe. Claims that are merely topically similar — or that
    quote different reference populations (e.g. "global, all ages" vs "adults only") — must
    stay separate. Optional surrounding chunk text is provided because scope often lives in
    the context, not the claim sentence itself.

    Returns (should_merge, reason). On any error, defaults to (False, ...) — the conservative
    choice, since a false merge silently corrupts cross-video agreement counts.
    """
    ctx_a = f'\nContext for A: "{context_a}"' if context_a else ""
    ctx_b = f'\nContext for B: "{context_b}"' if context_b else ""
    prompt = f"""You decide whether two claims extracted from videos express the SAME single
checkable assertion (and should be MERGED into one cluster) or are DIFFERENT assertions
that must stay SEPARATE.

SCOPE RUBRIC (strict): Merge ONLY if they assert the same fact about the same subject AND
the same scope — same reference population, region/country, timeframe, and units. If they
differ in ANY of those — even when topically similar, sharing numbers, or clearly related —
do NOT merge. Two claims reporting a similar finding for DIFFERENT populations or regions
are a distinct synthesis category (they may agree or corroborate at the synthesis layer),
but they are NOT the same claim and must not be collapsed into one.
Examples of NON-merges:
  - "ADHD affects 5-8% of the global population (all ages)" vs "~4% of adults have ADHD"
    (different reference population: all-ages global vs adults).
  - "adult ADHD prevalence in Canada is ~4.4%" vs "~4% of US adults have ADHD"
    (same metric, DIFFERENT region).

Claim A: "{claim_a}"{ctx_a}
Claim B: "{claim_b}"{ctx_b}

Return ONLY JSON, no markdown fences: {{"merge": true or false, "reason": "one sentence"}}"""

    try:
        raw = _call_gemini(prompt).strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        return bool(parsed.get("merge", False)), str(parsed.get("reason", ""))
    except Exception as e:
        return False, f"adjudication error (defaulting to no-merge): {e}"


def cluster_claims(
    claims: list[dict],
    within_threshold: float = WITHIN_VIDEO_MERGE_THRESHOLD,
    gray_low: float = CLAIM_CLUSTER_GRAY_ZONE_LOW,
    cross_floor: float = CROSS_VIDEO_ADJUDICATION_FLOOR,
    adjudicate: bool = False,
    context_by_claim_id: dict | None = None,
    provenance_by_claim_id: dict | None = None,
):
    """
    Greedy clustering with a SPLIT merge policy (Step 2): within-video vs cross-video are
    treated differently, decided by whether the incoming claim's video is already present in
    the best-matching cluster.

    WITHIN-VIDEO (unchanged):
      - cosine >= within_threshold (0.87)         -> auto-merge
      - gray_low (0.80) <= cosine < within_threshold and adjudicate -> LLM decides
      - otherwise                                  -> new cluster

    CROSS-VIDEO (new):
      - NO auto-merge at ANY score. Cosine is only a pre-filter.
      - cosine >= cross_floor (0.75) and adjudicate -> LLM decides (merge only if it says yes)
      - cosine < cross_floor                        -> split automatically, no call

    Returns (cluster_assignments, cluster_summaries, stats) where stats counts adjudication
    calls split by within/cross so the cost of the wider cross-video scope is visible.
    """
    model = get_model()
    texts = [c["claim"] for c in claims]
    embeddings = model.encode(texts, show_progress_bar=False)
    context_by_claim_id = context_by_claim_id or {}
    provenance_by_claim_id = provenance_by_claim_id or {}

    def video_of(idx):
        cid = claims[idx]["claim_id"]
        return provenance_by_claim_id.get(cid) or claims[idx].get("video_id")

    clusters = []  # each: {"cluster_id", "member_indices", "centroid", "rep_index", "videos": set}
    assignments = [None] * len(claims)
    stats = {
        "within_auto_merges": 0,
        "within_adjudications": 0,
        "cross_adjudications": 0,
        "cross_merges": 0,
    }

    for i, emb in enumerate(embeddings):
        best_cluster = None
        best_sim = -1.0
        for cluster in clusters:
            sim = cosine_sim(emb, cluster["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        should_merge = False
        if best_cluster is not None:
            rep = best_cluster["rep_index"]
            is_within = video_of(i) in best_cluster["videos"]

            if is_within:
                if best_sim >= within_threshold:
                    should_merge = True
                    stats["within_auto_merges"] += 1
                elif best_sim >= gray_low and adjudicate:
                    stats["within_adjudications"] += 1
                    merge, _reason = adjudicate_merge(
                        claims[i]["claim"], claims[rep]["claim"],
                        context_by_claim_id.get(claims[i]["claim_id"], ""),
                        context_by_claim_id.get(claims[rep]["claim_id"], ""),
                    )
                    should_merge = merge
            else:
                # CROSS-VIDEO: never auto-merge; adjudicate everything at/above the floor.
                if best_sim >= cross_floor and adjudicate:
                    stats["cross_adjudications"] += 1
                    merge, _reason = adjudicate_merge(
                        claims[i]["claim"], claims[rep]["claim"],
                        context_by_claim_id.get(claims[i]["claim_id"], ""),
                        context_by_claim_id.get(claims[rep]["claim_id"], ""),
                    )
                    should_merge = merge
                    if merge:
                        stats["cross_merges"] += 1

        if should_merge:
            best_cluster["member_indices"].append(i)
            n = len(best_cluster["member_indices"])
            # running average centroid — keeps the cluster's "center of meaning" updated
            best_cluster["centroid"] = (best_cluster["centroid"] * (n - 1) + emb) / n
            best_cluster["videos"].add(video_of(i))
            assignments[i] = best_cluster["cluster_id"]
        else:
            cluster_id = f"cluster_{len(clusters):04d}"
            clusters.append({
                "cluster_id": cluster_id, "member_indices": [i], "centroid": emb,
                "rep_index": i, "videos": {video_of(i)},
            })
            assignments[i] = cluster_id

    cluster_summaries = [
        {"cluster_id": c["cluster_id"], "size": len(c["member_indices"]),
         "member_claim_ids": [claims[idx]["claim_id"] for idx in c["member_indices"]]}
        for c in clusters
    ]

    return assignments, cluster_summaries, stats


def run_clustering_for_workspace(workspace_id: str, adjudicate: bool = True):
    claims_path = Path(WORKSPACES_DIR) / workspace_id / "claims.json"
    with open(claims_path) as f:
        claims = json.load(f)

    # Build claim_id -> source-chunk text, so the gray-zone adjudicator can see the scope/
    # reference-population context that often lives in the surrounding text, not the claim itself.
    context_by_claim_id = {}
    provenance_by_claim_id = {}   # claim_id -> true source video (from evidence chunk)
    chunks_path = Path(WORKSPACES_DIR) / workspace_id / "chunks.json"
    if chunks_path.exists():
        chunk_text = {c["chunk_id"]: c["text"] for c in json.load(open(chunks_path))}
        for c in claims:
            ev = c.get("evidence") or []
            if ev:
                context_by_claim_id[c["claim_id"]] = chunk_text.get(ev[0]["chunk_id"], "")
                provenance_by_claim_id[c["claim_id"]] = ev[0]["chunk_id"].rsplit("_c", 1)[0]

    assignments, cluster_summaries, stats = cluster_claims(
        claims, adjudicate=adjudicate,
        context_by_claim_id=context_by_claim_id,
        provenance_by_claim_id=provenance_by_claim_id,
    )

    # Attach cluster_id to each claim WITHOUT touching anything else about it
    for claim, cluster_id in zip(claims, assignments):
        claim["cluster_id"] = cluster_id

    claims_out_path = Path(WORKSPACES_DIR) / workspace_id / "claims.json"
    with open(claims_out_path, "w") as f:
        json.dump(claims, f, indent=2)

    clusters_out_path = Path(WORKSPACES_DIR) / workspace_id / "clusters.json"
    with open(clusters_out_path, "w") as f:
        json.dump(cluster_summaries, f, indent=2)

    return claims, cluster_summaries, stats


if __name__ == "__main__":
    workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"
    claims, clusters, stats = run_clustering_for_workspace(workspace_id)

    multi_member_clusters = [c for c in clusters if c["size"] > 1]

    print(f"{len(claims)} claims -> {len(clusters)} clusters")
    print(f"({len(multi_member_clusters)} clusters have more than 1 member — real grouping happened)")
    print(f"adjudications: within={stats['within_adjudications']} cross={stats['cross_adjudications']} "
          f"| cross-merges kept={stats['cross_merges']}\n")

    for c in sorted(multi_member_clusters, key=lambda x: -x["size"]):
        print(f"--- {c['cluster_id']} ({c['size']} members) ---")
        for claim_id in c["member_claim_ids"]:
            claim_text = next(cl["claim"] for cl in claims if cl["claim_id"] == claim_id)
            print(f"  [{claim_id}] {claim_text}")
        print()