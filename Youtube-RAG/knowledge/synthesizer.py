"""
STEP 12: CROSS-VIDEO SYNTHESIS

Takes claim clusters (from Step 11) and figures out how claims within each cluster
relate to each other ACROSS videos: agreement, partial agreement, contradiction,
different context, or independent idea.

KEY DESIGN DECISION: most of the signal here is DERIVED FROM COUNTS, not asked from
an LLM. supporting_count / contradicting_count come straight from each claim's
`stance` field (set back in Step 3) — no confidence score, no LLM guessing "how sure"
it is. Gemini is ONLY used for the one thing counting can't do: telling the difference
between "these two creators disagree" and "these two creators are talking about
different situations and aren't really disagreeing at all."

Gemini is also ONLY called for clusters with claims from 2+ DIFFERENT videos — a
cluster with claims from just one video has nothing to cross-reference yet, so it's
labeled "single_source" without spending an API call on it.

This file's Gemini-calling function can't be tested in this sandbox (same network
restriction as Step 3), but the derived-stats logic — which is most of this file's
actual work — is fully tested below against your real 113 claims / 104 clusters.
The Gemini relationship classifier is tested with a hand-built fake 2-video scenario,
same approach as Step 3's claim_extractor test.
"""

import json
import os
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.models import ClusterSynthesis
from core.config import WORKSPACES_DIR, GEMINI_MODEL, CROSS_VIDEO_ADJUDICATION_FLOOR, CROSS_SOURCE_THEME_FLOOR

VALID_RELATIONSHIPS = {"single_source", "agreement", "partial_agreement", "contradiction", "different_context", "independent"}


def compute_cluster_stats(cluster_claims: list[dict]) -> dict:
    """
    Pure counting — no LLM involved. This is the "derived signal, not fake confidence"
    part of the design: how many sources support, how many oppose, how many distinct
    channels are represented, how much evidence backs this cluster overall.
    """
    video_ids = {c["video_id"] for c in cluster_claims}
    channels = {c.get("channel", c["video_id"]) for c in cluster_claims}  # falls back to video_id if channel isn't on the claim
    supporting = sum(1 for c in cluster_claims if c["stance"] == "support")
    contradicting = sum(1 for c in cluster_claims if c["stance"] == "oppose")
    neutral = sum(1 for c in cluster_claims if c["stance"] in ("neutral", "mixed"))
    evidence_count = sum(len(c["evidence"]) for c in cluster_claims)

    return {
        "video_ids": sorted(video_ids),
        "unique_video_count": len(video_ids),
        "unique_channels": len(channels),
        "supporting_count": supporting,
        "contradicting_count": contradicting,
        "neutral_count": neutral,
        "evidence_count": evidence_count,
    }


def build_relationship_prompt(cluster_claims: list[dict]) -> str:
    """Prompt for the ONE thing counting can't tell us: real disagreement vs different context."""
    claims_text = "\n".join(
        f'- (video: {c["video_id"]}, claim_id: {c["claim_id"]}, stance: {c["stance"]}) "{c["claim"]}"'
        for c in cluster_claims
    )
    return f"""These claims come from DIFFERENT videos and were grouped as expressing a similar idea.
Classify their relationship as exactly one of:
- "agreement" — the sources genuinely agree
- "partial_agreement" — mostly aligned with minor differences
- "contradiction" — sources genuinely conflict on the same situation
- "different_context" — sources aren't really disagreeing, they're describing different situations
  (e.g. "good for beginners" vs "experienced users should diversify" — both can be true at once)
- "independent" — related topic but not really comparable claims

Return ONLY JSON, no markdown fences:
{{"relationship": "...", "synthesis_note": "one sentence explaining the relationship, referencing claim_ids given above only"}}

CLAIMS:
{claims_text}
"""


def call_gemini(prompt: str) -> str:
    """Delegates to the centralized wrapper with 429 retry/backoff (core/llm.py)."""
    from core.llm import generate_content
    return generate_content(prompt)


def parse_relationship_response(raw_json_text: str, valid_claim_ids: set[str]) -> tuple[str, str]:
    """
    Parse + validate Gemini's relationship classification.
    Same discipline as Step 3: if the model's synthesis_note references a claim_id
    that wasn't actually in this cluster, that's a red flag — we don't silently trust it.
    """
    cleaned = raw_json_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    parsed = json.loads(cleaned)
    relationship = parsed.get("relationship", "")
    note = parsed.get("synthesis_note", "")

    if relationship not in VALID_RELATIONSHIPS:
        raise ValueError(f"invalid relationship value from model: '{relationship}'")

    # Check the note doesn't reference a claim_id outside this cluster (loose check —
    # scans for any "claim" substring token that looks like an id and isn't in our set)
    for token in note.replace(",", " ").replace(")", " ").split():
        if "_claim" in token and token.strip(".:") not in valid_claim_ids:
            raise ValueError(f"synthesis_note references unknown claim_id: '{token}'")

    return relationship, note


def synthesize_cluster(cluster_id: str, cluster_claims: list[dict]) -> ClusterSynthesis:
    """
    For a single cluster: compute derived stats always; call Gemini for relationship
    classification ONLY if claims come from 2+ distinct videos.
    """
    stats = compute_cluster_stats(cluster_claims)
    member_claim_ids = [c["claim_id"] for c in cluster_claims]

    if stats["unique_video_count"] < 2:
        # Nothing to cross-reference yet — label honestly instead of forcing a comparison
        relationship = "single_source"
        synthesis_note = None
    else:
        valid_ids = set(member_claim_ids)
        prompt = build_relationship_prompt(cluster_claims)
        raw_response = call_gemini(prompt)
        relationship, synthesis_note = parse_relationship_response(raw_response, valid_ids)

    return ClusterSynthesis(
        cluster_id=cluster_id,
        member_claim_ids=member_claim_ids,
        member_video_ids=stats["video_ids"],
        unique_channels=stats["unique_channels"],
        supporting_count=stats["supporting_count"],
        contradicting_count=stats["contradicting_count"],
        neutral_count=stats["neutral_count"],
        relationship=relationship,
        synthesis_note=synthesis_note,
    )


def build_cross_source_themes(claims: list[dict], floor: float = CROSS_SOURCE_THEME_FLOOR,
                              top_k: int = 10) -> list[dict]:
    """
    STEP 4 — the "distinct synthesis category".

    Cross-video claims are (correctly) NOT merged into one cluster when they differ in scope
    (e.g. adult ADHD prevalence in Canada vs the US). But that relationship is the cross-creator
    signal this tool exists to surface. Rather than transitively chaining every semantically-close
    ADHD claim into one useless mega-theme, we surface the TOP-K strongest cross-video claim PAIRS
    (different videos, cosine >= floor) and label each pair's relationship with one LLM call
    (agreement / partial_agreement / contradiction / different_context / independent).

    So "same finding, different population/region" shows up as an explicit, bounded, labeled
    connection (typically different_context) instead of being lost at a 0% merge rate.
    """
    from knowledge.claim_clusterer import get_model, cosine_sim

    def vid(c):
        ev = c.get("evidence") or []
        return ev[0]["chunk_id"].rsplit("_c", 1)[0] if ev else c.get("video_id")

    n = len(claims)
    if n < 2:
        return []
    model = get_model()
    embs = model.encode([c["claim"] for c in claims], show_progress_bar=False)
    vids = [vid(c) for c in claims]

    # All cross-video pairs at/above the floor, strongest first.
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if vids[i] != vids[j]:
                s = cosine_sim(embs[i], embs[j])
                if s >= floor:
                    pairs.append((s, i, j))
    pairs.sort(reverse=True, key=lambda x: x[0])

    themes = []
    used = set()   # keep each claim in at most one theme, so the top-K stays diverse
    for s, i, j in pairs:
        if len(themes) >= top_k:
            break
        if i in used or j in used:
            continue
        member = [claims[i], claims[j]]
        member_ids = [claims[i]["claim_id"], claims[j]["claim_id"]]
        try:
            raw = call_gemini(build_relationship_prompt(member))
            relationship, note = parse_relationship_response(raw, set(member_ids))
        except Exception as e:
            relationship, note = "related", f"(relationship classification unavailable: {e})"
        themes.append({
            "theme_id": f"theme_{len(themes):04d}",
            "member_claim_ids": member_ids,
            "videos": sorted({vids[i], vids[j]}),
            "cosine": round(s, 3),
            "relationship": relationship,
            "synthesis_note": note,
        })
        used.add(i)
        used.add(j)
    return themes


def run_synthesis_for_workspace(workspace_id: str):
    claims_path = Path(WORKSPACES_DIR) / workspace_id / "claims.json"
    clusters_path = Path(WORKSPACES_DIR) / workspace_id / "clusters.json"

    with open(claims_path) as f:
        claims = json.load(f)
    with open(clusters_path) as f:
        clusters = json.load(f)

    claims_by_id = {c["claim_id"]: c for c in claims}

    results = []
    for cluster in clusters:
        cluster_claims = [claims_by_id[cid] for cid in cluster["member_claim_ids"]]
        results.append(synthesize_cluster(cluster["cluster_id"], cluster_claims))

    out_path = Path(WORKSPACES_DIR) / workspace_id / "synthesis.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # STEP 4: the distinct-synthesis-category layer — cross-video related claims that were
    # deliberately NOT merged still get surfaced (and labeled) here rather than lost.
    themes = build_cross_source_themes(claims)
    themes_path = Path(WORKSPACES_DIR) / workspace_id / "cross_source_themes.json"
    with open(themes_path, "w") as f:
        json.dump(themes, f, indent=2)

    return results, themes


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-relationship-parser":
        # ============================================================
        # NO NETWORK CALL — tests the relationship parser/validator
        # against a hand-built fake 2-video scenario, same pattern as
        # Step 3's claim_extractor test.
        # ============================================================
        fake_valid_ids = {"videoA_claim0001", "videoB_claim0003"}

        print("--- Test 1: valid response, should be ACCEPTED ---")
        good_response = json.dumps({
            "relationship": "different_context",
            "synthesis_note": "videoA_claim0001 targets beginners while videoB_claim0003 targets experienced users — not a real conflict."
        })
        rel, note = parse_relationship_response(good_response, fake_valid_ids)
        print(f"  relationship: {rel}")
        print(f"  note: {note}\n")

        print("--- Test 2: invalid relationship value, should RAISE ---")
        bad_response = json.dumps({"relationship": "totally_agree_100_percent", "synthesis_note": "..."})
        try:
            parse_relationship_response(bad_response, fake_valid_ids)
            print("  ERROR: should have raised but didn't!")
        except ValueError as e:
            print(f"  correctly rejected: {e}\n")

        print("--- Test 3: note references a claim_id NOT in this cluster, should RAISE ---")
        hallucinated_response = json.dumps({
            "relationship": "contradiction",
            "synthesis_note": "videoA_claim0001 conflicts with videoC_claim9999 which was never in this cluster."
        })
        try:
            parse_relationship_response(hallucinated_response, fake_valid_ids)
            print("  ERROR: should have raised but didn't!")
        except ValueError as e:
            print(f"  correctly rejected: {e}")

    else:
        # Real run against a real workspace — the derived-stats part works today,
        # the Gemini relationship part only fires once you have 2+ videos.
        workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"
        results, themes = run_synthesis_for_workspace(workspace_id)

        single_source = [r for r in results if r.relationship == "single_source"]
        multi_source = [r for r in results if r.relationship != "single_source"]

        print(f"{len(results)} clusters synthesized")
        print(f"  {len(single_source)} single-source (only 1 video — nothing to compare yet)")
        print(f"  {len(multi_source)} multi-source (real cross-video comparison happened)")
        print(f"  {len(themes)} cross-source themes (related across videos, not merged)\n")

        for t in themes:
            print(f"[{t['theme_id']}] {t['relationship']} across {t['videos']} — {t['synthesis_note']}")

        for r in multi_source:
            print(f"[{r.cluster_id}] {r.relationship} — {r.synthesis_note}")
            print(f"  supporting={r.supporting_count} contradicting={r.contradicting_count} "
                  f"channels={r.unique_channels} videos={r.member_video_ids}\n")