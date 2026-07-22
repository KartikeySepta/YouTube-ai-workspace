"""
STEP 3: CLAIM EXTRACTION

Sends chunks to Gemini, asks for atomic claims (one clean idea per claim),
each claim pointing back at the chunk_id it came from.

THE MOST IMPORTANT RULE IN THIS FILE:
Gemini can SUGGEST a chunk_id, but we NEVER trust it blindly. Every claim's
evidence chunk_id is checked against the real chunk_ids we sent in. If Gemini
references a chunk_id that doesn't exist (hallucinated, typo'd, or made up),
that claim is thrown away — never silently kept.

This file also works without ever calling Gemini: run it directly to see the
validation logic tested against a hand-built fake response with one bad claim
mixed in with good ones, so you can see the rejection actually happen.
"""

import json
import os
import sys
from collections import OrderedDict, defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.models import TranscriptChunk, Claim, EvidenceRef
from core.config import WORKSPACES_DIR, GEMINI_MODEL

VALID_CLAIM_TYPES = {"recommendation", "opinion", "fact", "prediction", "warning"}
VALID_STANCES = {"support", "oppose", "neutral", "mixed"}


def build_claim_prompt(chunks_batch: list[TranscriptChunk]) -> str:
    """
    Build the prompt sent to Gemini for one batch of chunks.
    Batching a few chunks per call is cheaper/faster than one chunk at a time.
    """
    chunk_blocks = []
    for c in chunks_batch:
        chunk_blocks.append(f'chunk_id: "{c.chunk_id}"\ntext: "{c.text}"')
    chunks_text = "\n\n".join(chunk_blocks)

    return f"""You are extracting atomic claims from video transcript chunks for a research tool.

RULES:
- Each claim must represent ONE independently checkable idea — not a vague summary.
- Every claim MUST reference the exact chunk_id it came from, using ONLY the chunk_ids given below.
- NEVER invent a chunk_id. NEVER invent evidence that isn't actually in the chunk text.
- Do NOT include a confidence score — it will be ignored if present.
- claim_type must be one of: recommendation, opinion, fact, prediction, warning
- stance must be one of: support, oppose, neutral, mixed

Return ONLY a JSON array, no markdown fences, no preamble. Each item:
{{
  "video_id": "...",
  "claim": "...",
  "claim_type": "...",
  "stance": "...",
  "evidence": [{{"chunk_id": "...", "evidence_text": "..."}}],
  "topics": ["..."]
}}

CHUNKS:
{chunks_text}
"""


def call_gemini(prompt: str) -> str:
    """Delegates to the centralized wrapper (per-task routing + fallback, core/llm.py)."""
    from core.llm import generate_content
    return generate_content(prompt, task="extraction")


def parse_and_validate_claims(raw_json_text: str, valid_chunk_ids: set[str], video_id: str, start_index: int = 0) -> tuple[list[Claim], list[dict]]:
    """
    Parse Gemini's JSON response and validate every claim's evidence.
    Returns (accepted_claims, rejected_claims_with_reasons).

    `start_index` is the GLOBAL claim counter — pass in how many claims have already
    been accepted across all previous batches, so claim_ids stay unique across the
    whole run instead of restarting at 0 in every batch.

    THIS is the function that actually protects you from hallucinated evidence —
    read it carefully, it's the safety net for the whole claims pipeline.
    """
    accepted = []
    rejected = []
    next_index = start_index

    # Strip markdown fences if Gemini added them despite instructions not to
    cleaned = raw_json_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    cleaned = cleaned.strip()

    try:
        raw_claims = json.loads(cleaned)
    except json.JSONDecodeError as e:
        rejected.append({"reason": f"invalid JSON from model: {e}", "raw": raw_json_text[:200]})
        return accepted, rejected

    for raw_claim in raw_claims:
        claim_text = raw_claim.get("claim", "")
        claim_type = raw_claim.get("claim_type", "")
        stance = raw_claim.get("stance", "")
        raw_evidence = raw_claim.get("evidence", [])

        # --- Validation 1: claim_type and stance must be from the allowed set ---
        if claim_type not in VALID_CLAIM_TYPES:
            rejected.append({"reason": f"invalid claim_type '{claim_type}'", "claim": claim_text})
            continue
        if stance not in VALID_STANCES:
            rejected.append({"reason": f"invalid stance '{stance}'", "claim": claim_text})
            continue

        # --- Validation 2: must have at least one evidence reference ---
        if not raw_evidence:
            rejected.append({"reason": "no evidence provided", "claim": claim_text})
            continue

        # --- Validation 3 (THE CRITICAL ONE): every chunk_id must actually exist ---
        evidence_refs = []
        claim_is_valid = True
        for ev in raw_evidence:
            chunk_id = ev.get("chunk_id", "")
            if chunk_id not in valid_chunk_ids:
                rejected.append({
                    "reason": f"evidence references chunk_id '{chunk_id}' which DOES NOT EXIST — hallucinated or invalid",
                    "claim": claim_text,
                })
                claim_is_valid = False
                break
            evidence_refs.append(EvidenceRef(
                chunk_id=chunk_id,
                evidence_text=ev.get("evidence_text", ""),
            ))

        if not claim_is_valid:
            continue

        # Passed all checks — accept it, using the GLOBAL counter for a unique id
        accepted.append(Claim(
            claim_id=f"{video_id}_claim{next_index:04d}",
            video_id=video_id,
            claim=claim_text,
            claim_type=claim_type,
            stance=stance,
            evidence=evidence_refs,
            topics=raw_claim.get("topics", []),
        ))
        next_index += 1

    return accepted, rejected


def extract_claims_for_chunks(
    chunks: list[TranscriptChunk],
    batch_size: int = 3,
    cache: dict | None = None,
) -> tuple[list[dict], list[dict], dict, dict]:
    """
    Extract claims for a corpus of chunks.

    STEP 1 fix — provenance: chunks are grouped by video_id BEFORE batching, so no
    batch ever straddles a video boundary. Previously batches of `batch_size` crossed
    video boundaries (e.g. last McCabe chunk + first Bilkey chunk), and every claim in
    a batch was tagged with batch[0].video_id — mislabeling the later video's claims.

    STEP 2 — caching: claims are cached keyed by their SOURCE-CHUNK content_hash. Chunks
    whose content_hash is already in `cache` are skipped entirely (no Gemini call), so
    re-running on an unchanged corpus is a true no-op. claim_ids are derived
    deterministically from the source chunk (`{chunk_id}_claim{j}`) so cached claims keep
    stable ids across runs instead of shifting with an LLM-dependent global counter.

    Returns: (claims_as_dicts_for_current_corpus, rejections, updated_cache, stats)
    """
    cache = dict(cache or {})
    valid_chunk_ids = {c.chunk_id for c in chunks}
    chid2hash = {c.chunk_id: c.content_hash for c in chunks}

    # Only chunks whose content we haven't extracted before need a Gemini call.
    to_process = [c for c in chunks if c.content_hash not in cache]

    # Group the to-process chunks by video, preserving corpus order, then batch WITHIN a video.
    by_video: "OrderedDict[str, list]" = OrderedDict()
    for c in to_process:
        by_video.setdefault(c.video_id, []).append(c)

    all_rejections = []
    gemini_calls = 0
    new_by_hash: "defaultdict[str, list[Claim]]" = defaultdict(list)

    for video_id, vchunks in by_video.items():
        for i in range(0, len(vchunks), batch_size):
            batch = vchunks[i:i + batch_size]
            prompt = build_claim_prompt(batch)
            raw_response = call_gemini(prompt)
            gemini_calls += 1
            # video_id is now guaranteed correct for every chunk in the batch.
            claims, rejections = parse_and_validate_claims(raw_response, valid_chunk_ids, video_id, start_index=0)
            all_rejections.extend(rejections)
            for claim in claims:
                src_chunk_id = claim.evidence[0].chunk_id
                h = chid2hash.get(src_chunk_id)
                if h is None:
                    continue  # evidence chunk not in corpus map (shouldn't happen; already validated in-set)
                new_by_hash[h].append(claim)

    # Assign deterministic, cache-stable claim_ids per source chunk, then store in the cache.
    for h, claim_list in new_by_hash.items():
        for j, claim in enumerate(claim_list):
            claim.claim_id = f"{claim.evidence[0].chunk_id}_claim{j}"
        cache[h] = [asdict(c) for c in claim_list]

    # Record chunks that produced ZERO claims too, so we don't re-call Gemini for them next run.
    for c in to_process:
        cache.setdefault(c.content_hash, [])

    # Assemble the current corpus's claims in corpus order, entirely from the cache.
    final_claims = []
    for c in chunks:
        final_claims.extend(cache.get(c.content_hash, []))

    stats = {
        "total_chunks": len(chunks),
        "cached_chunks": len(chunks) - len(to_process),
        "new_chunks": len(to_process),
        "gemini_calls": gemini_calls,
        "new_claims": sum(len(v) for v in new_by_hash.values()),
        "total_claims": len(final_claims),
    }
    return final_claims, all_rejections, cache, stats


def load_claim_cache(workspace_id: str) -> dict:
    """Load the content_hash -> [claim dicts] cache for a workspace (empty if none yet)."""
    cache_path = Path(WORKSPACES_DIR) / workspace_id / "claims_cache.json"
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_claim_cache(cache: dict, workspace_id: str) -> str:
    workspace_dir = Path(WORKSPACES_DIR) / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    out_path = workspace_dir / "claims_cache.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    return str(out_path)


def save_claims_to_workspace(claims: list, workspace_id: str) -> str:
    """Write claims.json. Accepts either Claim dataclasses or already-dict claims."""
    workspace_dir = Path(WORKSPACES_DIR) / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    out_path = workspace_dir / "claims.json"

    payload = [c if isinstance(c, dict) else asdict(c) for c in claims]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(out_path)


if __name__ == "__main__":
    # ============================================================
    # NO NETWORK CALL HERE — this proves the validator works using
    # a hand-built fake Gemini response with ONE bad claim mixed in.
    # ============================================================

    fake_valid_chunk_ids = {"EP9zPS1jNwA_c0000", "EP9zPS1jNwA_c0001"}

    fake_gemini_response = json.dumps([
        {
            "video_id": "EP9zPS1jNwA",
            "claim": "Obsidian plus Claude works well for a single person's personal knowledge base.",
            "claim_type": "opinion",
            "stance": "support",
            "evidence": [{"chunk_id": "EP9zPS1jNwA_c0000", "evidence_text": "Obsidian works from a graph structure"}],
            "topics": ["obsidian", "personal-knowledge-management"],
        },
        {
            "video_id": "EP9zPS1jNwA",
            "claim": "RAG uses embeddings to convert text into vector form.",
            "claim_type": "fact",
            "stance": "neutral",
            "evidence": [{"chunk_id": "EP9zPS1jNwA_c0001", "evidence_text": "you are embedding it... converting this into a vector form"}],
            "topics": ["rag", "embeddings"],
        },
        {
            # THIS ONE IS BAD ON PURPOSE — references a chunk_id that was never provided
            "video_id": "EP9zPS1jNwA",
            "claim": "RAG systems always use PostgreSQL as the primary database.",
            "claim_type": "fact",
            "stance": "neutral",
            "evidence": [{"chunk_id": "EP9zPS1jNwA_c9999", "evidence_text": "made up text that was never in any real chunk"}],
            "topics": ["rag", "database"],
        },
    ])

    claims, rejections = parse_and_validate_claims(fake_gemini_response, fake_valid_chunk_ids, "EP9zPS1jNwA", start_index=0)

    print(f"ACCEPTED: {len(claims)} claim(s)\n")
    for c in claims:
        print(f"  [{c.claim_type}/{c.stance}] {c.claim}")
        print(f"    evidence chunk: {c.evidence[0].chunk_id}\n")

    print(f"REJECTED: {len(rejections)} claim(s)\n")
    for r in rejections:
        print(f"  REJECTED: {r['reason']}")
        print(f"    claim was: \"{r['claim']}\"\n")
