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
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.models import TranscriptChunk, Claim, EvidenceRef
from core.config import WORKSPACES_DIR

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
    """
    Real Gemini call. Requires GEMINI_API_KEY in your environment (.env file).
    This function is NOT tested in this sandbox — no network access to Google's API here.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)
    return response.text


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


def extract_claims_for_chunks(chunks: list[TranscriptChunk], batch_size: int = 3) -> tuple[list[Claim], list[dict]]:
    """Batch chunks, call Gemini per batch, validate every response."""
    all_claims = []
    all_rejections = []
    valid_chunk_ids = {c.chunk_id for c in chunks}
    claim_counter = 0   # GLOBAL counter, carried across every batch

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        prompt = build_claim_prompt(batch)
        raw_response = call_gemini(prompt)
        video_id = batch[0].video_id
        claims, rejections = parse_and_validate_claims(raw_response, valid_chunk_ids, video_id, start_index=claim_counter)
        claim_counter += len(claims)
        all_claims.extend(claims)
        all_rejections.extend(rejections)

    return all_claims, all_rejections


def save_claims_to_workspace(claims: list[Claim], workspace_id: str) -> str:
    workspace_dir = Path(WORKSPACES_DIR) / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    out_path = workspace_dir / "claims.json"

    payload = [asdict(c) for c in claims]
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