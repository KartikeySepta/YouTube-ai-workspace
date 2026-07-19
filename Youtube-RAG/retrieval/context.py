"""
STEP 9: CONTEXT ASSEMBLY

Takes the final reranked chunks (Step 8's output, ~5-10 items) and turns them into:
  1. A formatted text block with labeled sources, ready to hand to Gemini
  2. A source_map dict — "Source 1" -> real chunk_id/video/timestamp — so that after
     Gemini answers, we can VERIFY every citation it used actually exists (Step 10).

Two things this file does deliberately:
  - Drops near-duplicate chunks before assembling (chunks can overlap in content by
    design from Step 2 — no point spending token budget on the same sentence twice).
  - Marks estimated timestamps with "~" so nothing downstream mistakes a guess for
    a fact. This matches the project's core rule: never invent precision you don't have.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Rough word-to-token ratio for budget estimation. Not exact (Gemini's real tokenizer
# differs), but good enough for a soft budget — precision here isn't worth the complexity.
WORDS_TO_TOKENS_RATIO = 1.3


def format_timestamp(seconds: float) -> str:
    """123.4 -> '2:03'. Hours included only if >= 1 hour."""
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def format_timestamp_range(chunk: dict) -> str:
    start = format_timestamp(chunk["start_seconds"])
    end = format_timestamp(chunk["end_seconds"])
    prefix = "~" if chunk.get("is_estimated") else ""
    return f"{prefix}{start}–{prefix}{end}"


def _word_overlap_ratio(text_a: str, text_b: str) -> float:
    """
    Containment coefficient — intersection divided by the SMALLER chunk's word count,
    not the union. This matters because Jaccard (intersection/union) under-detects
    overlap when one chunk is much longer than the other: a big chunk absorbing a
    small chunk's entire content looks "small" relative to the big chunk's total size,
    even though the smaller chunk is almost entirely duplicated. Containment catches
    that case correctly — confirmed on real data where two chunks scored only 0.29 on
    Jaccard despite one being 62% verbatim-contained in the other.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    smaller_size = min(len(words_a), len(words_b))
    return len(intersection) / smaller_size


def deduplicate_chunks(chunks: list[dict], similarity_threshold: float = 0.6) -> list[dict]:
    """
    Drop chunks that are near-duplicates of one already kept (in rank order, so the
    higher-ranked one wins). Chunks overlapping by design (Step 2) can trigger this
    if two overlapping chunks both make it into the final ranked list.
    """
    kept = []
    for chunk in chunks:
        is_duplicate = any(
            _word_overlap_ratio(chunk["text"], kept_chunk["text"]) >= similarity_threshold
            for kept_chunk in kept
        )
        if not is_duplicate:
            kept.append(chunk)
    return kept


def assemble_context(ranked_chunks: list[dict], max_tokens: int = 1500) -> tuple[str, dict]:
    """
    Build the final context block + source map from ranked chunks.
    Respects a soft token budget — stops adding sources once the budget would be exceeded,
    rather than silently overloading the prompt.

    Returns:
      context_text: the full formatted string ready to insert into the Gemini prompt
      source_map: {"Source 1": {chunk_id, video_id, video_title, channel, timestamp}, ...}
    """
    deduped = deduplicate_chunks(ranked_chunks)

    blocks = []
    source_map = {}
    running_tokens = 0

    for i, chunk in enumerate(deduped):
        label = f"Source {i + 1}"
        timestamp = format_timestamp_range(chunk)

        block = (
            f"[{label}]\n"
            f"Video: {chunk.get('video_title', 'Unknown')} | "
            f"Channel: {chunk.get('channel', 'Unknown')} | "
            f"Timestamp: {timestamp}\n\n"
            f"{chunk['text']}"
        )

        block_tokens = len(block.split()) * WORDS_TO_TOKENS_RATIO
        if running_tokens + block_tokens > max_tokens and blocks:
            # Budget reached — stop here rather than blowing past it (unless this
            # would be the very first source, in which case include it anyway;
            # a prompt with zero sources is useless).
            break

        blocks.append(block)
        source_map[label] = {
            "chunk_id": chunk["chunk_id"],
            "video_id": chunk["video_id"],
            "video_title": chunk.get("video_title", ""),
            "channel": chunk.get("channel", ""),
            "timestamp": timestamp,
            "is_estimated": chunk.get("is_estimated", False),
        }
        running_tokens += block_tokens

    context_text = "\n\n".join(blocks)
    return context_text, source_map


if __name__ == "__main__":
    import json

    chunks = json.load(open("data/workspaces/rag_research/chunks.json"))

    # Simulate the real reranked order from your actual run: chunk2, chunk6, chunk24, chunk23, chunk8
    order = ["EP9zPS1jNwA_c0002", "EP9zPS1jNwA_c0006", "EP9zPS1jNwA_c0024",
             "EP9zPS1jNwA_c0023", "EP9zPS1jNwA_c0008"]
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    ranked = [chunks_by_id[cid] for cid in order]

    context_text, source_map = assemble_context(ranked, max_tokens=1500)

    print("=== ASSEMBLED CONTEXT (what Gemini will see) ===\n")
    print(context_text)
    print("\n\n=== SOURCE MAP (used later to verify citations) ===\n")
    for label, info in source_map.items():
        print(f"{label}: {info['chunk_id']} | {info['timestamp']} | estimated={info['is_estimated']}")