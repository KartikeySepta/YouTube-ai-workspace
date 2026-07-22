"""
STEP 2: CHUNK

Your real data has a problem Step 1 uncovered: transcript segments are chapter-length
blocks (one segment = the ENTIRE 19-minute video), not sentence-length. So this step
has to do two jobs at once:

  1. Split the long segment text into ~300-700 word pieces, at sentence boundaries.
  2. Since the raw transcript doesn't give per-sentence timestamps, ESTIMATE each
     chunk's time by its position in the segment (word N out of 3759 total words),
     scaled across the segment's known start/end time. Then snap to the nearest real
     chapter marker as a trustworthy label.

Every chunk is honestly marked is_estimated=True, because we are guessing based on
word position, not reading a real per-sentence timestamp off the transcript.
"""

import hashlib
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.models import Video, TranscriptChunk, Chapter
from core.config import CHUNK_TARGET_WORDS, CHUNK_OVERLAP_WORDS, WORKSPACES_DIR

# Split on sentence-ending punctuation followed by whitespace — keeps the punctuation
# attached to the sentence it belongs to, so we never cut mid-sentence.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _nearest_chapter(chapters: list[Chapter], seconds: float) -> Chapter | None:
    """Find the latest chapter that starts at or before `seconds`."""
    best = None
    for ch in sorted(chapters, key=lambda c: c.start_seconds):
        if ch.start_seconds <= seconds:
            best = ch
        else:
            break
    return best


def _estimate_sentence_times(segment, sentence_word_counts, video_duration):
    """
    Estimate a (start, end) time for each sentence in ONE segment by word position,
    proportionally across that segment's known time span. For a one-sentence segment
    this just returns the segment's own [start, end].
    """
    effective_end = segment.end_seconds if segment.end_seconds is not None else video_duration
    effective_start = segment.start_seconds
    time_span = max(effective_end - effective_start, 0.0)
    total_words = sum(sentence_word_counts)

    times = []
    words_before = 0
    for wc in sentence_word_counts:
        frac_start = words_before / total_words if total_words else 0.0
        frac_end = (words_before + wc) / total_words if total_words else 1.0
        times.append((
            effective_start + frac_start * time_span,
            effective_start + frac_end * time_span,
        ))
        words_before += wc
    return times


def chunk_video(
    video: Video,
    workspace_id: str,
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[TranscriptChunk]:
    """
    Turn one Video's segments into TranscriptChunks.

    Sentences are collected across ALL segments into one flat stream, THEN packed into
    ~target_words chunks. This is deliberately segment-agnostic, because real transcripts
    come in two broken extremes:
      - one giant segment (a single 0:00 marker for the whole video), and
      - one segment PER sentence (a timestamp on every line).
    An earlier version reset chunking at every segment boundary, which produced one
    enormous chunk in the first case and hundreds of one-sentence chunks in the second.
    Packing a flat sentence stream produces consistent ~target_words chunks either way.

    Each sentence keeps its own estimated (start, end), so a chunk that spans multiple
    segments still gets an honest start (its first sentence) and end (its last sentence).
    """
    # 1) Flatten every segment into timed sentence units: {"text", "words", "start", "end"}.
    units = []
    for segment in video.segments:
        sentences = split_into_sentences(segment.text)
        if not sentences:
            continue
        word_counts = [len(s.split()) for s in sentences]
        times = _estimate_sentence_times(segment, word_counts, video.duration_seconds)
        for sentence, wc, (start, end) in zip(sentences, word_counts, times):
            units.append({"text": sentence, "words": wc, "start": start, "end": end})

    # 2) Greedily pack sentence units into chunks up to target_words, with sentence overlap.
    chunks = []
    chunk_index = 0

    def flush(group):
        nonlocal chunk_index
        if not group:
            return
        text = " ".join(u["text"] for u in group)
        est_start = group[0]["start"]
        est_end = max(u["end"] for u in group)
        chapter = _nearest_chapter(video.chapters, est_start)

        chunks.append(TranscriptChunk(
            chunk_id=f"{video.video_id}_c{chunk_index:04d}",
            workspace_id=workspace_id,
            video_id=video.video_id,
            video_title=video.title,
            channel=video.channel,
            start_seconds=round(est_start, 1),
            end_seconds=round(est_end, 1),
            is_estimated=True,   # always True here — we never got real per-sentence timestamps
            nearest_chapter_title=chapter.title if chapter else None,
            text=text,
            chunk_index=chunk_index,
            content_hash=_content_hash(text),
        ))
        chunk_index += 1

    current = []
    current_words = 0
    added_since_flush = 0   # guards against emitting a trailing chunk that is ONLY overlap

    for unit in units:
        current.append(unit)
        current_words += unit["words"]
        added_since_flush += 1

        if current_words >= target_words:
            flush(current)

            # Carry trailing sentences (up to overlap_words) into the next chunk for context.
            # Never carry the ENTIRE just-flushed group — that would re-emit the same chunk and
            # stall progress when a single sentence already exceeds target_words.
            overlap = []
            overlap_count = 0
            for u in reversed(current[:-1]):
                if overlap_count >= overlap_words:
                    break
                overlap.insert(0, u)
                overlap_count += u["words"]

            current = overlap
            current_words = overlap_count
            added_since_flush = 0

    # Flush the genuine leftover tail — but not if `current` is purely carried-over overlap.
    if added_since_flush > 0:
        flush(current)

    return chunks


def chunk_all_videos(videos: list[Video], workspace_id: str) -> list[TranscriptChunk]:
    all_chunks = []
    for video in videos:
        all_chunks.extend(chunk_video(video, workspace_id))
    return all_chunks


def save_chunks_to_workspace(chunks: list[TranscriptChunk], workspace_id: str) -> str:
    workspace_dir = Path(WORKSPACES_DIR) / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    out_path = workspace_dir / "chunks.json"

    payload = [asdict(c) for c in chunks]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(out_path)


if __name__ == "__main__":
    # Usage: python3 ingestion/chunker.py <workspace_id>
    import sys as _sys
    workspace_id = _sys.argv[1] if len(_sys.argv) > 1 else "default_workspace"

    videos_path = Path(WORKSPACES_DIR) / workspace_id / "videos.json"
    with open(videos_path) as f:
        raw_videos = json.load(f)

    # Rebuild Video objects (including nested Chapter/TranscriptSegment) from the saved JSON
    from core.models import TranscriptSegment
    videos = []
    for v in raw_videos:
        v["chapters"] = [Chapter(**c) for c in v.get("chapters", [])]
        v["segments"] = [TranscriptSegment(**s) for s in v.get("segments", [])]
        videos.append(Video(**v))

    chunks = chunk_all_videos(videos, workspace_id)

    print(f"Produced {len(chunks)} chunks from {len(videos)} video(s)\n")
    for c in chunks:
        word_count = len(c.text.split())
        print(f"[{c.chunk_index}] {c.start_seconds:.0f}s -> {c.end_seconds:.0f}s "
              f"(~{word_count}w, near chapter: '{c.nearest_chapter_title}')")
        print(f"     {c.text[:100]}...")
        print()

    saved_path = save_chunks_to_workspace(chunks, workspace_id)
    print(f"Saved chunks to: {saved_path}")