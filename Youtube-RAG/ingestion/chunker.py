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


def chunk_video(
    video: Video,
    workspace_id: str,
    target_words: int = CHUNK_TARGET_WORDS,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[TranscriptChunk]:
    """
    Turn one Video's segments into TranscriptChunks.
    Groups sentences up to ~target_words, carries `overlap_words` worth of trailing
    sentences into the next chunk, and estimates timestamps by word position.
    """
    chunks = []
    chunk_index = 0

    for segment in video.segments:
        sentences = split_into_sentences(segment.text)
        if not sentences:
            continue

        # Effective end time for this segment — fall back to video duration if unknown
        # (this happens for the LAST segment, which has no "next segment" to bound it).
        effective_end = segment.end_seconds if segment.end_seconds is not None else video.duration_seconds
        effective_start = segment.start_seconds
        time_span = effective_end - effective_start

        # Pre-compute total word count in this segment, so we can turn
        # "word position" into "fraction of the way through the segment".
        sentence_word_counts = [len(s.split()) for s in sentences]
        total_words = sum(sentence_word_counts)

        current_group: list[str] = []
        current_word_count = 0
        group_start_word_index = 0   # word index where the CURRENT chunk begins
        words_seen_so_far = 0

        def flush_chunk(group: list[str], start_word_index: int, end_word_index: int):
            nonlocal chunk_index
            if not group:
                return
            text = " ".join(group)

            # Estimate start/end time by word position, proportional across this segment's time span
            frac_start = start_word_index / total_words if total_words else 0
            frac_end = end_word_index / total_words if total_words else 1
            est_start = effective_start + frac_start * time_span
            est_end = effective_start + frac_end * time_span

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

        for i, sentence in enumerate(sentences):
            current_group.append(sentence)
            current_word_count += sentence_word_counts[i]
            words_seen_so_far += sentence_word_counts[i]

            if current_word_count >= target_words:
                end_word_index = words_seen_so_far
                flush_chunk(current_group, group_start_word_index, end_word_index)

                # Build overlap: carry the last few sentences forward into the next chunk
                overlap_group = []
                overlap_count = 0
                for s, wc in zip(reversed(current_group), reversed([sentence_word_counts[j] for j in range(i - len(current_group) + 1, i + 1)])):
                    if overlap_count >= overlap_words:
                        break
                    overlap_group.insert(0, s)
                    overlap_count += wc

                current_group = overlap_group
                current_word_count = overlap_count
                group_start_word_index = end_word_index - overlap_count

        # Flush whatever's left at the end of the segment
        if current_group:
            flush_chunk(current_group, group_start_word_index, words_seen_so_far)

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