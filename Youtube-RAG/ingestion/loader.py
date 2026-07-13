"""
STEP 1: PARSE

Your scraper does NOT give you pre-split transcript segments with start/end times.
It gives you ONE long string per video, with timestamps embedded inline like:

    "[00:00] Speaker 1: text here... [01:07] Speaker 1: more text..."

So "parsing" here has two jobs:
  1. Pull out the clean metadata fields you actually need (ignore the rest for now)
  2. Split the transcript string into real segments using the [MM:SS] markers,
     and figure out each segment's end time from the NEXT segment's start time.

This is the step everything downstream depends on. Get this right once.
"""

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.models import Video, TranscriptSegment, Chapter
from core.config import WORKSPACES_DIR

# Matches "[12:43]" or "[1:12:43]" (hours optional), followed by an optional "Speaker N:" label
TIMESTAMP_PATTERN = re.compile(
    r"\[(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\]\s*(?:(Speaker \d+):)?\s*"
)

# Matches chapter lines in a description, e.g. "12:51 Claude writes my blogs (with images)"
CHAPTER_PATTERN = re.compile(r"^(?:(\d{1,2}):)?(\d{1,2}):(\d{2})\s+(.+)$", re.MULTILINE)


def _timestamp_to_seconds(hours: str, minutes: str, seconds: str) -> float:
    h = int(hours) if hours else 0
    m = int(minutes)
    s = int(seconds)
    return h * 3600 + m * 60 + s


def extract_chapters(description: str) -> list[Chapter]:
    """
    Pull real, creator-provided chapter timestamps out of the video description, e.g.:
        "1:07 How RAG actually works"
    These are REAL timestamps (the creator set them), unlike anything we'd guess ourselves.
    Returns an empty list if the description has no chapter-style lines — that's fine,
    not every video has them.
    """
    chapters = []
    for m in CHAPTER_PATTERN.finditer(description):
        hours, minutes, seconds, title = m.group(1), m.group(2), m.group(3), m.group(4)
        start_seconds = _timestamp_to_seconds(hours, minutes, seconds)
        chapters.append(Chapter(start_seconds=start_seconds, title=title.strip()))
    return chapters


def parse_transcript(raw_transcript: str) -> list[TranscriptSegment]:
    """Split the raw transcript string into segments at each [MM:SS] marker."""
    matches = list(TIMESTAMP_PATTERN.finditer(raw_transcript))

    if not matches:
        # Fallback: no timestamp markers found at all — treat the whole thing as one segment.
        # This shouldn't happen with this scraper's format, but don't crash silently on bad data.
        return [TranscriptSegment(start_seconds=0.0, end_seconds=None, speaker=None, text=raw_transcript.strip())]

    segments = []
    for i, m in enumerate(matches):
        start_seconds = _timestamp_to_seconds(m.group(1), m.group(2), m.group(3))
        speaker = m.group(4)

        text_start = m.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_transcript)
        text = raw_transcript[text_start:text_end].strip()

        if not text:
            continue  # skip empty segments (e.g. two markers back to back)

        segments.append(TranscriptSegment(
            start_seconds=start_seconds,
            end_seconds=None,  # filled in below
            speaker=speaker,
            text=text,
        ))

    # Now that we have all segments in order, set each one's end_seconds
    # to the next segment's start_seconds.
    for i in range(len(segments) - 1):
        segments[i].end_seconds = segments[i + 1].start_seconds

    return segments


def load_videos(path: str) -> list[Video]:
    """Read output.json (a list of {metadata, transcript} objects) into normalized Video objects."""
    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected output.json to be a list of videos, got {type(raw_data)}")

    videos = []
    for item in raw_data:
        meta = item.get("metadata", {})
        raw_transcript = item.get("transcript", "")

        if not raw_transcript:
            print(f"WARNING: video {meta.get('video_id')} has no transcript, skipping")
            continue

        segments = parse_transcript(raw_transcript)
        chapters = extract_chapters(meta.get("description", ""))

        if len(segments) == 1 and not chapters:
            print(f"WARNING: video {meta.get('video_id')} has only 1 transcript segment "
                  f"and no chapter markers — per-chunk timestamps will need to be estimated later.")
        elif len(segments) == 1 and chapters:
            print(f"NOTE: video {meta.get('video_id')} has only 1 raw transcript segment, "
                  f"but {len(chapters)} real chapter timestamps were found in the description — "
                  f"use these as anchors when chunking.")

        videos.append(Video(
            video_id=meta.get("video_id", ""),
            title=meta.get("title", ""),
            channel=meta.get("channel", ""),
            channel_url=meta.get("channel_url", ""),
            upload_date=meta.get("upload_date", ""),
            duration_seconds=meta.get("duration_seconds", 0),
            view_count=meta.get("view_count", 0),
            like_count=meta.get("like_count", 0),
            tags=meta.get("tags", []),
            description=meta.get("description", ""),
            chapters=chapters,
            segments=segments,
        ))

    return videos


def save_videos_to_workspace(videos: list[Video], workspace_id: str) -> str:
    """
    Write normalized videos out to data/workspaces/<workspace_id>/videos.json,
    matching the project structure. This is what makes Step 1's output persistent
    instead of just living in memory for one script run.
    """
    workspace_dir = Path(WORKSPACES_DIR) / workspace_id
    workspace_dir.mkdir(parents=True, exist_ok=True)
    out_path = workspace_dir / "videos.json"

    # asdict() turns our dataclasses into plain dicts so json.dump can write them
    payload = [asdict(v) for v in videos]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(out_path)


if __name__ == "__main__":
    # Quick manual check — run this file directly to sanity-check parsing on real data.
    # Usage: python3 ingestion/loader.py data/raw/output.json <workspace_id>
    import sys as _sys
    path = _sys.argv[1] if len(_sys.argv) > 1 else "data/raw/output.json"
    workspace_id = _sys.argv[2] if len(_sys.argv) > 2 else "default_workspace"

    videos = load_videos(path)

    print(f"Loaded {len(videos)} video(s)\n")
    for v in videos:
        print(f"video_id: {v.video_id}")
        print(f"title: {v.title}")
        print(f"channel: {v.channel}")
        print(f"upload_date: {v.upload_date}  duration_seconds: {v.duration_seconds}")
        print(f"segments found: {len(v.segments)}\n")
        for seg in v.segments:
            end = f"{seg.end_seconds:.0f}s" if seg.end_seconds is not None else "END"
            print(f"  [{seg.start_seconds:.0f}s -> {end}] ({seg.speaker}) {seg.text[:80]}...")

    saved_path = save_videos_to_workspace(videos, workspace_id)
    print(f"\nSaved normalized videos to: {saved_path}")