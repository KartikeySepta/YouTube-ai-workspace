from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TranscriptSegment:
    """One spoken block, extracted from the raw '[MM:SS] Speaker N: text' transcript string."""
    start_seconds: float
    end_seconds: Optional[float]   # None until we know the next segment's start (set during parsing)
    speaker: Optional[str]
    text: str


@dataclass
class Chapter:
    """A creator-provided chapter marker, pulled from the video description — a REAL timestamp anchor."""
    start_seconds: float
    title: str


@dataclass
class TranscriptChunk:
    """
    One chunk of transcript text, ready for embedding/search.
    Timestamps come from real chapter anchors where possible; is_estimated tells you
    honestly whether start_seconds/end_seconds are exact or interpolated.
    """
    chunk_id: str
    workspace_id: str
    video_id: str
    video_title: str
    channel: str
    start_seconds: float
    end_seconds: float
    is_estimated: bool          # False only if these times came directly from real markers
    nearest_chapter_title: Optional[str]
    text: str
    chunk_index: int
    content_hash: str


@dataclass
class EvidenceRef:
    """Points a claim back at the exact chunk it came from. chunk_id MUST be verified to exist."""
    chunk_id: str
    evidence_text: str   # the specific snippet from the chunk that supports this claim


@dataclass
class Claim:
    """
    One atomic, checkable idea pulled from a single chunk.
    NOTE: no confidence field on purpose — LLM self-rated confidence isn't reliable.
    Any "confidence" signal gets derived later from counted data (how many videos agree, etc),
    not asked for here.
    """
    claim_id: str
    video_id: str
    claim: str
    claim_type: str     # "recommendation" | "opinion" | "fact" | "prediction" | "warning"
    stance: str         # "support" | "oppose" | "neutral" | "mixed"
    evidence: list       # list[EvidenceRef]
    topics: list = field(default_factory=list)
    cluster_id: Optional[str] = None   # set by Step 11 (claim_clusterer.py)


@dataclass
class ClusterSynthesis:
    """
    One cluster's synthesis result — always backed by real claim_ids, never invented.
    relationship is one of: "single_source" | "agreement" | "partial_agreement" |
    "contradiction" | "different_context" | "independent"
    """
    cluster_id: str
    member_claim_ids: list
    member_video_ids: list
    unique_channels: int
    supporting_count: int    # derived from claim.stance == "support", NOT an LLM confidence score
    contradicting_count: int  # derived from claim.stance == "oppose"
    neutral_count: int
    relationship: str
    synthesis_note: Optional[str] = None   # only set for multi-video clusters, via Gemini + validation


@dataclass
class Video:
    """Normalized video record — same shape regardless of scraper quirks."""
    video_id: str
    title: str
    channel: str
    channel_url: str
    upload_date: str            # raw format "20260708" (YYYYMMDD) from this scraper
    duration_seconds: int
    view_count: int
    like_count: int
    tags: list = field(default_factory=list)
    description: str = ""
    chapters: list = field(default_factory=list)   # list[Chapter] — real timestamp anchors, if any
    segments: list = field(default_factory=list)   # list[TranscriptSegment]