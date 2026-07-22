"""
Centralized settings. Nothing else in the project should hardcode these values —
if you want to change chunk size later, you change it here once, not in five files.
"""

import os

# LLM (used in Steps 3, 10, 12 — every file that calls Gemini)
# Pin a concrete, production-suitable model rather than the "-latest" alias: Google
# marks the alias as experimental/not production-suitable, and it can shift underneath
# you without warning. Still centralized here (never hardcode in the 3 caller files) and
# still overridable per-environment via the GEMINI_MODEL env var. If Google retires this
# specific version, change it in this ONE place.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

# Chunking (used in Step 2)
CHUNK_TARGET_WORDS = 180
CHUNK_OVERLAP_WORDS = 25
# NOTE: originally 500/50. Reduced after diagnosing a real reranker bug — chunks that
# size (~650-700 tokens) get silently truncated by the cross-encoder's 512-token limit,
# cutting off relevant content before the model ever sees it (confirmed by inspecting
# actual truncated text). 180 words ≈ 230-250 tokens, safely under the limit even with
# the query added. This also fixes chunks mixing multiple topics (flagged back in Step 2).

# Retrieval (used in later steps)
VECTOR_TOP_K = 25
BM25_TOP_K = 25
RERANK_KEEP_TOP = 8

# Claim clustering (Step 11)
# --- WITHIN-VIDEO policy (unchanged): auto-merge at/above the ceiling, adjudicate the gray zone.
WITHIN_VIDEO_MERGE_THRESHOLD = 0.87   # within-video cosine >= this = auto-merge (same cluster)
# NOTE: raised from 0.82 after real testing showed 0.82 over-merged distinct-but-related
# claims (e.g. 4 different RAG pipeline steps lumped as "one idea"). True duplicates from
# chunk overlap have near-1.0 similarity, so they stay merged even at this higher bar.
CLAIM_CLUSTER_SIMILARITY_THRESHOLD = WITHIN_VIDEO_MERGE_THRESHOLD  # back-compat alias

# Within-video gray zone: pairs in [LOW, THRESHOLD) are ambiguous — too high to confidently
# separate, too low to confidently merge. An LLM adjudicator decides these case-by-case.
CLAIM_CLUSTER_GRAY_ZONE_LOW = 0.80

# --- CROSS-VIDEO policy (Step 2): NO auto-merge at ANY score. Cosine is only a cheap
# pre-filter — any cross-video pair with cosine >= this floor goes to the adjudicator;
# below the floor we split automatically without spending a call. The floor (0.75) sits
# below last round's register-sensitive cross-video pairs (0.83-0.867) so those still get
# adjudicated rather than silently dropped.
CROSS_VIDEO_ADJUDICATION_FLOOR = 0.75

# Cross-source THEME grouping (Step 4 "distinct synthesis category"). Tighter than the
# adjudication floor and applied ONLY between different-video claims, so themes stay focused
# (e.g. "adult prevalence across regions") instead of transitively chaining every ADHD claim
# into one giant blob.
CROSS_SOURCE_THEME_FLOOR = 0.80

# Claim extraction batching. Chunks per Gemini call — bigger batch = fewer calls (kinder to
# the free-tier 15 req/min limit). Kept modest so the model doesn't drop claims from a huge
# prompt. 5 chunks (~900 words) is a good balance.
CLAIM_EXTRACTION_BATCH_SIZE = 5

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
DATA_DIR = "data"
RAW_DIR = "data/raw"
WORKSPACES_DIR = "data/workspaces"