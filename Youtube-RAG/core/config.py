"""
Centralized settings. Nothing else in the project should hardcode these values —
if you want to change chunk size later, you change it here once, not in five files.
"""

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
CLAIM_CLUSTER_SIMILARITY_THRESHOLD = 0.87   # cosine similarity above this = same cluster
# NOTE: raised from 0.82 after real testing showed 0.82 over-merged distinct-but-related
# claims (e.g. 4 different RAG pipeline steps lumped as "one idea"). True duplicates from
# chunk overlap have near-1.0 similarity, so they stay merged even at this higher bar.

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
DATA_DIR = "data"
RAW_DIR = "data/raw"
WORKSPACES_DIR = "data/workspaces"