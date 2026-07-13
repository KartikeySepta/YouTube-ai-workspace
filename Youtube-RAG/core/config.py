"""
Centralized settings. Nothing else in the project should hardcode these values —
if you want to change chunk size later, you change it here once, not in five files.
"""

# Chunking (used in Step 2)
CHUNK_TARGET_WORDS = 500
CHUNK_OVERLAP_WORDS = 50

# Retrieval (used in later steps)
VECTOR_TOP_K = 25
BM25_TOP_K = 25
RERANK_KEEP_TOP = 8

# Models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
DATA_DIR = "data"
RAW_DIR = "data/raw"
WORKSPACES_DIR = "data/workspaces"