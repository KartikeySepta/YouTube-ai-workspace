"""
DIAGNOSTIC: is the reranker truncating chunks before it ever sees the relevant part?

Run this on your machine (needs the same model already downloaded from reranker.py):
    python3 diagnose_reranker_truncation.py rag_research
"""

import json
import sys
from pathlib import Path
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core.config import WORKSPACES_DIR, RERANKER_MODEL

workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"

chunks = json.load(open(Path(WORKSPACES_DIR) / workspace_id / "chunks.json"))

model = CrossEncoder(RERANKER_MODEL)
tokenizer = model.tokenizer
max_len = model.max_length or getattr(model.model.config, "max_position_embeddings", None)

print(f"Reranker's max sequence length: {max_len}\n")

query = "how does RAG work"
query_tokens = len(tokenizer.encode(query))
print(f"Query token count: {query_tokens}\n")

for c in chunks:
    token_count = len(tokenizer.encode(c["text"]))
    combined = token_count + query_tokens
    truncated = "TRUNCATED" if max_len and combined > max_len else "ok"
    print(f"chunk {c['chunk_index']}: {token_count} tokens (+query = {combined}) -> {truncated}")

    if truncated == "TRUNCATED":
        # show what actually survives after truncation vs what gets cut off
        keep_tokens = max_len - query_tokens
        kept_text = tokenizer.decode(tokenizer.encode(c["text"])[:keep_tokens])
        cut_text = tokenizer.decode(tokenizer.encode(c["text"])[keep_tokens:])
        print(f"  KEPT (what model actually sees): {kept_text[:150]}...")
        print(f"  CUT OFF (never seen by model):   {cut_text[:150]}...")