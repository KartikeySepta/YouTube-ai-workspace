"""
Runner for Step 3 (claim extraction).
Usage: python3 run_step3.py rag_research
"""

import json
import sys
from core.models import TranscriptChunk
from knowledge.claim_extractor import extract_claims_for_chunks, save_claims_to_workspace

workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"

chunks_path = f"data/workspaces/{workspace_id}/chunks.json"
chunks_raw = json.load(open(chunks_path))
chunks = [TranscriptChunk(**c) for c in chunks_raw]

print(f"Loaded {len(chunks)} chunks from {chunks_path}")
print("Calling Gemini to extract claims (this may take a moment)...\n")

claims, rejections = extract_claims_for_chunks(chunks)

print(f"{len(claims)} accepted, {len(rejections)} rejected\n")

if rejections:
    print("--- REJECTED CLAIMS (and why) ---")
    for r in rejections:
        print(f"  {r.get('reason')}")
        if 'claim' in r:
            print(f"    claim was: \"{r['claim']}\"")
    print()

saved_path = save_claims_to_workspace(claims, workspace_id)
print(f"Saved to: {saved_path}")