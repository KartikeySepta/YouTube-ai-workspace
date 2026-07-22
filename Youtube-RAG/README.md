# Video RAG Tool — evidence-first multi-video research

Turn a pile of YouTube videos on one topic into an **evidence-first research brief and a
grounded Q&A chat**, where every answer and every claim is traceable to a real transcript
timestamp — and where the model is never trusted blind:

- **Hybrid retrieval** (vector + BM25 + cross-encoder rerank), workspace-isolated.
- **Claim extraction with hallucination guards** — every claim's evidence `chunk_id` is
  validated against real chunks; invented ones are discarded.
- **Split merge policy** — within-video near-duplicates auto-merge (0.87) with a gray-zone
  LLM adjudicator [0.80,0.87); cross-video claims are **never** blindly merged — a scope
  rubric keeps "same finding, different region/population" as a *distinct synthesis category*
  instead of collapsing it into one number.
- **Grounded chat** — answers use only retrieved sources and every `[Source N]` citation is
  verified before it reaches you.
- **Cost/robustness** — one centralized Gemini wrapper with 429 retry+backoff; claim
  extraction is cached by chunk `content_hash`, so re-runs and incremental video adds only
  pay for genuinely new chunks.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env
# optional: GEMINI_MODEL=gemini-3.1-flash-lite  (default)
```

Scraping new videos additionally needs `ffmpeg` and `yt-dlp` (see the repo root `youtube.py`).

## End-to-end usage

A **workspace** is one research topic (e.g. `rag_research`). Ingest one scraped
`output.json` (a list of `{metadata, transcript}`) at a time — ingest is additive and
deduped by `content_hash`, so you build a multi-video corpus incrementally.

```bash
# 1. Parse + chunk a scraped file into the workspace (merge + dedup)
python3 cli.py ingest data/raw/output.json rag_research

# 2. Embed + index chunks into the local vector store (Qdrant on disk)
python3 cli.py index rag_research

# 3. Extract atomic, evidence-validated claims (cached by content_hash)
python3 cli.py extract-claims rag_research

# 4. Cluster claims (within-video auto-merge + gray-zone adjudication;
#    cross-video adjudicated, never blind-merged)
python3 cli.py cluster rag_research

# 5. Synthesize: per-cluster stats + cross-source themes across videos
python3 cli.py synthesize rag_research

# 6. PRODUCT OUTPUT: a cited Markdown research brief (no LLM calls, fast)
python3 cli.py report rag_research
#    -> data/workspaces/rag_research/report.md

# 7. Ask questions with grounded, citation-verified answers
python3 cli.py chat rag_research "how is adult ADHD diagnosed?"
#    or an interactive session:
python3 cli.py talk rag_research

# Retrieval quality check any time you change chunking/retrieval
python3 cli.py evaluate
```

### Adding another video to the same topic
Scrape it to its own `output.json`, then repeat steps 1–6 with the **same workspace_id**.
Already-ingested videos are a no-op; only the new one is processed. Cross-source themes then
show how the new creator agrees/disagrees/adds context versus the others.

## What you get in a workspace (`data/workspaces/<id>/`)
| file | contents |
|---|---|
| `videos.json` / `chunks.json` | normalized videos and timestamped chunks |
| `claims.json` / `claims_cache.json` | validated claims + the content_hash extraction cache |
| `clusters.json` | claim groupings (within-video merges) |
| `synthesis.json` | per-cluster derived stats + relationship |
| `cross_source_themes.json` | labeled cross-video connections (agreement / different_context / …) |
| `report.md` | the human-readable, cited research brief |

## Config (`core/config.py`)
Everything tunable lives here: chunk size, retrieval top-k, models, and the merge policy
(`WITHIN_VIDEO_MERGE_THRESHOLD`, `CLAIM_CLUSTER_GRAY_ZONE_LOW`,
`CROSS_VIDEO_ADJUDICATION_FLOOR`, `CROSS_SOURCE_THEME_FLOOR`).

## Offline self-tests (no network)
```bash
python3 knowledge/claim_extractor.py            # evidence-validation demo
python3 chat/engine.py --test-citation-verifier # citation verifier
python3 knowledge/synthesizer.py --test-relationship-parser
python3 evals/evaluate.py --test-scoring
python3 cli.py --test-dispatch
```
