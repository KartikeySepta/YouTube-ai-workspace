# Video RAG Pipeline — Complete Build Log & Learning Guide

**Author:** Kartikey Septa  
**Built:** July 21–22, 2026  
**For:** Abhishek (and anyone who wants to understand how this was built from scratch)

---

## What Is This?

An **evidence-first multi-video research tool.** You paste YouTube video URLs about a topic (e.g. "how to get your first Fiverr client"), the system scrapes all of them, extracts knowledge, and gives you:

1. **A cited research brief** — what do the creators collectively agree on, where do they disagree, what's each one's emphasis
2. **A grounded Q&A chat** — ask questions, get answers backed by real transcript timestamps (the AI cannot hallucinate — every citation is verified)

The key insight: **the AI is never trusted blind.** Every claim traces back to a real timestamp. Every citation is verified before showing it to you. Hallucinated evidence is caught and discarded.

---

## Why This Exists (The Problem It Solves)

When you want to *do* something (get a Fiverr job, learn a tool, grow a channel):
- Good knowledge is trapped in YouTube videos — slow, unsearchable, padded
- Creators **contradict each other** — one says "niche down," another says "offer everything"
- Most advice is generic — not "what should *I* specifically do Monday morning?"

**The question nobody can answer by watching videos manually:**
> "What do 15 experienced people *actually agree on*, where do they disagree and why, and what does that mean for me?"

That's what this tool answers, with citations.

---

## The Architecture (How It Works)

```
YouTube URL(s)
     │
     ▼
┌─────────────────────┐
│  SCRAPER (youtube.py)│ ← yt-dlp + Gemini/Whisper transcription
│  → output.json       │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│  INGESTION                                            │
│  loader.py: parse transcript → segments               │
│  chunker.py: merge into ~180-word chunks              │
│  (dedup via content_hash — re-ingest = no-op)         │
└──────────┬───────────────────────────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌──────────┐ ┌──────────────────────────────────────────┐
│ RETRIEVAL│ │ KNOWLEDGE                                 │
│          │ │                                           │
│ embed    │ │ claim_extractor: Gemini extracts claims   │
│ (BGE)    │ │   → evidence validated (chunk_id check)  │
│    ↓     │ │   → cached by content_hash               │
│ Qdrant   │ │          ↓                               │
│ (vector) │ │ claim_clusterer: within-video auto-merge  │
│    ↓     │ │   + gray-zone adjudication               │
│ BM25     │ │   cross-video: NEVER auto-merge          │
│ (keyword)│ │   → adjudicate with scope rubric         │
│    ↓     │ │          ↓                               │
│ RRF      │ │ synthesizer: per-cluster stats           │
│ (fusion) │ │   + cross-source themes (labeled pairs)  │
│    ↓     │ │          ↓                               │
│ Reranker │ │ report.md (cited research brief)         │
│ (cross-  │ └──────────────────────────────────────────┘
│  encoder)│
│    ↓     │
│ Context  │
│ Builder  │
│    ↓     │
│ Gemini   │
│ (ground) │
│    ↓     │
│ Citation │
│ Verify   │
│    ↓     │
│ ANSWER   │
└──────────┘
```

---

## Everything We Built, Step By Step

### Step 1 — Parse (`ingestion/loader.py`)
**What:** Reads the scraper's `output.json`, splits the raw `[MM:SS] text` into timed segments.  
**What we learned:** Real transcripts don't always have per-sentence timestamps. Some have one marker for the whole video, others have one per line. The parser handles both by also extracting chapter markers from the description as anchor points.  
**Key concept:** Data normalization — messy input → clean internal format.

### Step 2 — Chunk (`ingestion/chunker.py`)
**What:** Groups sentences into ~180-word chunks at sentence boundaries, with overlap for context.  
**Bug we found & fixed:** Original version treated each segment as isolated, producing 198 one-sentence "chunks" on one video and one 3,700-word mega-chunk on another. Fixed by flattening all sentences into one stream then packing.  
**Another bug:** The cross-encoder reranker silently truncates at 512 tokens. 500-word chunks (~700 tokens) had their relevant content cut off *before the model ever saw it*. Shrunk to 180 words (~250 tokens).  
**Key concepts:** Chunking strategy, overlap for context, token budgets.

### Step 3 — Claim Extraction (`knowledge/claim_extractor.py`)
**What:** Sends chunks to Gemini, asks for atomic claims (one checkable idea each), validates every evidence `chunk_id` actually exists.  
**The core rule:** If Gemini references a `chunk_id` that doesn't exist in our data — the claim is THROWN AWAY, never trusted.  
**Bug found:** Claim IDs restarted at 0 per batch (duplicates). Fixed with a global counter.  
**Bug found:** Batches crossing video boundaries mislabeled claims (used batch[0]'s video_id for everything). Fixed by grouping chunks per video before batching.  
**Feature added:** Claims are cached by source-chunk `content_hash` — re-running doesn't re-call Gemini.  
**Key concepts:** LLM output validation, anti-hallucination guardrails, caching for cost control.

### Steps 4-5 — Embeddings + Vector Store (`retrieval/embeddings.py`, `vector_store.py`)
**What:** Turn text into vectors (BGE model), store in Qdrant with workspace isolation.  
**Critical detail:** BGE needs a special query prefix for queries but NOT passages (asymmetric retrieval). Missing this silently produces bad results with no error.  
**Bug found:** Python's `hash()` is randomized per process — re-indexing duplicated points instead of overwriting. Fixed with stable `sha1`-based IDs.  
**Key concepts:** Dense retrieval, workspace isolation, idempotent indexing.

### Step 6 — BM25 Keyword Search (`retrieval/bm25.py`)
**What:** Old-school term-frequency scoring. Good at exact terms where vector search is uncertain.  
**Key concept:** Lexical vs semantic retrieval — they fail in *different* ways, which is why combining them helps.

### Step 7 — Hybrid Fusion (`retrieval/hybrid.py`)
**What:** Reciprocal Rank Fusion combines vector + BM25 rankings.  
**Key concept:** RRF doesn't care about scores, only ranks. An item does well if it ranks well in EITHER list.

### Step 8 — Cross-Encoder Reranking (`retrieval/reranker.py`)
**What:** Reads query + candidate text *together* and scores relevance directly. Much better than pre-computed embeddings, but slow (only run on top ~25 candidates).  
**Key concept:** Bi-encoder (fast, approximate) → cross-encoder (slow, precise). This is the standard production pattern.

### Step 9 — Context Assembly (`retrieval/context.py`)
**What:** Builds labeled `[Source N]` blocks + a source_map for citation verification.  
**Bug found:** Near-duplicate detection used Jaccard similarity, which missed duplicates of different lengths. Fixed with containment coefficient.  
**Key concepts:** Token budgets, deduplication, source attribution.

### Step 10 — Grounded Chat (`chat/engine.py`)
**What:** The full loop: retrieve → rerank → assemble → prompt Gemini (only use these sources!) → verify every citation against the real source_map.  
**Bug found:** The citation regex only matched `[Source N]` but Gemini outputs `[Source 1, Source 2]` (grouped). Those slipped through UNVERIFIED. Fixed with a two-stage regex.  
**Key concepts:** Grounded generation, citation verification, prompt engineering for factual constraints.

### Step 11 — Claim Clustering (`knowledge/claim_clusterer.py`)
**What:** Groups claims expressing the same idea. Split policy:
- **Within-video:** Auto-merge at cosine ≥ 0.87 + LLM adjudication for gray zone [0.80, 0.87)
- **Cross-video:** NEVER auto-merge. Adjudicate everything ≥ 0.75 with a strict scope rubric.  
**Key concepts:** Cosine similarity thresholds, greedy clustering, LLM-as-judge, scope-aware rubric.

### Step 12 — Synthesis + Cross-Source Themes (`knowledge/synthesizer.py`)
**What:** Two layers:
1. Per-cluster stats (supporting/contradicting counts — derived from data, NOT LLM confidence)
2. Cross-source themes: top-10 strongest cross-video claim pairs, each labeled (agreement / different_context / contradiction)  
**Key concepts:** Cross-video agreement detection, the "distinct synthesis category" (same finding for different populations ≠ same claim, but still surfaceable as a connection).

### Step 13 — Evaluation (`evals/evaluate.py`)
**What:** 13 hand-written questions with expected chunk_ids. Measures hit@1/3/5 and MRR.  
**Key finding:** Chunk-derived questions (written by looking at chunk text) inflate scores to ~0.90. Description-sourced questions (leakage-free) give honest numbers of ~0.45 MRR.  
**Key concepts:** Retrieval evaluation, leakage risk, honest baselines.

### Infrastructure
- **`core/llm.py`** — Centralized Gemini wrapper with 429 retry + exponential backoff (free tier = 15 req/min)
- **`core/config.py`** — All thresholds, model names, paths in ONE place
- **`cli.py`** — 9 commands: ingest, index, extract-claims, cluster, synthesize, report, chat, talk, evaluate

---

## Bugs Found & Fixed (The Portfolio Story)

| # | Bug | How Found | Fix |
|---|-----|-----------|-----|
| 1 | Chunker produced 198 one-sentence chunks OR one mega-chunk (segment-boundary reset) | First real ingest | Flatten sentences across segments then pack |
| 2 | Reranker silently truncated chunks at 512 tokens (cut relevant content) | Token-count diagnostic script | Reduced chunk target 500→180 words |
| 3 | `claim_extractor.py` was overwritten with `engine.py` (byte-identical files) | `diff -q` during analysis | Restored from memory, unified SDK |
| 4 | Gemini model `gemini-2.5-flash` retired (404) | Live pipeline crash | Centralized in config, pinned to `gemini-3.1-flash-lite` |
| 5 | Citation regex missed grouped `[Source 1, Source 2]` — unverified citations | Inspecting real chat output | Two-stage regex (bracket + number extraction) |
| 6 | Claim IDs restarted at 0 per batch (duplicates) | Claim count mismatch | Global counter across batches |
| 7 | Cross-video batching mislabeled claims (batch[0].video_id) | Cross-ref claim.video_id vs evidence chunk | Group-by-video before batching |
| 8 | Python `hash()` randomized per process → re-index duplicates points | Point count 37 ≠ expected 20 | Stable `sha1`-based point IDs |
| 9 | Near-duplicate detection (Jaccard) missed size-asymmetric duplicates | Manual context inspection | Containment coefficient instead |
| 10 | 429 crashes pipeline (no retry) | `synthesize` hit free-tier limit | Centralized retry with exponential backoff |
| 11 | Above-threshold false merges (Canada vs US prevalence at cos 0.899) | Direct adjudicator test | Split within/cross policy; cross never auto-merges |
| 12 | Transitive-closure themes blobbed all 74 claims into one | First synthesize run | Bounded top-K cross-video pairs only |

---

## How to Run (From Scratch)

```bash
# Clone
git clone git@github.com:KartikeySepta/YouTube-ai-workspace.git
cd YouTube-ai-workspace/Youtube-RAG

# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "GEMINI_API_KEY=your_key" > .env

# Add a video (needs ffmpeg installed)
cd ..
python3 youtube.py "https://www.youtube.com/watch?v=VIDEO_ID" --engine cloud --output /tmp/vid.json
cd Youtube-RAG

# Full pipeline
python3 cli.py ingest /tmp/vid.json my_research
python3 cli.py index my_research
python3 cli.py extract-claims my_research
python3 cli.py cluster my_research
python3 cli.py synthesize my_research
python3 cli.py report my_research

# Use it
python3 cli.py talk my_research
cat data/workspaces/my_research/report.md
```

---

## Tech Stack

| Layer | Tool | Why |
|-------|------|-----|
| Scraping | yt-dlp + FFmpeg | Best YouTube downloader; reliable |
| Transcription | Gemini (cloud) / Faster Whisper (local) | Cloud is easier on Mac; local for GPU machines |
| Embeddings | `BAAI/bge-small-en-v1.5` | Strong, small (384d), asymmetric query support |
| Vector Store | Qdrant (local mode) | No server needed; just a file on disk |
| Keyword Search | rank_bm25 | Simple, no dependencies, fast |
| Reranking | `ms-marco-MiniLM-L-6-v2` | Proven cross-encoder for passage re-ranking |
| LLM | Google Gemini 3.1 Flash Lite | Free tier, fast, good at structured output |
| Framework | Pure Python + FastAPI (optional) | No heavy framework; everything transparent |

---

## Key Concepts to Learn (In Order)

If Abhishek wants to understand this deeply, study in this order:

### 1. RAG Basics
- What is Retrieval-Augmented Generation
- Why "just paste everything into the prompt" doesn't scale
- Chunking strategies and why they matter

### 2. Embeddings & Vector Search
- What are embeddings (text → numbers)
- Cosine similarity (how you measure "closeness")
- Why query vs passage embeddings differ (asymmetric models)
- Vector databases (Qdrant, Pinecone, Chroma)

### 3. Hybrid Retrieval
- Why vector search alone isn't enough (misses exact terms)
- BM25 for keyword matching
- RRF (Reciprocal Rank Fusion) to combine them
- Cross-encoder reranking (the two-stage pattern)

### 4. Grounding & Citation Verification
- Why you can't trust LLM output
- The grounded prompt pattern ("use ONLY these sources")
- Post-hoc citation verification (regex → source_map check)
- The hallucination rejection pattern (claim_extractor)

### 5. LLM-as-Judge
- Using the LLM to adjudicate ambiguous decisions
- Why scope-aware rubrics matter (same topic ≠ same claim)
- When to trust the judge vs when to use hard rules

### 6. Evaluation
- hit@K and MRR (retrieval quality metrics)
- Leakage risk (writing tests from the data you're testing)
- Why honest baselines matter more than high scores

### 7. Production Hardening
- 429 retry with exponential backoff
- Idempotent operations (re-run = no-op)
- Content-hash caching for cost control
- Stable IDs for reproducibility

---

## What Makes This a Job-Getting Project

1. **Not a wrapper** — it's a multi-stage pipeline with real engineering decisions
2. **12 real bugs found and fixed** — each one is a story for interviews
3. **Eval-driven** — "I measured it, found it was wrong, and this is the before/after"
4. **Anti-hallucination is the core feature** — exactly what production LLM teams care about
5. **Cross-video synthesis** — nobody else does "what do multiple creators agree/disagree on"
6. **Works end-to-end** — clone, setup, run, get a real cited answer in 5 minutes

---

## Quick Reference: CLI Commands

```bash
python3 cli.py ingest <file> <workspace>     # Parse + chunk (additive, deduped)
python3 cli.py index <workspace>             # Embed + index (idempotent)
python3 cli.py extract-claims <workspace>    # Gemini → claims (cached)
python3 cli.py cluster <workspace>           # Group claims (adjudicated)
python3 cli.py synthesize <workspace>        # Cross-source themes
python3 cli.py report <workspace>            # → report.md (no LLM, instant)
python3 cli.py chat <workspace> "question"   # Single cited Q&A
python3 cli.py talk <workspace>              # Interactive chat session
python3 cli.py evaluate                      # Retrieval quality check
```

---

## What's Next (If You Want to Keep Building)

1. **`add` command** — paste a URL, it scrapes + runs the full pipeline in one shot
2. **`batch` command** — give it 20 URLs in a text file, processes all
3. **FastAPI endpoints** — so a frontend (React/Next.js) can call it
4. **Persistent chat history** — conversations saved across sessions
5. **Personalized playbook** — ask 3 questions about yourself, filter advice for your situation
6. **Frontend** — simple chat UI with the report rendered as a page

---

*Built by Kartikey. Shared with Abhishek because knowledge should compound, not stay locked in one brain.*
