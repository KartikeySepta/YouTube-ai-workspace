# YouTube AI Workspace

An end-to-end system for turning YouTube videos into structured, cited research.  
Two modules, one shared `.env`:

```
YouTube-ai-workspace/
├── scraper/              # TRANSCRIPTION — extract audio + metadata + transcript
│   ├── youtube.py        #   Main CLI: URL → audio → transcript → JSON
│   ├── api.py            #   FastAPI wrapper for the scraper
│   ├── requirements.txt  #   yt-dlp, faster-whisper, google-genai, fastapi, uvicorn
│   ├── test_*.py         #   Component tests (audio, GPU, metadata, pipeline)
│   └── data.json         #   Sample scraped output
│
├── rag/                  # RESEARCH ENGINE — evidence-first multi-video RAG
│   ├── cli.py            #   Main CLI (12 commands: add, batch, ingest, index,
│   │                     #     extract-claims, cluster, synthesize, report,
│   │                     #     chat, talk, status, evaluate)
│   ├── api.py            #   FastAPI endpoints (POST /add, /chat, GET /report, etc.)
│   ├── core/             #   Config, Gemini wrapper (429 retry), data models
│   ├── ingestion/        #   Parse + chunk transcripts
│   ├── retrieval/        #   Embed, vector store, BM25, hybrid fusion, reranker
│   ├── knowledge/        #   Claim extraction, clustering, synthesis, themes
│   ├── chat/             #   Grounded Q&A with citation verification
│   ├── evals/            #   Retrieval evaluation harness + dataset
│   ├── data/             #   Workspaces (generated, gitignored)
│   ├── requirements.txt  #   sentence-transformers, qdrant-client, rank-bm25, etc.
│   └── README.md         #   Detailed RAG usage docs
│
├── .env                  # GEMINI_API_KEY (shared by both modules)
├── .gitignore
└── README.md             # ← you are here
```

## Quick Start

```bash
# 1. Clone
git clone git@github.com:KartikeySepta/YouTube-ai-workspace.git
cd YouTube-ai-workspace

# 2. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r scraper/requirements.txt
pip install -r rag/requirements.txt
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Add a video (one command does everything)
cd rag
python3 cli.py add "https://www.youtube.com/watch?v=VIDEO_ID" my_research

# 4. Use it
python3 cli.py talk my_research          # Interactive cited Q&A
python3 cli.py report my_research        # Generate research brief
cat data/workspaces/my_research/report.md
```

## Add Multiple Videos at Once

```bash
# Create a URL file
cat > urls.txt << EOF
https://www.youtube.com/watch?v=abc123
https://www.youtube.com/watch?v=def456
https://www.youtube.com/watch?v=ghi789
EOF

# Batch process
cd rag
python3 cli.py batch urls.txt my_research
```

## Run as a Web API

```bash
# RAG API (for frontend)
cd rag && uvicorn api:app --reload --port 8000
# Docs: http://localhost:8000/docs

# Scraper API (standalone transcription)
cd scraper && uvicorn api:app --reload --port 8001
```

## How It Works

```
YouTube URL
     │
     ▼
┌─────────────┐     ┌─────────────────────────────────────────────┐
│   SCRAPER   │     │              RAG ENGINE                      │
│             │     │                                              │
│ yt-dlp      │     │  ingest → chunk → embed → index             │
│ + Gemini/   │────▶│  extract-claims → cluster → synthesize      │
│   Whisper   │     │  ────────────────────────────────────────    │
│             │     │  chat: retrieve → rerank → Gemini → verify  │
│ → output.json     │  report: cited Markdown brief                │
└─────────────┘     └─────────────────────────────────────────────┘
```

## What Makes This Different

- **Every claim is evidence-backed** — hallucinated chunk_ids are discarded
- **Every citation is verified** — grouped `[Source 1, Source 2]` all checked
- **Cross-video synthesis** — surfaces agreement/disagreement across creators
- **Scope-aware** — "same stat for different countries" is flagged, not collapsed
- **Cost-controlled** — claims cached by content_hash; re-runs don't re-call Gemini
- **Rate-limit resilient** — centralized 429 retry with exponential backoff
