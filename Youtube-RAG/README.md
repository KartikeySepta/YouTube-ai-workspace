video-rag-tool/
│
├── data/
│   ├── raw/                    # your scraped output.json files land here
│   └── workspaces/             # one folder per research topic
│       └── fiverr/
│           ├── videos.json         # Part 1: normalized video list
│           ├── chunks.json         # Part 2: chunker output
│           ├── claims.json         # Part 3: extracted claims
│           ├── clusters.json       # Part 11: claim clusters
│           ├── synthesis.json      # Part 12: cross-video synthesis
│           └── messages.json       # chat history
│
├── core/
│   ├── config.py                # settings: chunk size, top_k, model names
│   └── models.py                # TranscriptChunk, Claim, EvidenceRef dataclasses
│
├── ingestion/
│   ├── loader.py                 # Part 1: read + normalize output.json
│   └── chunker.py                 # Part 2: split transcript into chunks
│
├── knowledge/
│   ├── claim_extractor.py         # Part 3: pull claims from chunks
│   ├── claim_clusterer.py         # Part 11: group similar claims
│   └── synthesizer.py             # Part 12: cross-video agreement/disagreement
│
├── retrieval/
│   ├── embeddings.py              # Part 4: text → vectors
│   ├── vector_store.py            # Part 5: Qdrant wrapper, workspace-scoped
│   ├── bm25.py                    # Part 6: keyword search
│   ├── hybrid.py                  # Part 7: fuse vector + keyword (RRF)
│   ├── reranker.py                # Part 8: cross-encoder reranking
│   └── context.py                 # Part 9: build labeled source blocks
│
├── chat/
│   ├── engine.py                  # Part 10: the full query → answer flow
│   └── citations.py                # verify [Source N] against source_map
│
├── storage/
│   └── workspace_store.py         # read/write workspace JSON, Part 14 migration lives here too
│
├── evals/
│   ├── dataset.json                # Part 13: hand-written test questions
│   └── evaluate.py                  # hit rate / precision scoring script
│
├── cli.py                          # commands: ingest, chat, evaluate, migrate
├── requirements.txt
├── .env.example                    # GEMINI_API_KEY, QDRANT_URL, etc.
└── inspect.py                      # throwaway script for Part 1, keep it around