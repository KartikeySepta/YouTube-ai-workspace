"""
Video RAG Tool — FastAPI endpoints.

Run:  uvicorn api:app --reload --port 8000
Docs: http://localhost:8000/docs (auto-generated Swagger UI)
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))

app = FastAPI(
    title="Video RAG API",
    description="Evidence-first multi-video research tool — grounded Q&A with citation verification",
    version="1.0.0",
)

# Allow frontend on any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REQUEST / RESPONSE MODELS ────────────────────────────────────────────────

class AddVideoRequest(BaseModel):
    url: str
    workspace_id: str
    engine: str = "cloud"


class ChatRequest(BaseModel):
    workspace_id: str
    question: str
    history: list[dict] | None = None
    mode: str = "grounded"   # "grounded" (cite-only) or "assist" (build/apply)


class ChatResponse(BaseModel):
    answer: str
    sources: dict[str, Any]
    citations_valid: bool
    cited_count: int


# ─── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Video RAG API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/workspaces")
def list_workspaces():
    """List all research workspaces."""
    from core.config import WORKSPACES_DIR
    ws_dir = Path(WORKSPACES_DIR)
    if not ws_dir.exists():
        return {"workspaces": []}
    workspaces = []
    for d in sorted(ws_dir.iterdir()):
        if d.is_dir():
            videos_path = d / "videos.json"
            n_videos = len(json.load(open(videos_path))) if videos_path.exists() else 0
            claims_path = d / "claims.json"
            n_claims = len(json.load(open(claims_path))) if claims_path.exists() else 0
            workspaces.append({
                "id": d.name,
                "videos": n_videos,
                "claims": n_claims,
                "has_report": (d / "report.md").exists(),
            })
    return {"workspaces": workspaces}


@app.get("/workspaces/{workspace_id}")
def workspace_detail(workspace_id: str):
    """Detailed workspace info: videos, claim counts, themes."""
    from core.config import WORKSPACES_DIR
    ws = Path(WORKSPACES_DIR) / workspace_id
    if not ws.exists():
        raise HTTPException(404, f"Workspace '{workspace_id}' not found")

    videos = json.load(open(ws / "videos.json")) if (ws / "videos.json").exists() else []
    claims = json.load(open(ws / "claims.json")) if (ws / "claims.json").exists() else []
    themes = json.load(open(ws / "cross_source_themes.json")) if (ws / "cross_source_themes.json").exists() else []

    return {
        "workspace_id": workspace_id,
        "videos": [{"video_id": v["video_id"], "title": v.get("title"), "channel": v.get("channel"),
                     "duration_seconds": v.get("duration_seconds")} for v in videos],
        "claim_count": len(claims),
        "theme_count": len(themes),
        "has_report": (ws / "report.md").exists(),
    }


@app.post("/add")
def add_video(req: AddVideoRequest):
    """Scrape a YouTube video and run the full pipeline (ingest → report)."""
    scraper = Path(__file__).resolve().parent.parent / "scraper" / "youtube.py"
    if not scraper.exists():
        raise HTTPException(500, "Scraper (youtube.py) not found at repo root")

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()

    try:
        # 1. Scrape
        result = subprocess.run(
            ["python3", str(scraper), req.url, "--engine", req.engine, "--output", tmp.name],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            raise HTTPException(500, f"Scraping failed: {result.stderr[-500:]}")

        # 2. Pipeline
        from argparse import Namespace
        from cli import cmd_ingest, cmd_index, cmd_extract_claims, cmd_cluster, cmd_synthesize, cmd_report

        ns_ingest = Namespace(raw_path=tmp.name, workspace_id=req.workspace_id)
        cmd_ingest(ns_ingest)

        ns = Namespace(workspace_id=req.workspace_id)
        cmd_index(ns)
        cmd_extract_claims(ns)
        cmd_cluster(ns)
        try:
            cmd_synthesize(ns)
        except Exception:
            pass  # synthesis can fail on rate limits; non-critical
        cmd_report(ns)

        return {"status": "ok", "workspace_id": req.workspace_id}
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Ask a grounded, citation-verified question against a workspace."""
    from core.config import WORKSPACES_DIR
    ws = Path(WORKSPACES_DIR) / req.workspace_id
    if not ws.exists():
        raise HTTPException(404, f"Workspace '{req.workspace_id}' not found")

    from chat.engine import ask
    result = ask(req.question, workspace_id=req.workspace_id, recent_history=req.history, mode=req.mode)

    return ChatResponse(
        answer=result["answer"],
        sources=result["source_map"],
        citations_valid=result["citation_check"]["all_valid"],
        cited_count=result["citation_check"]["cited_count"],
    )


@app.get("/report/{workspace_id}")
def get_report(workspace_id: str):
    """Return the generated research brief as Markdown."""
    from core.config import WORKSPACES_DIR
    path = Path(WORKSPACES_DIR) / workspace_id / "report.md"
    if not path.exists():
        raise HTTPException(404, "Report not generated yet — add videos and run the pipeline first.")
    return {"workspace_id": workspace_id, "markdown": path.read_text(encoding="utf-8")}


@app.get("/themes/{workspace_id}")
def get_themes(workspace_id: str):
    """Return cross-source themes as JSON."""
    from core.config import WORKSPACES_DIR
    path = Path(WORKSPACES_DIR) / workspace_id / "cross_source_themes.json"
    if not path.exists():
        return {"workspace_id": workspace_id, "themes": []}
    return {"workspace_id": workspace_id, "themes": json.load(open(path))}


@app.get("/claims/{workspace_id}")
def get_claims(workspace_id: str):
    """Return all validated claims for a workspace."""
    from core.config import WORKSPACES_DIR
    path = Path(WORKSPACES_DIR) / workspace_id / "claims.json"
    if not path.exists():
        raise HTTPException(404, "Claims not extracted yet.")
    return {"workspace_id": workspace_id, "claims": json.load(open(path))}


@app.get("/messages/{workspace_id}")
def get_messages(workspace_id: str):
    """Return the persisted conversation for a chat (workspace)."""
    from core.config import WORKSPACES_DIR
    path = Path(WORKSPACES_DIR) / workspace_id / "messages.json"
    if not path.exists():
        return {"workspace_id": workspace_id, "messages": []}
    return {"workspace_id": workspace_id, "messages": json.load(open(path))}


@app.delete("/workspaces/{workspace_id}")
def delete_workspace_endpoint(workspace_id: str):
    """Delete a chat (workspace): its data folder + vectors."""
    import shutil
    from core.config import WORKSPACES_DIR
    ws = Path(WORKSPACES_DIR) / workspace_id
    if not ws.exists():
        raise HTTPException(404, f"Workspace '{workspace_id}' not found")
    try:
        from retrieval.vector_store import delete_workspace
        delete_workspace(workspace_id)
    except Exception:
        pass
    shutil.rmtree(ws)
    return {"status": "deleted", "workspace_id": workspace_id}
