from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from youtube import process_video

app = FastAPI(title="YouTube Transcript API")


class VideoRequest(BaseModel):
    """Request body for a YouTube transcription job."""

    url: str
    engine: str = "local"
    model: str = "small"
    output: str | None = None


@app.get("/")
def root() -> dict[str, str]:
    """Return basic API information."""
    return {"message": "YouTube Transcript API"}


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.post("/transcribe")
def transcribe(video: VideoRequest) -> dict[str, Any]:
    """Transcribe a YouTube video using the shared process_video workflow."""
    return process_video(
        url=video.url,
        engine=video.engine,
        model=video.model,
        output=video.output,
    )
