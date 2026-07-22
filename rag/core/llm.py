"""
Centralized LLM call with a swappable backend (Gemini or local Ollama) + 429 backoff.

Every LLM caller in the project (claim_extractor, engine, synthesizer, claim_clusterer)
goes through here, so backend choice and rate-limit handling live in ONE place.

Backends (set LLM_BACKEND in .env):
  - "gemini" (default): Google Gemini. Free tier = 15 req/min → 429s handled with backoff.
  - "ollama": a LOCAL model via Ollama (http://localhost:11434). No API key, no rate limits,
    no cost — great for testing on a laptop (e.g. MacBook Air M4). Small models are weaker at
    strict JSON, but the pipeline validates/rejects bad output, so it degrades gracefully.
"""

import json as _json
import os
import random
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Retry policy (Gemini only)
MAX_RETRIES = 6
BASE_DELAY = 2.0     # seconds; doubles each retry
MAX_DELAY = 45.0     # cap any single backoff sleep


def _parse_retry_delay(msg: str) -> float | None:
    """Pull the server-suggested wait out of a 429 message, if present."""
    # e.g. "'retryDelay': '31s'"  or  "Please retry in 31.200815084s."
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)\s*s", msg)
    if not m:
        m = re.search(r"retry in (\d+(?:\.\d+)?)\s*s", msg)
    return float(m.group(1)) if m else None


def _is_rate_limit(exc) -> bool:
    status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    msg = str(exc)
    return status == 429 or "RESOURCE_EXHAUSTED" in msg or "429" in msg


def _ollama_generate(prompt: str, model: str) -> str:
    """
    Call a local Ollama server. No rate limits, no cost. Requires Ollama running:
        brew install ollama && ollama serve
        ollama pull llama3.2        # or qwen2.5:3b, phi3.5, etc.
    """
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    payload = _json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},   # deterministic-ish for extraction/adjudication
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama call failed ({e}). Is Ollama running? Try:\n"
            f"  brew install ollama && ollama serve\n"
            f"  ollama pull {model}"
        )


def _gemini_generate(prompt: str, model: str | None) -> str:
    from google import genai
    from google.genai import errors

    if model is None:
        try:
            from core.config import GEMINI_MODEL
            model = GEMINI_MODEL
        except Exception:
            model = "gemini-3.1-flash-lite"

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")
    client = genai.Client(api_key=api_key)

    delay = BASE_DELAY
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            return response.text
        except errors.APIError as e:
            # Only retry rate limits; surface everything else (404 bad model, 400, etc.) at once.
            if not _is_rate_limit(e) or attempt == MAX_RETRIES - 1:
                raise
            wait = _parse_retry_delay(str(e))
            wait = (wait if wait is not None else delay)
            wait = min(wait, MAX_DELAY) + random.uniform(0, 1.0)  # jitter to de-sync parallel callers
            print(f"[llm] 429 rate-limited; backing off {wait:.1f}s then retrying "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})", file=sys.stderr)
            time.sleep(wait)
            delay = min(delay * 2, MAX_DELAY)

    raise RuntimeError("generate_content: exhausted retries without returning")


def generate_content(prompt: str, model: str | None = None) -> str:
    """
    Single entry point for LLM text generation. Routes to the configured backend.
    """
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    backend = os.environ.get("LLM_BACKEND", "gemini").strip().lower()

    if backend == "ollama":
        ollama_model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        return _ollama_generate(prompt, ollama_model)

    return _gemini_generate(prompt, model)
