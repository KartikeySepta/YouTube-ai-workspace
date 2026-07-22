"""
Centralized LLM call with multiple swappable backends + automatic fallback.

Every LLM caller in the project (claim_extractor, engine, synthesizer, claim_clusterer)
goes through generate_content(), so provider choice, fallback, and rate-limit handling
live in ONE place.

Backends (set LLM_BACKEND in .env):
  - "gemini"  (default) : Google Gemini            (free tier: 15 req/min)
  - "mistral"           : Mistral API              (OpenAI-compatible)
  - "grok"              : xAI Grok                 (OpenAI-compatible)
  - "ollama"            : LOCAL model via Ollama    (no key, no limits, no cost)
  - "auto"              : try gemini -> mistral -> grok (whichever have keys), then
                          fall through on rate limits. This stacks your free tiers so
                          you rarely wait on any single provider's limit.
  - "gemini,mistral"    : an explicit comma-separated fallback chain of your choosing.

Only the providers whose API key is present are used. Ollama needs a running server.
"""

import json as _json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Retry policy (applies when the WHOLE chain is exhausted in one pass)
MAX_RETRIES = 6
BASE_DELAY = 2.0
MAX_DELAY = 45.0

DEFAULT_MODELS = {
    "gemini": "gemini-3.1-flash-lite",
    "mistral": "mistral-small-latest",
    "grok": "grok-2-latest",
    "ollama": "llama3.2",
}

OPENAI_COMPATIBLE = {
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "grok": "https://api.x.ai/v1/chat/completions",
}


class _RateLimit(Exception):
    """Raised when a provider signals a rate limit (429) — triggers fallback."""


def _looks_rate_limited(msg: str, status=None) -> bool:
    return (status == 429) or "RESOURCE_EXHAUSTED" in msg or "429" in msg or "rate limit" in msg.lower()


def _has_key(provider: str) -> bool:
    if provider == "gemini":
        return bool(os.environ.get("GEMINI_API_KEY"))
    if provider == "mistral":
        return bool(os.environ.get("MISTRAL_API_KEY"))
    if provider == "grok":
        return bool(os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"))
    if provider == "ollama":
        return True  # no key; server availability checked at call time
    return False


# ─── PER-PROVIDER SINGLE-ATTEMPT CALLERS (raise _RateLimit on 429) ───────────────

def _gemini_once(prompt: str, model: str) -> str:
    from google import genai
    from google.genai import errors
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        return client.models.generate_content(model=model, contents=prompt).text
    except errors.APIError as e:
        if _looks_rate_limited(str(e), getattr(e, "code", None)):
            raise _RateLimit(str(e))
        raise


def _openai_compatible_once(prompt: str, url: str, api_key: str, model: str) -> str:
    body = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="ignore")
        if _looks_rate_limited(msg, e.code):
            raise _RateLimit(msg)
        raise RuntimeError(f"{url} HTTP {e.code}: {msg[:200]}")


def _ollama_once(prompt: str, model: str) -> str:
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434") + "/api/generate"
    body = _json.dumps({"model": model, "prompt": prompt, "stream": False,
                        "options": {"temperature": 0}}).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return _json.loads(resp.read().decode("utf-8")).get("response", "")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama not reachable ({e}). Run: ollama serve && ollama pull {model}")


def _call_provider(provider: str, prompt: str, model: str | None = None) -> str:
    if provider == "gemini":
        return _gemini_once(prompt, model or os.environ.get("GEMINI_MODEL") or DEFAULT_MODELS["gemini"])
    if provider in OPENAI_COMPATIBLE:
        key = (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")) if provider == "grok" \
            else os.environ.get("MISTRAL_API_KEY")
        env_model = os.environ.get(f"{provider.upper()}_MODEL")
        return _openai_compatible_once(prompt, OPENAI_COMPATIBLE[provider], key,
                                       model or env_model or DEFAULT_MODELS[provider])
    if provider == "ollama":
        return _ollama_once(prompt, model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODELS["ollama"])
    raise ValueError(f"Unknown LLM provider: {provider}")


def _resolve_chain(backend: str) -> list[str]:
    if backend == "auto":
        chain = [p for p in ("gemini", "mistral", "grok") if _has_key(p)]
        return chain or ["ollama"]
    if "," in backend:
        return [p.strip() for p in backend.split(",") if p.strip()]
    return [backend]


def _task_backend(task: str | None) -> str:
    """Per-task backend: {TASK}_BACKEND env if set, else the global LLM_BACKEND."""
    if task:
        v = os.environ.get(f"{task.upper()}_BACKEND")
        if v:
            return v.strip().lower()
    return os.environ.get("LLM_BACKEND", "gemini").strip().lower()


def generate_content(prompt: str, task: str | None = None, model: str | None = None) -> str:
    """
    Single entry point for LLM text generation, with PER-TASK routing.

    `task` is one of: "extraction", "adjudication", "synthesis", "chat" (or None).
    Each task can be pointed at its own backend via a {TASK}_BACKEND env var — e.g. route
    high-volume EXTRACTION to a free local model while quality-critical CHAT uses the best
    cloud tier. If a task has no override, it uses the global LLM_BACKEND. Within the chosen
    backend, providers fall through on rate limits; we back off only when the whole chain
    is exhausted in a pass.
    """
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

    backend = _task_backend(task)
    chain = _resolve_chain(backend)
    # A per-task/explicit model only applies when the task points at ONE provider (models are
    # provider-specific; in a chain each provider uses its own default/env model).
    per_call_model = model or (os.environ.get(f"{task.upper()}_MODEL") if task else None)
    per_call_model = per_call_model if len(chain) == 1 else None

    delay = BASE_DELAY
    last_err = None
    for attempt in range(MAX_RETRIES):
        for provider in chain:
            if provider in ("gemini", "mistral", "grok") and not _has_key(provider):
                continue
            try:
                return _call_provider(provider, prompt, per_call_model)
            except _RateLimit as e:
                last_err = e
                print(f"[llm:{task or 'default'}] {provider} rate-limited → next provider", file=sys.stderr)
            except Exception as e:
                last_err = e
                print(f"[llm:{task or 'default'}] {provider} failed ({str(e)[:80]}) → next provider", file=sys.stderr)

        # Whole chain failed this pass → back off, then retry the chain.
        wait = min(delay, MAX_DELAY) + random.uniform(0, 1.0)
        print(f"[llm:{task or 'default'}] all providers exhausted; backing off {wait:.1f}s "
              f"(pass {attempt + 1}/{MAX_RETRIES})", file=sys.stderr)
        time.sleep(wait)
        delay = min(delay * 2, MAX_DELAY)

    raise RuntimeError(f"generate_content: all providers failed. Last error: {last_err}")
