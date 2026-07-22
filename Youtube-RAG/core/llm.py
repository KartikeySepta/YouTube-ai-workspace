"""
Centralized Gemini call with 429 retry-and-backoff.

Every Gemini caller in the project (claim_extractor, engine, synthesizer, claim_clusterer)
goes through here, so rate-limit handling lives in ONE place instead of four. The free tier
for gemini-3.1-flash-lite allows only 15 requests/minute; a burst of extraction / adjudication
/ synthesis calls trips that and previously crashed the pipeline with an uncaught 429
(RESOURCE_EXHAUSTED). This retries with exponential backoff, honoring the server's suggested
retryDelay when it provides one.
"""

import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Retry policy
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


def generate_content(prompt: str, model: str | None = None) -> str:
    """
    Single entry point for a Gemini text generation, with 429 backoff.
    Non-429 errors are raised immediately (no point retrying a bad request).
    """
    from dotenv import load_dotenv
    load_dotenv()
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

    # Unreachable: loop either returns or raises on the last attempt.
    raise RuntimeError("generate_content: exhausted retries without returning")
