"""
Silence noisy ML library output (Hugging Face token warning + model-loading progress
bars) for a clean CLI/demo experience. Import this FIRST, before sentence-transformers /
transformers / huggingface_hub get imported.

Nothing here changes behavior — it only quiets logging/telemetry/progress output.
"""

import logging
import os
import warnings
from pathlib import Path

# Disable HF telemetry + progress bars + noisy verbosity (safe, no behavior change).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# The "sending unauthenticated requests to the HF Hub / set HF_TOKEN" warning comes from a
# network check against the Hub. Our two models (bge-small, ms-marco reranker) are tiny and
# cached after first use, so if they're already cached we go OFFLINE — that skips the Hub
# check entirely (no warning, faster load). If NOT cached yet (fresh install), we stay online
# so the first download still works.
def _models_cached() -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub.exists():
        return False
    names = " ".join(p.name for p in hub.iterdir())
    return ("bge-small" in names) and ("ms-marco" in names)

if _models_cached():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Belt-and-suspenders: quiet the loggers and the one-off warnings too.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*unauthenticated.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
