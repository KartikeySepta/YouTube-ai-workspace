"""
STEP 10: GROUNDED CHAT

Wires together everything built so far into the actual feature users interact with:

  question -> hybrid retrieve (Steps 5-7) -> rerank (Step 8) -> assemble context (Step 9)
    -> build grounded prompt -> call Gemini -> verify citations against source_map
    -> return answer + verified sources

The two rules this file enforces, both inherited from the project's original spec:
  1. Gemini is ONLY allowed to use the sources handed to it — never prior conversation
     turns, never its own training knowledge, for factual claims about the videos.
  2. Every citation in the response gets checked against the real source_map BEFORE
     it's shown to the user. If Gemini cites "Source 7" but only 4 sources existed,
     that's caught here, not silently trusted.
"""

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import WORKSPACES_DIR, VECTOR_TOP_K, RERANK_KEEP_TOP, GEMINI_MODEL
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from retrieval.context import assemble_context

# Match citation brackets that mention Source(s), including grouped forms the model
# actually produces: "[Source 1]", "[Source 1, Source 2]", "[Source 1 and Source 3]".
# An earlier version only matched a number immediately followed by "]", so grouped
# citations slipped through UNVERIFIED — a hole in the whole anti-hallucination guarantee.
CITATION_BRACKET_PATTERN = re.compile(r"\[([^\[\]]*?Source[^\[\]]*?)\]")
SOURCE_NUMBER_PATTERN = re.compile(r"Source\s+(\d+)")


def build_grounded_prompt(question: str, context_text: str, recent_history: list[dict] = None) -> str:
    """
    Build the actual prompt sent to Gemini. Recent history is optional and limited —
    per the spec, previous AI answers are NEVER treated as evidence, only as
    conversational context for understanding follow-up questions.
    """
    history_block = ""
    if recent_history:
        history_lines = [f"{turn['role']}: {turn['content']}" for turn in recent_history[-12:]]
        history_block = "Recent conversation (for context only, NOT evidence):\n" + "\n".join(history_lines) + "\n\n"

    return f"""You are answering questions using ONLY the sources below. These sources are excerpts
from YouTube video transcripts, with approximate timestamps.

RULES:
- Answer using ONLY the sources provided. Do not use outside knowledge about the topic.
- Cite every factual claim with the source it came from, like [Source 1].
- If sources disagree, say so explicitly rather than picking one silently.
- If the sources don't contain enough information to answer, say so honestly —
  do not guess or fill gaps with general knowledge.
- Never invent a timestamp, quote, or detail not actually present in the sources.

{history_block}SOURCES:
{context_text}

QUESTION: {question}

ANSWER:"""


def build_assist_prompt(question: str, context_text: str, recent_history: list[dict] = None) -> str:
    """
    ASSIST / BUILD mode. Unlike grounded mode, this lets the model APPLY the video
    knowledge — plus its own expertise — to actually help the user DO something (e.g.
    "build an MCP server based on these videos"). The videos are the foundation and
    should be cited when used, but the model may fill gaps with correct general knowledge
    to give complete, actionable help. This trades strict no-hallucination for usefulness,
    so it's an explicit opt-in mode.
    """
    history_block = ""
    if recent_history:
        history_lines = [f"{turn['role']}: {turn['content']}" for turn in recent_history[-12:]]
        history_block = "Recent conversation:\n" + "\n".join(history_lines) + "\n\n"

    return f"""You are an expert assistant helping the user ACT on what a set of YouTube videos teach.
The sources below are excerpts from those video transcripts.

HOW TO ANSWER:
- Treat the sources as the primary foundation. When you use something from them, cite it like [Source 1].
- You MAY go beyond the sources and use your own expert knowledge to give a COMPLETE, actionable
  answer (real steps, real code, real examples) — the user WANTS you to build/explain fully,
  not just quote the transcript.
- When you add information that is NOT from the sources, make it clearly your own contribution
  (e.g. "Beyond what the video shows, a typical implementation is…"). Never fabricate a quote or
  timestamp and attribute it to a source.
- Stay on the topic the videos are about. If the user asks something entirely unrelated to the
  videos' subject, say so.

{history_block}SOURCES (foundation from the videos):
{context_text}

USER REQUEST: {question}

ANSWER (grounded in the videos where possible, completed with your expertise where needed):"""


def call_gemini(prompt: str) -> str:
    """Delegates to the centralized wrapper (per-task routing + fallback, core/llm.py)."""
    from core.llm import generate_content
    return generate_content(prompt, task="chat")


def verify_citations(answer_text: str, source_map: dict) -> dict:
    """
    Scan the answer for every [Source N] reference and check it against the real
    source_map. Returns a report — this is what stops a hallucinated citation from
    silently reaching the user.
    """
    cited_labels = set()
    for bracket_contents in CITATION_BRACKET_PATTERN.findall(answer_text):
        for n in SOURCE_NUMBER_PATTERN.findall(bracket_contents):
            cited_labels.add(f"Source {n}")

    valid_citations = cited_labels & source_map.keys()
    invalid_citations = cited_labels - source_map.keys()

    return {
        "cited_count": len(cited_labels),
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "all_valid": len(invalid_citations) == 0,
    }


def ask(question: str, workspace_id: str, recent_history: list[dict] = None, mode: str = "grounded") -> dict:
    """
    The full chat flow, end to end. Returns the answer, the source map, and a citation
    verification report.

    mode:
      "grounded" (default) — answer ONLY from the sources; refuse if not present. Zero
                             hallucination; best for research/fact-checking.
      "assist"             — use the sources as foundation + the model's own expertise to
                             actually help the user DO the thing (build/apply). More useful,
                             less strictly verifiable.
    """
    fused = hybrid_search(question, workspace_id=workspace_id, top_k=VECTOR_TOP_K)
    reranked = rerank(question, fused, top_k=RERANK_KEEP_TOP)
    context_text, source_map = assemble_context(reranked)

    if not source_map:
        return {
            "answer": "I couldn't find any relevant information in this workspace to answer that.",
            "source_map": {},
            "citation_check": {"cited_count": 0, "valid_citations": [], "invalid_citations": [], "all_valid": True},
            "mode": mode,
            "verified": mode == "grounded",
            "caveat": None,
        }

    if mode == "assist":
        prompt = build_assist_prompt(question, context_text, recent_history)
    else:
        prompt = build_grounded_prompt(question, context_text, recent_history)
    answer_text = call_gemini(prompt)
    citation_check = verify_citations(answer_text, source_map)

    # Anti-hallucination surfacing: grounded mode is verifiable (answer must come from
    # sources); assist mode is NOT — it may use the model's own knowledge, so its [Source N]
    # tags are the model's own attributions and are NOT a guarantee. Make that explicit to
    # every caller (CLI + API) rather than letting an assist answer look verified.
    verified = (mode == "grounded") and citation_check["all_valid"]
    caveat = (
        None if mode == "grounded"
        else "assist mode — answer may include the model's own knowledge; [Source N] tags "
             "are NOT verified against the videos."
    )

    return {
        "answer": answer_text,
        "source_map": source_map,
        "citation_check": citation_check,
        "mode": mode,
        "verified": verified,
        "caveat": caveat,
    }


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-citation-verifier":
        # ============================================================
        # NO NETWORK CALL — tests citation verification against a
        # hand-built fake answer with one hallucinated citation mixed in.
        # ============================================================
        fake_source_map = {
            "Source 1": {"chunk_id": "c1", "video_id": "vidA"},
            "Source 2": {"chunk_id": "c2", "video_id": "vidA"},
        }
        fake_answer = "RAG works by chunking text [Source 1] and then embedding it [Source 2]. It also does X [Source 7]."

        result = verify_citations(fake_answer, fake_source_map)
        print("Citation check result:")
        print(json.dumps(result, indent=2))
        print("\nEXPECTED: Source 1 and 2 valid, Source 7 invalid, all_valid=False")
        assert result["valid_citations"] == ["Source 1", "Source 2"]
        assert result["invalid_citations"] == ["Source 7"]
        assert result["all_valid"] is False
        print("\nALL ASSERTIONS PASSED")

    else:
        workspace_id = sys.argv[1] if len(sys.argv) > 1 else "rag_research"
        question = sys.argv[2] if len(sys.argv) > 2 else "how does RAG work"

        result = ask(question, workspace_id=workspace_id)

        print(f"Question: {question}\n")
        print(f"Answer:\n{result['answer']}\n")
        print(f"Citation check: {result['citation_check']}\n")
        print("Source map:")
        for label, info in result["source_map"].items():
            print(f"  {label}: {info['video_title']} @ {info['timestamp']}")