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
from core.config import WORKSPACES_DIR, VECTOR_TOP_K, RERANK_KEEP_TOP
from retrieval.hybrid import hybrid_search
from retrieval.reranker import rerank
from retrieval.context import assemble_context

CITATION_PATTERN = re.compile(r"\[Source (\d+)\]")


def build_grounded_prompt(question: str, context_text: str, recent_history: list[dict] = None) -> str:
    """
    Build the actual prompt sent to Gemini. Recent history is optional and limited —
    per the spec, previous AI answers are NEVER treated as evidence, only as
    conversational context for understanding follow-up questions.
    """
    history_block = ""
    if recent_history:
        history_lines = [f"{turn['role']}: {turn['content']}" for turn in recent_history[-4:]]
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


def call_gemini(prompt: str) -> str:
    """Uses google-genai — the current SDK (google-generativeai is deprecated as of 2026)."""
    from dotenv import load_dotenv
    load_dotenv()

    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text


def verify_citations(answer_text: str, source_map: dict) -> dict:
    """
    Scan the answer for every [Source N] reference and check it against the real
    source_map. Returns a report — this is what stops a hallucinated citation from
    silently reaching the user.
    """
    cited_labels = set(CITATION_PATTERN.findall(answer_text))
    cited_labels = {f"Source {n}" for n in cited_labels}

    valid_citations = cited_labels & source_map.keys()
    invalid_citations = cited_labels - source_map.keys()

    return {
        "cited_count": len(cited_labels),
        "valid_citations": sorted(valid_citations),
        "invalid_citations": sorted(invalid_citations),
        "all_valid": len(invalid_citations) == 0,
    }


def ask(question: str, workspace_id: str, recent_history: list[dict] = None) -> dict:
    """
    The full grounded chat flow, end to end. Returns a dict with the answer,
    the source map (for displaying real citations to the user), and a citation
    verification report (for catching any hallucinated source references).
    """
    fused = hybrid_search(question, workspace_id=workspace_id, top_k=VECTOR_TOP_K)
    reranked = rerank(question, fused, top_k=RERANK_KEEP_TOP)
    context_text, source_map = assemble_context(reranked)

    if not source_map:
        return {
            "answer": "I couldn't find any relevant information in this workspace to answer that.",
            "source_map": {},
            "citation_check": {"cited_count": 0, "valid_citations": [], "invalid_citations": [], "all_valid": True},
        }

    prompt = build_grounded_prompt(question, context_text, recent_history)
    answer_text = call_gemini(prompt)
    citation_check = verify_citations(answer_text, source_map)

    return {
        "answer": answer_text,
        "source_map": source_map,
        "citation_check": citation_check,
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