"""
CLI — one entry point for the whole pipeline instead of remembering separate
one-off commands for each step.

Usage:
  python3 cli.py ingest data/raw/output.json rag_research
  python3 cli.py index rag_research
  python3 cli.py extract-claims rag_research
  python3 cli.py cluster rag_research
  python3 cli.py synthesize rag_research
  python3 cli.py chat rag_research "how does RAG work"
  python3 cli.py evaluate
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_ingest(args):
    """Steps 1-2: parse output.json and chunk it, saving videos.json + chunks.json."""
    from ingestion.loader import load_videos, save_videos_to_workspace
    from ingestion.chunker import chunk_all_videos, save_chunks_to_workspace

    videos = load_videos(args.raw_path)
    videos_path = save_videos_to_workspace(videos, args.workspace_id)
    print(f"Parsed {len(videos)} video(s) -> {videos_path}")

    chunks = chunk_all_videos(videos, args.workspace_id)
    chunks_path = save_chunks_to_workspace(chunks, args.workspace_id)
    print(f"Chunked into {len(chunks)} chunk(s) -> {chunks_path}")


def cmd_index(args):
    """Steps 4-5: embed chunks and index them into the vector store."""
    from core.config import WORKSPACES_DIR
    from retrieval.vector_store import upsert_chunks

    chunks_path = Path(WORKSPACES_DIR) / args.workspace_id / "chunks.json"
    chunks = json.load(open(chunks_path))
    n = upsert_chunks(chunks)
    print(f"Indexed {n} chunk(s) from {chunks_path}")


def cmd_extract_claims(args):
    """Step 3: pull atomic claims out of chunks, with evidence validation."""
    from core.config import WORKSPACES_DIR
    from core.models import TranscriptChunk
    from knowledge.claim_extractor import extract_claims_for_chunks, save_claims_to_workspace

    chunks_path = Path(WORKSPACES_DIR) / args.workspace_id / "chunks.json"
    chunks_raw = json.load(open(chunks_path))
    chunks = [TranscriptChunk(**c) for c in chunks_raw]

    claims, rejections = extract_claims_for_chunks(chunks)
    print(f"{len(claims)} accepted, {len(rejections)} rejected")
    if rejections:
        for r in rejections:
            print(f"  REJECTED: {r.get('reason')}")

    out_path = save_claims_to_workspace(claims, args.workspace_id)
    print(f"Saved to {out_path}")


def cmd_cluster(args):
    """Step 11: group similar claims together."""
    from knowledge.claim_clusterer import run_clustering_for_workspace

    claims, clusters = run_clustering_for_workspace(args.workspace_id)
    multi = [c for c in clusters if c["size"] > 1]
    print(f"{len(claims)} claims -> {len(clusters)} clusters ({len(multi)} with 2+ members)")


def cmd_synthesize(args):
    """Step 12: cross-video agreement/disagreement analysis."""
    from knowledge.synthesizer import run_synthesis_for_workspace

    results = run_synthesis_for_workspace(args.workspace_id)
    single = [r for r in results if r.relationship == "single_source"]
    multi = [r for r in results if r.relationship != "single_source"]
    print(f"{len(results)} clusters synthesized ({len(single)} single-source, {len(multi)} cross-video)")


def cmd_chat(args):
    """Step 10: ask a question, get a grounded, cited answer."""
    from chat.engine import ask

    result = ask(args.question, workspace_id=args.workspace_id)
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Citation check: {result['citation_check']}")


def cmd_evaluate(args):
    """Step 13: run the retrieval evaluation harness."""
    from evals.evaluate import evaluate_retrieval, real_retrieve_fn

    dataset = json.load(open(Path(__file__).parent / "evals" / "dataset.json"))
    results = evaluate_retrieval(dataset, real_retrieve_fn)

    print("=== Aggregate ===")
    print(json.dumps(results["aggregate"], indent=2))
    print("\n=== By category ===")
    for cat, agg in results["by_category"].items():
        print(f"\n{cat}:")
        print(json.dumps(agg, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description="Video RAG tool CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Parse + chunk a raw output.json file")
    p_ingest.add_argument("raw_path", help="Path to scraper output.json")
    p_ingest.add_argument("workspace_id", help="Workspace name, e.g. rag_research")
    p_ingest.set_defaults(func=cmd_ingest)

    p_index = subparsers.add_parser("index", help="Embed + index chunks into the vector store")
    p_index.add_argument("workspace_id")
    p_index.set_defaults(func=cmd_index)

    p_claims = subparsers.add_parser("extract-claims", help="Extract atomic claims from chunks")
    p_claims.add_argument("workspace_id")
    p_claims.set_defaults(func=cmd_extract_claims)

    p_cluster = subparsers.add_parser("cluster", help="Group similar claims together")
    p_cluster.add_argument("workspace_id")
    p_cluster.set_defaults(func=cmd_cluster)

    p_synth = subparsers.add_parser("synthesize", help="Cross-video agreement/disagreement analysis")
    p_synth.add_argument("workspace_id")
    p_synth.set_defaults(func=cmd_synthesize)

    p_chat = subparsers.add_parser("chat", help="Ask a grounded, cited question")
    p_chat.add_argument("workspace_id")
    p_chat.add_argument("question")
    p_chat.set_defaults(func=cmd_chat)

    p_eval = subparsers.add_parser("evaluate", help="Run the retrieval evaluation harness")
    p_eval.set_defaults(func=cmd_evaluate)

    return parser


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-dispatch":
        # ============================================================
        # NO real pipeline calls — proves argparse routes each command
        # to the right function with the right arguments.
        # ============================================================
        parser = build_parser()

        test_cases = [
            (["ingest", "data/raw/output.json", "rag_research"], cmd_ingest,
             {"raw_path": "data/raw/output.json", "workspace_id": "rag_research"}),
            (["index", "rag_research"], cmd_index, {"workspace_id": "rag_research"}),
            (["extract-claims", "rag_research"], cmd_extract_claims, {"workspace_id": "rag_research"}),
            (["cluster", "rag_research"], cmd_cluster, {"workspace_id": "rag_research"}),
            (["synthesize", "rag_research"], cmd_synthesize, {"workspace_id": "rag_research"}),
            (["chat", "rag_research", "how does RAG work"], cmd_chat,
             {"workspace_id": "rag_research", "question": "how does RAG work"}),
            (["evaluate"], cmd_evaluate, {}),
        ]

        all_passed = True
        for argv, expected_func, expected_attrs in test_cases:
            args = parser.parse_args(argv)
            func_ok = args.func == expected_func
            attrs_ok = all(getattr(args, k) == v for k, v in expected_attrs.items())
            status = "PASS" if (func_ok and attrs_ok) else "FAIL"
            if status == "FAIL":
                all_passed = False
            print(f"[{status}] {' '.join(argv)} -> {args.func.__name__}")

        print("\nALL DISPATCH TESTS PASSED" if all_passed else "\nSOME TESTS FAILED")

    else:
        parser = build_parser()
        args = parser.parse_args()
        args.func(args)