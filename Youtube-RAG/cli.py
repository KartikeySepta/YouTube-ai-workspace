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
    """
    Steps 1-2: parse output.json, chunk it, and MERGE into the workspace.

    Ingest is additive, not replace: it loads the workspace's existing videos.json /
    chunks.json and only appends what's genuinely new. Deduplication uses the existing
    per-chunk content_hash — re-ingesting a video already in the corpus produces identical
    hashes, so nothing is added (a true no-op instead of a duplicate). This is what lets
    you build a multi-video workspace by ingesting one output.json at a time.
    """
    from dataclasses import asdict
    from collections import defaultdict
    from ingestion.loader import load_videos
    from ingestion.chunker import chunk_all_videos
    from core.config import WORKSPACES_DIR

    ws_dir = Path(WORKSPACES_DIR) / args.workspace_id
    videos_path = ws_dir / "videos.json"
    chunks_path = ws_dir / "chunks.json"

    # Existing corpus (empty on first ingest).
    existing_videos = json.load(open(videos_path)) if videos_path.exists() else []
    existing_chunks = json.load(open(chunks_path)) if chunks_path.exists() else []
    existing_video_ids = {v["video_id"] for v in existing_videos}
    existing_hashes = {c["content_hash"] for c in existing_chunks}

    # New material from this raw file.
    new_videos = load_videos(args.raw_path)
    new_chunks = chunk_all_videos(new_videos, args.workspace_id)
    chunks_by_video = defaultdict(list)
    for c in new_chunks:
        chunks_by_video[c.video_id].append(c)

    added_videos, skipped_videos = [], []
    added_chunks, skipped_chunk_count = [], 0

    for v in new_videos:
        v_new_chunks = [c for c in chunks_by_video[v.video_id] if c.content_hash not in existing_hashes]
        skipped_chunk_count += len(chunks_by_video[v.video_id]) - len(v_new_chunks)
        already_have_video = v.video_id in existing_video_ids

        if not v_new_chunks and already_have_video:
            skipped_videos.append(v.video_id)   # fully duplicate -> no-op
            continue
        if not already_have_video:
            added_videos.append(v)
            existing_video_ids.add(v.video_id)
        for c in v_new_chunks:
            added_chunks.append(c)
            existing_hashes.add(c.content_hash)

    merged_videos = existing_videos + [asdict(v) for v in added_videos]
    merged_chunks = existing_chunks + [asdict(c) for c in added_chunks]

    ws_dir.mkdir(parents=True, exist_ok=True)
    with open(videos_path, "w", encoding="utf-8") as f:
        json.dump(merged_videos, f, indent=2)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(merged_chunks, f, indent=2)

    print(f"videos: +{len(added_videos)} added, {len(skipped_videos)} skipped (already in corpus) "
          f"-> {len(merged_videos)} total")
    print(f"chunks: +{len(added_chunks)} added, {skipped_chunk_count} skipped (duplicate content_hash) "
          f"-> {len(merged_chunks)} total")
    if skipped_videos:
        print(f"  no-op videos: {', '.join(skipped_videos)}")


def cmd_index(args):
    """Steps 4-5: embed chunks and index them into the vector store."""
    from core.config import WORKSPACES_DIR
    from retrieval.vector_store import upsert_chunks

    chunks_path = Path(WORKSPACES_DIR) / args.workspace_id / "chunks.json"
    chunks = json.load(open(chunks_path))
    n = upsert_chunks(chunks)
    print(f"Indexed {n} chunk(s) from {chunks_path}")


def cmd_extract_claims(args):
    """Step 3: pull atomic claims out of chunks, with evidence validation + content_hash caching."""
    from core.config import WORKSPACES_DIR
    from core.models import TranscriptChunk
    from knowledge.claim_extractor import (
        extract_claims_for_chunks, save_claims_to_workspace,
        load_claim_cache, save_claim_cache,
    )

    chunks_path = Path(WORKSPACES_DIR) / args.workspace_id / "chunks.json"
    chunks_raw = json.load(open(chunks_path))
    chunks = [TranscriptChunk(**c) for c in chunks_raw]

    cache = load_claim_cache(args.workspace_id)
    claims, rejections, cache, stats = extract_claims_for_chunks(chunks, cache=cache)
    save_claim_cache(cache, args.workspace_id)

    print(f"chunks: {stats['cached_chunks']} cached-skipped, {stats['new_chunks']} newly extracted "
          f"({stats['gemini_calls']} Gemini calls)")
    print(f"claims: +{stats['new_claims']} new -> {stats['total_claims']} total, {len(rejections)} rejected")
    if rejections:
        for r in rejections:
            print(f"  REJECTED: {r.get('reason')}")

    out_path = save_claims_to_workspace(claims, args.workspace_id)
    print(f"Saved to {out_path}")


def cmd_cluster(args):
    """Step 11: group similar claims together (split within/cross-video merge policy)."""
    from knowledge.claim_clusterer import run_clustering_for_workspace

    claims, clusters, stats = run_clustering_for_workspace(args.workspace_id)
    multi = [c for c in clusters if c["size"] > 1]
    print(f"{len(claims)} claims -> {len(clusters)} clusters ({len(multi)} with 2+ members)")
    print(f"adjudication calls: within-video={stats['within_adjudications']}, "
          f"cross-video={stats['cross_adjudications']} "
          f"(total={stats['within_adjudications'] + stats['cross_adjudications']}); "
          f"cross-video merges kept={stats['cross_merges']}")


def cmd_synthesize(args):
    """Step 12: cross-video agreement/disagreement analysis + cross-source themes."""
    from knowledge.synthesizer import run_synthesis_for_workspace

    results, themes = run_synthesis_for_workspace(args.workspace_id)
    single = [r for r in results if r.relationship == "single_source"]
    multi = [r for r in results if r.relationship != "single_source"]
    print(f"{len(results)} clusters synthesized ({len(single)} single-source, {len(multi)} cross-video)")
    print(f"{len(themes)} cross-source themes (related across videos, not merged):")
    for t in themes:
        print(f"  [{t['relationship']}] {', '.join(t['videos'])} — {t['synthesis_note']}")


def cmd_chat(args):
    """Step 10: ask a single question, get a grounded, cited answer."""
    from chat.engine import ask

    result = ask(args.question, workspace_id=args.workspace_id)
    print(f"\nAnswer:\n{result['answer']}\n")
    print(f"Citation check: {result['citation_check']}")


def cmd_talk(args):
    """
    Start an INTERACTIVE chat session — keeps conversation history in memory for
    the length of this session (so follow-up questions make sense), and prints a
    warning if any answer's citations don't check out.

    NOTE: history lives only in memory for this run — it is NOT saved to disk
    between separate `talk` sessions yet. That's a real, known gap (messages.json
    persistence hasn't been built) — flagging it honestly rather than pretending
    this is full persistent chat history.
    """
    from chat.engine import ask

    print(f"Chatting with workspace '{args.workspace_id}'. Type 'exit' or 'quit' to stop.\n")
    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not question:
            continue

        result = ask(question, workspace_id=args.workspace_id, recent_history=history)
        print(f"\nAssistant: {result['answer']}\n")

        if not result["citation_check"]["all_valid"]:
            print(f"[WARNING: unverified citations found: {result['citation_check']['invalid_citations']}]\n")

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


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


def cmd_report(args):
    """
    Generate a human-readable, cited research brief (Markdown) from a processed workspace.
    Pure assembly from existing JSON — no LLM calls — so it is fast and deterministic.
    Produces: sources, cross-source themes (what multiple creators address + how they relate),
    and each creator's most-emphasized points, every claim cited to video + timestamp.
    """
    from core.config import WORKSPACES_DIR
    ws = Path(WORKSPACES_DIR) / args.workspace_id

    def load(name, default):
        p = ws / name
        return json.load(open(p)) if p.exists() else default

    videos = load("videos.json", [])
    claims = load("claims.json", [])
    clusters = load("clusters.json", [])
    themes = load("cross_source_themes.json", [])
    chunks = {c["chunk_id"]: c for c in load("chunks.json", [])}
    if not claims:
        print("Nothing to report — run ingest/index/extract-claims/cluster/synthesize first.")
        return

    vmeta = {v["video_id"]: v for v in videos}
    claim_by_id = {c["claim_id"]: c for c in claims}

    def fmt_ts(sec):
        sec = int(sec or 0)
        h, r = divmod(sec, 3600); m, s = divmod(r, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def cite(claim):
        ev = (claim.get("evidence") or [{}])[0]
        ch = chunks.get(ev.get("chunk_id", ""), {})
        vid = ch.get("video_id") or claim.get("video_id", "")
        title = vmeta.get(vid, {}).get("title", vid)
        start = ch.get("start_seconds", 0)
        est = "~" if ch.get("is_estimated") else ""
        url = f"https://youtu.be/{vid}?t={int(start or 0)}"
        return f'“{claim["claim"]}” — *{title}* @ {est}{fmt_ts(start)} ([watch]({url}))'

    lines = []
    lines.append(f"# Research Brief: {args.workspace_id}\n")
    lines.append(f"Synthesized from **{len(videos)} videos** and **{len(claims)} validated, "
                 f"evidence-backed claims**. Every claim below is traceable to a real transcript "
                 f"timestamp; claims whose evidence chunk_id could not be verified were discarded "
                 f"during extraction.\n")

    lines.append("## Sources\n")
    for v in videos:
        dur = fmt_ts(v.get("duration_seconds", 0))
        n = sum(1 for c in claims if c.get("video_id") == v["video_id"])
        lines.append(f"- **{v.get('title','(untitled)')}** — {v.get('channel','?')} "
                     f"({dur}, {n} claims)")
    lines.append("")

    lines.append("## Cross-source themes\n")
    lines.append("_Topics addressed by more than one creator. Related claims are surfaced "
                 "together and labeled by how they relate — they are **not** blindly merged, so "
                 "e.g. the same statistic for different regions is flagged as different context, "
                 "not collapsed into one number._\n")
    if not themes:
        lines.append("_No cross-source themes detected yet (need 2+ videos discussing overlapping points)._\n")
    for t in themes:
        rel = t.get("relationship", "related").replace("_", " ")
        lines.append(f"### {rel.title()} — across {', '.join(t.get('videos', []))}")
        if t.get("synthesis_note"):
            lines.append(f"> {t['synthesis_note']}\n")
        for cid in t.get("member_claim_ids", []):
            if cid in claim_by_id:
                lines.append(f"- {cite(claim_by_id[cid])}")
        lines.append("")

    lines.append("## Most-emphasized points per source\n")
    lines.append("_Claims a creator made more than once (grouped by the within-video merge policy) "
                 "— a proxy for the points each creator stressed._\n")
    prov = {c["claim_id"]: ((c.get("evidence") or [{}])[0].get("chunk_id", "").rsplit("_c", 1)[0]
                            or c.get("video_id")) for c in claims}
    emphasized = [cl for cl in clusters if cl["size"] > 1]
    if not emphasized:
        lines.append("_No repeated claims detected._\n")
    for cl in sorted(emphasized, key=lambda c: -c["size"]):
        vids = {prov.get(m) for m in cl["member_claim_ids"]}
        vtitle = ", ".join(sorted(vmeta.get(v, {}).get("title", v or "?") for v in vids))
        first = claim_by_id.get(cl["member_claim_ids"][0])
        if first:
            lines.append(f"- ({cl['size']}×, *{vtitle}*) {cite(first)}")
    lines.append("")

    out = ws / "report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote research brief -> {out}")
    print(f"  {len(videos)} sources, {len(claims)} claims, {len(themes)} cross-source themes, "
          f"{len(emphasized)} emphasized-point groups")


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

    p_report = subparsers.add_parser("report", help="Generate a cited Markdown research brief (no LLM calls)")
    p_report.add_argument("workspace_id")
    p_report.set_defaults(func=cmd_report)

    p_chat = subparsers.add_parser("chat", help="Ask a single grounded, cited question")
    p_chat.add_argument("workspace_id")
    p_chat.add_argument("question")
    p_chat.set_defaults(func=cmd_chat)

    p_talk = subparsers.add_parser("talk", help="Start an interactive chat session (in-memory history)")
    p_talk.add_argument("workspace_id")
    p_talk.set_defaults(func=cmd_talk)

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
            (["talk", "rag_research"], cmd_talk, {"workspace_id": "rag_research"}),
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