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
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _signature(*parts) -> str:
    """Stable short hash of inputs — used to skip re-running expensive (Gemini) steps
    when their inputs (claims, thresholds) haven't changed since the last run."""
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:16]


def _sig_unchanged(ws_dir: Path, tag: str, sig: str) -> bool:
    f = ws_dir / f".{tag}.sig"
    return f.exists() and f.read_text().strip() == sig


def _sig_write(ws_dir: Path, tag: str, sig: str) -> None:
    (ws_dir / f".{tag}.sig").write_text(sig)


def _require_file(path: Path, hint: str) -> bool:
    """Print a friendly message and return False if a required file is missing."""
    if not path.exists():
        print(f"⚠️  Missing: {path.name}\n   {hint}")
        return False
    return True


def _messages_path(workspace_id: str) -> Path:
    from core.config import WORKSPACES_DIR
    return Path(WORKSPACES_DIR) / workspace_id / "messages.json"


def _load_messages(workspace_id: str) -> list:
    p = _messages_path(workspace_id)
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return []
    return []


def _save_messages(workspace_id: str, messages: list) -> None:
    p = _messages_path(workspace_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(messages, f, indent=2)


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

    if not _require_file(Path(args.raw_path), "Path to a scraped output.json (from the scraper). "
                         "Use 'cli.py add <url> <workspace>' to scrape + ingest in one step."):
        return

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
    if not _require_file(chunks_path, "Run 'ingest' first to create chunks."):
        return
    chunks = json.load(open(chunks_path))
    n = upsert_chunks(chunks)
    print(f"Indexed {n} chunk(s) from {chunks_path}")


def cmd_extract_claims(args):
    """Step 3: pull atomic claims out of chunks, with evidence validation + content_hash caching."""
    from core.config import WORKSPACES_DIR, CLAIM_EXTRACTION_BATCH_SIZE
    from core.models import TranscriptChunk
    from knowledge.claim_extractor import (
        extract_claims_for_chunks, save_claims_to_workspace,
        load_claim_cache, save_claim_cache,
    )

    chunks_path = Path(WORKSPACES_DIR) / args.workspace_id / "chunks.json"
    if not _require_file(chunks_path, "Run 'ingest' first to create chunks."):
        return
    chunks_raw = json.load(open(chunks_path))
    chunks = [TranscriptChunk(**c) for c in chunks_raw]

    cache = load_claim_cache(args.workspace_id)
    claims, rejections, cache, stats = extract_claims_for_chunks(
        chunks, batch_size=CLAIM_EXTRACTION_BATCH_SIZE, cache=cache,
    )
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
    """Step 11: group similar claims together (split within/cross-video merge policy).

    Skips entirely (0 Gemini calls) if claims + thresholds are unchanged since last run,
    unless --force is passed. This is the main free-tier saver on re-runs."""
    from core.config import (
        WORKSPACES_DIR, WITHIN_VIDEO_MERGE_THRESHOLD,
        CLAIM_CLUSTER_GRAY_ZONE_LOW, CROSS_VIDEO_ADJUDICATION_FLOOR,
    )
    from knowledge.claim_clusterer import run_clustering_for_workspace

    ws = Path(WORKSPACES_DIR) / args.workspace_id
    claims_path = ws / "claims.json"
    if not _require_file(claims_path, "Run 'extract-claims' first."):
        return

    claims_raw = json.load(open(claims_path))
    sig = _signature([c["claim"] for c in claims_raw],
                     WITHIN_VIDEO_MERGE_THRESHOLD, CLAIM_CLUSTER_GRAY_ZONE_LOW,
                     CROSS_VIDEO_ADJUDICATION_FLOOR)

    if not getattr(args, "force", False) and (ws / "clusters.json").exists() and _sig_unchanged(ws, "cluster", sig):
        n = len(json.load(open(ws / "clusters.json")))
        print(f"✓ claims + thresholds unchanged — reusing {n} cached clusters "
              f"(0 Gemini calls). Use --force to re-run.")
        return

    claims, clusters, stats = run_clustering_for_workspace(args.workspace_id)
    _sig_write(ws, "cluster", sig)
    multi = [c for c in clusters if c["size"] > 1]
    print(f"{len(claims)} claims -> {len(clusters)} clusters ({len(multi)} with 2+ members)")
    print(f"adjudication calls: within-video={stats['within_adjudications']}, "
          f"cross-video={stats['cross_adjudications']} "
          f"(total={stats['within_adjudications'] + stats['cross_adjudications']}); "
          f"cross-video merges kept={stats['cross_merges']}")


def cmd_synthesize(args):
    """Step 12: cross-video agreement/disagreement analysis + cross-source themes.

    Skips (0 Gemini calls) if claims + clusters are unchanged since last run, unless --force."""
    from core.config import WORKSPACES_DIR
    from knowledge.synthesizer import run_synthesis_for_workspace

    ws = Path(WORKSPACES_DIR) / args.workspace_id
    if not _require_file(ws / "clusters.json", "Run 'cluster' first."):
        return

    claims_raw = json.load(open(ws / "claims.json")) if (ws / "claims.json").exists() else []
    clusters_raw = json.load(open(ws / "clusters.json"))
    sig = _signature([c["claim"] for c in claims_raw],
                     [cl["member_claim_ids"] for cl in clusters_raw])

    if not getattr(args, "force", False) and (ws / "cross_source_themes.json").exists() and _sig_unchanged(ws, "synthesis", sig):
        themes = json.load(open(ws / "cross_source_themes.json"))
        print(f"✓ claims + clusters unchanged — reusing synthesis + {len(themes)} themes "
              f"(0 Gemini calls). Use --force to re-run.")
        return

    results, themes = run_synthesis_for_workspace(args.workspace_id)
    _sig_write(ws, "synthesis", sig)
    single = [r for r in results if r.relationship == "single_source"]
    multi = [r for r in results if r.relationship != "single_source"]
    print(f"{len(results)} clusters synthesized ({len(single)} single-source, {len(multi)} cross-video)")
    print(f"{len(themes)} cross-source themes (related across videos, not merged):")
    for t in themes:
        print(f"  [{t['relationship']}] {', '.join(t['videos'])} — {t['synthesis_note']}")


def cmd_chat(args):
    """Ask a single question. --mode grounded (cite-only) or assist (build/apply).
    Conversation is persisted to the workspace so a 'chat' is permanent + resumable."""
    from chat.engine import ask

    history = _load_messages(args.workspace_id)
    recent = [{"role": m["role"], "content": m["content"]} for m in history[-4:]]
    result = ask(args.question, workspace_id=args.workspace_id, recent_history=recent, mode=args.mode)
    print(f"\n[{args.mode} mode] Answer:\n{result['answer']}\n")
    if args.mode == "grounded":
        print(f"Citation check: {result['citation_check']}")

    history.append({"role": "user", "content": args.question, "mode": args.mode})
    history.append({"role": "assistant", "content": result["answer"], "mode": args.mode})
    _save_messages(args.workspace_id, history)


def cmd_talk(args):
    """
    Start an INTERACTIVE chat session — keeps conversation history in memory for
    the length of this session (so follow-up questions make sense), and prints a
    warning if any answer's citations don't check out.

    NOTE: conversation history is PERSISTED to messages.json in the workspace, so a
    chat is permanent and resumes where you left off. Use --fresh to start over, or
    '/clear' mid-session to wipe it.
    """
    from chat.engine import ask

    mode = args.mode
    history = [] if getattr(args, "fresh", False) else _load_messages(args.workspace_id)
    intro = (f"{len(history) // 2} prior message(s) loaded" if history else "new conversation")
    print(f"Chatting with workspace '{args.workspace_id}' [{mode} mode] — {intro}.")
    print("Commands: 'exit'/'quit' stop | '/assist' build mode | '/grounded' fact-only | '/clear' wipe.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if question.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if question.lower() in ("/assist", "/grounded"):
            mode = question.lower().lstrip("/")
            print(f"[switched to {mode} mode]\n")
            continue
        if question.lower() == "/clear":
            history = []
            _save_messages(args.workspace_id, history)
            print("[history cleared]\n")
            continue
        if not question:
            continue

        recent = [{"role": m["role"], "content": m["content"]} for m in history[-4:]]
        result = ask(question, workspace_id=args.workspace_id, recent_history=recent, mode=mode)
        print(f"\nAssistant: {result['answer']}\n")

        if mode == "grounded" and not result["citation_check"]["all_valid"]:
            print(f"[WARNING: unverified citations found: {result['citation_check']['invalid_citations']}]\n")

        history.append({"role": "user", "content": question, "mode": mode})
        history.append({"role": "assistant", "content": result["answer"], "mode": mode})
        _save_messages(args.workspace_id, history)   # persist after every turn


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


def cmd_add(args):
    """One command: paste a YouTube URL → scrape → full pipeline (ingest through report)."""
    import subprocess
    import tempfile
    import os

    scraper = Path(__file__).resolve().parent.parent / "scraper" / "youtube.py"
    if not scraper.exists():
        print(f"ERROR: scraper not found at {scraper}")
        return

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()

    try:
        print(f"📡 Scraping: {args.url}")
        subprocess.run(
            ["python3", str(scraper), args.url, "--engine", args.engine, "--output", tmp.name],
            check=True,
        )

        print(f"\n⚙️ Running pipeline for workspace '{args.workspace_id}'...")
        # Re-use the existing cmd functions by building Namespace objects
        from argparse import Namespace

        ns_ingest = Namespace(raw_path=tmp.name, workspace_id=args.workspace_id)
        cmd_ingest(ns_ingest)

        ns = Namespace(workspace_id=args.workspace_id)
        print("\n📐 Indexing...")
        cmd_index(ns)
        print("\n🧠 Extracting claims...")
        cmd_extract_claims(ns)
        print("\n🔗 Clustering...")
        cmd_cluster(ns)
        print("\n🌐 Synthesizing...")
        try:
            cmd_synthesize(ns)
        except Exception as e:
            print(f"  ⚠️ Synthesis skipped (likely rate limit, will work on next run): {e}")
        print("\n📄 Generating report...")
        cmd_report(ns)

        print(f"\n✅ Done! Your workspace '{args.workspace_id}' is ready.")
        print(f"   → Read the brief: cat data/workspaces/{args.workspace_id}/report.md")
        print(f"   → Chat with it:   python3 cli.py talk {args.workspace_id}")
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)


def cmd_batch(args):
    """Add multiple videos from a text file (one URL per line)."""
    urls = [line.strip() for line in open(args.url_file)
            if line.strip() and not line.strip().startswith('#')]

    if not urls:
        print("No URLs found in file.")
        return

    print(f"Found {len(urls)} video(s) to process.\n")
    from argparse import Namespace

    for i, url in enumerate(urls, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(urls)}] {url}")
        print('='*60)
        add_args = Namespace(url=url, workspace_id=args.workspace_id, engine=args.engine)
        try:
            cmd_add(add_args)
        except Exception as e:
            print(f"  ❌ Failed: {e} — skipping, continuing with next video.")

    print(f"\n{'='*60}")
    print(f"✅ Batch complete. {len(urls)} video(s) processed into '{args.workspace_id}'.")
    print(f"   → python3 cli.py talk {args.workspace_id}")


def cmd_status(args):
    """Show a workspace summary: what's in it, how much, what's generated."""
    from core.config import WORKSPACES_DIR
    ws = Path(WORKSPACES_DIR) / args.workspace_id
    if not ws.exists():
        print(f"Workspace '{args.workspace_id}' does not exist.")
        return

    print(f"Workspace: {args.workspace_id}\n")
    files = {
        "videos": "videos.json",
        "chunks": "chunks.json",
        "claims": "claims.json",
        "clusters": "clusters.json",
        "themes": "cross_source_themes.json",
    }
    for label, fname in files.items():
        p = ws / fname
        if p.exists():
            data = json.load(open(p))
            print(f"  {label:10}: {len(data)} records")
        else:
            print(f"  {label:10}: ❌ not generated")

    report = ws / "report.md"
    print(f"  {'report':10}: {'✅ generated' if report.exists() else '❌ not yet'}")
    cache = ws / "claims_cache.json"
    if cache.exists():
        c = json.load(open(cache))
        print(f"  {'cache':10}: {len(c)} chunk hashes cached")

    # Show video titles
    videos_path = ws / "videos.json"
    if videos_path.exists():
        videos = json.load(open(videos_path))
        print(f"\n  Videos ({len(videos)}):")
        for v in videos:
            print(f"    • {v.get('title', '?')[:60]} ({v.get('channel', '?')})")


def cmd_list(args):
    """List all chats (workspaces) with a one-line summary each."""
    from core.config import WORKSPACES_DIR
    ws_dir = Path(WORKSPACES_DIR)
    dirs = sorted([d for d in ws_dir.iterdir() if d.is_dir()]) if ws_dir.exists() else []
    if not dirs:
        print("No chats yet. Create one:  python3 cli.py add <youtube_url> <chat_name>")
        return
    print(f"{len(dirs)} chat(s):\n")
    for d in dirs:
        def _n(f):
            p = d / f
            return len(json.load(open(p))) if p.exists() else 0
        videos = _n("videos.json")
        claims = _n("claims.json")
        msgs = _n("messages.json") // 2
        titles = ""
        if (d / "videos.json").exists():
            vs = json.load(open(d / "videos.json"))
            titles = "; ".join(v.get("title", "?")[:40] for v in vs[:3])
        print(f"  • {d.name}  —  {videos} video(s), {claims} claims, {msgs} chat msg(s)")
        if titles:
            print(f"      {titles}")


def cmd_delete(args):
    """Delete a chat (workspace): its data, vectors, and conversation history."""
    from core.config import WORKSPACES_DIR
    import shutil
    ws = Path(WORKSPACES_DIR) / args.workspace_id
    if not ws.exists():
        print(f"Chat '{args.workspace_id}' does not exist.")
        return
    if not getattr(args, "yes", False):
        confirm = input(f"Delete chat '{args.workspace_id}' and all its data? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return
    # Remove vectors for this workspace, then the folder.
    try:
        from retrieval.vector_store import delete_workspace
        delete_workspace(args.workspace_id)
    except Exception as e:
        print(f"  (vector cleanup skipped: {e})")
    shutil.rmtree(ws)
    print(f"✅ Deleted chat '{args.workspace_id}'.")


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
    p_cluster.add_argument("--force", action="store_true", help="Re-run even if inputs unchanged")
    p_cluster.set_defaults(func=cmd_cluster)

    p_synth = subparsers.add_parser("synthesize", help="Cross-video agreement/disagreement analysis")
    p_synth.add_argument("workspace_id")
    p_synth.add_argument("--force", action="store_true", help="Re-run even if inputs unchanged")
    p_synth.set_defaults(func=cmd_synthesize)

    p_report = subparsers.add_parser("report", help="Generate a cited Markdown research brief (no LLM calls)")
    p_report.add_argument("workspace_id")
    p_report.set_defaults(func=cmd_report)

    p_chat = subparsers.add_parser("chat", help="Ask a single grounded, cited question")
    p_chat.add_argument("workspace_id")
    p_chat.add_argument("question")
    p_chat.add_argument("--mode", choices=["grounded", "assist"], default="grounded",
                        help="grounded = cite-only (no hallucination); assist = build/apply using video knowledge + expertise")
    p_chat.set_defaults(func=cmd_chat)

    p_talk = subparsers.add_parser("talk", help="Start an interactive chat session (persistent history)")
    p_talk.add_argument("workspace_id")
    p_talk.add_argument("--mode", choices=["grounded", "assist"], default="grounded",
                        help="starting mode; switch live with /assist or /grounded")
    p_talk.add_argument("--fresh", action="store_true", help="Start a new conversation (ignore saved history)")
    p_talk.set_defaults(func=cmd_talk)

    p_list = subparsers.add_parser("list", help="List all chats (workspaces)")
    p_list.set_defaults(func=cmd_list)

    p_delete = subparsers.add_parser("delete", help="Delete a chat (workspace) and all its data")
    p_delete.add_argument("workspace_id")
    p_delete.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    p_delete.set_defaults(func=cmd_delete)

    p_eval = subparsers.add_parser("evaluate", help="Run the retrieval evaluation harness")
    p_eval.set_defaults(func=cmd_evaluate)

    p_add = subparsers.add_parser("add", help="Add a YouTube video: scrape + full pipeline in one shot")
    p_add.add_argument("url", help="YouTube video URL")
    p_add.add_argument("workspace_id", help="Workspace name, e.g. fiverr_tips")
    p_add.add_argument("--engine", choices=["cloud", "local"], default="cloud",
                       help="Transcription engine (default: cloud/Gemini)")
    p_add.set_defaults(func=cmd_add)

    p_batch = subparsers.add_parser("batch", help="Add multiple videos from a URL list file")
    p_batch.add_argument("url_file", help="Text file with one YouTube URL per line")
    p_batch.add_argument("workspace_id", help="Workspace name")
    p_batch.add_argument("--engine", choices=["cloud", "local"], default="cloud")
    p_batch.set_defaults(func=cmd_batch)

    p_status = subparsers.add_parser("status", help="Show workspace summary")
    p_status.add_argument("workspace_id")
    p_status.set_defaults(func=cmd_status)

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