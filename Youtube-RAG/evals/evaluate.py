"""
STEP 13: EVALUATION

Measures whether retrieval actually finds the right evidence — the question that
matters BEFORE judging prose quality. Uses the hand-written dataset (real questions,
real expected chunk_ids from your actual video) to compute hit rate at K.

Run this after ANY change to chunking, embeddings, fusion weights, or reranking —
it's the only way to know if a change actually helped or just felt like it should.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import WORKSPACES_DIR, VECTOR_TOP_K, RERANK_KEEP_TOP


def hit_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> bool:
    """True if ANY expected chunk_id appears in the top k retrieved results."""
    top_k = set(retrieved_ids[:k])
    return any(exp_id in top_k for exp_id in expected_ids)


def reciprocal_rank(retrieved_ids: list[str], expected_ids: list[str]) -> float:
    """1/rank of the first expected chunk found, or 0 if not found at all."""
    for rank, retrieved_id in enumerate(retrieved_ids, start=1):
        if retrieved_id in expected_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(dataset: list[dict], retrieve_fn, k_values: list[int] = [1, 3, 5]) -> dict:
    """
    `retrieve_fn(question, workspace_id) -> list[chunk_id]` should return chunk_ids
    in rank order (best first) — this lets us test with either the real pipeline
    or a fake stand-in, same pattern as every other network-blocked step so far.
    """
    per_question_results = []

    for item in dataset:
        retrieved_ids = retrieve_fn(item["question"], item["workspace_id"])
        expected_ids = item["expected_chunk_ids"]

        result = {
            "question": item["question"],
            "category": item.get("category", "unlabeled"),
            "expected": expected_ids,
            "retrieved_top5": retrieved_ids[:5],
            "reciprocal_rank": reciprocal_rank(retrieved_ids, expected_ids),
        }
        for k in k_values:
            result[f"hit@{k}"] = hit_at_k(retrieved_ids, expected_ids, k)

        per_question_results.append(result)

    def compute_aggregate(subset):
        if not subset:
            return None
        agg = {}
        for k in k_values:
            hits = sum(1 for r in subset if r[f"hit@{k}"])
            agg[f"hit_rate@{k}"] = hits / len(subset)
        agg["mean_reciprocal_rank"] = sum(r["reciprocal_rank"] for r in subset) / len(subset)
        agg["n"] = len(subset)
        return agg

    aggregate = compute_aggregate(per_question_results)
    by_category = {}
    categories = {r["category"] for r in per_question_results}
    for cat in categories:
        subset = [r for r in per_question_results if r["category"] == cat]
        by_category[cat] = compute_aggregate(subset)

    return {"per_question": per_question_results, "aggregate": aggregate, "by_category": by_category}


def real_retrieve_fn(question: str, workspace_id: str) -> list[str]:
    """The actual pipeline: hybrid search -> rerank -> return ranked chunk_ids."""
    from retrieval.hybrid import hybrid_search
    from retrieval.reranker import rerank

    fused = hybrid_search(question, workspace_id=workspace_id, top_k=VECTOR_TOP_K)
    reranked = rerank(question, fused, top_k=RERANK_KEEP_TOP)
    return [r["chunk_id"] for r in reranked]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test-scoring":
        # ============================================================
        # NO retrieval pipeline needed — proves hit_at_k / reciprocal_rank
        # math is correct using a fake retrieve_fn with known right/wrong answers.
        # ============================================================
        fake_dataset = [
            {"question": "q1", "workspace_id": "w", "expected_chunk_ids": ["c1"]},
            {"question": "q2", "workspace_id": "w", "expected_chunk_ids": ["c5"]},
            {"question": "q3", "workspace_id": "w", "expected_chunk_ids": ["c9"]},
        ]

        def fake_retrieve(question, workspace_id):
            fake_results = {
                "q1": ["c1", "c2", "c3"],       # correct answer at rank 1 -> hit@1
                "q2": ["c2", "c3", "c5"],       # correct answer at rank 3 -> hit@3 but not hit@1
                "q3": ["c2", "c3", "c4"],       # correct answer never appears -> miss entirely
            }
            return fake_results[question]

        results = evaluate_retrieval(fake_dataset, fake_retrieve, k_values=[1, 3, 5])
        print(json.dumps(results["aggregate"], indent=2))

        # q1 hits at 1,3,5 / q2 misses at 1 but hits at 3,5 / q3 always misses
        assert results["aggregate"]["hit_rate@1"] == 1/3   # only q1
        assert results["aggregate"]["hit_rate@3"] == 2/3   # q1 and q2
        assert results["aggregate"]["hit_rate@5"] == 2/3   # still just q1 and q2, q3 never hits
        print("\nALL ASSERTIONS PASSED — scoring math is correct")

    else:
        dataset = json.load(open(Path(__file__).parent / "dataset.json"))
        results = evaluate_retrieval(dataset, real_retrieve_fn)

        print("=== Per-question results ===\n")
        for r in results["per_question"]:
            status = "HIT" if r["hit@5"] else "MISS"
            print(f"[{status}] ({r['category']}) {r['question']}")
            print(f"  expected: {r['expected']}")
            print(f"  got top5: {r['retrieved_top5']}\n")

        print("=== Aggregate (all questions) ===")
        print(json.dumps(results["aggregate"], indent=2))

        print("\n=== By category (this is the real signal — direct vs paraphrased) ===")
        for cat, agg in results["by_category"].items():
            print(f"\n{cat} (n={agg['n']}):")
            print(json.dumps(agg, indent=2))

        out_path = Path(__file__).parent / "results.json"
        json.dump(results, open(out_path, "w"), indent=2)
        print(f"\nSaved to {out_path}")