"""Scorer: load eval_results.json, run the judge, produce a scorecard.

Reads data/eval_results.json (produced by the harness runner), runs the SQL
judge on every query × backend pair, aggregates into a scorecard, and saves
to data/scored_results.json.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from src.judge.comparator import judge_batch
from src.judge.taxonomy import EvalResult, ScoreCard

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
EVAL_RESULTS_PATH = ROOT / "data" / "eval_results.json"
SCORED_OUTPUT_PATH = ROOT / "data" / "scored_results.json"


# ── Scorecard computation ────────────────────────────────────────────────────

def compute_scorecard(results: list[EvalResult]) -> ScoreCard:
    """Aggregate results into a scorecard."""
    total = len(results)
    correct = sum(1 for r in results if r.is_correct)

    # Per-backend accuracy
    by_backend: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_backend[r.backend].append(r.is_correct)
    accuracy_by_backend = {
        b: sum(v) / len(v) if v else 0.0
        for b, v in sorted(by_backend.items())
    }

    # Per-difficulty accuracy
    by_difficulty: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_difficulty[r.difficulty].append(r.is_correct)
    accuracy_by_difficulty = {
        d: sum(v) / len(v) if v else 0.0
        for d, v in sorted(by_difficulty.items())
    }

    # Failure distribution per backend
    failure_dist: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        failure_dist[r.backend][r.failure_type.value] += 1

    return ScoreCard(
        total_queries=total,
        total_correct=correct,
        overall_accuracy=correct / total if total else 0.0,
        accuracy_by_backend=accuracy_by_backend,
        accuracy_by_difficulty=accuracy_by_difficulty,
        failure_distribution={b: dict(d) for b, d in failure_dist.items()},
        results=results,
    )


def print_scorecard(sc: ScoreCard) -> None:
    """Pretty-print the scorecard to stdout."""
    print(f"\n{'='*70}")
    print("SQLPROBE SCORECARD")
    print(f"{'='*70}")
    print(f"Total queries judged:  {sc.total_queries}")
    print(f"Total correct:         {sc.total_correct}")
    print(f"Overall accuracy:      {sc.overall_accuracy:.1%}")

    print(f"\n{'─'*70}")
    print("ACCURACY BY BACKEND")
    print(f"{'─'*70}")
    for backend, acc in sc.accuracy_by_backend.items():
        count = sum(1 for r in sc.results if r.backend == backend)
        correct = sum(1 for r in sc.results if r.backend == backend and r.is_correct)
        print(f"  {backend:20s}  {acc:6.1%}  ({correct}/{count})")

    print(f"\n{'─'*70}")
    print("ACCURACY BY DIFFICULTY")
    print(f"{'─'*70}")
    for diff, acc in sc.accuracy_by_difficulty.items():
        count = sum(1 for r in sc.results if r.difficulty == diff)
        correct = sum(1 for r in sc.results if r.difficulty == diff and r.is_correct)
        print(f"  {diff:20s}  {acc:6.1%}  ({correct}/{count})")

    print(f"\n{'─'*70}")
    print("FAILURE TYPE DISTRIBUTION")
    print(f"{'─'*70}")
    for backend, dist in sc.failure_distribution.items():
        print(f"\n  [{backend}]")
        for ft, count in sorted(dist.items(), key=lambda x: -x[1]):
            print(f"    {ft:25s}  {count:3d}")

    failures = [r for r in sc.results if not r.is_correct]
    if failures:
        print(f"\n{'─'*70}")
        print(f"FAILURE DETAILS ({len(failures)} failures)")
        print(f"{'─'*70}")
        for r in failures:
            print(f"\n  [{r.query_id}] {r.backend} / {r.difficulty}")
            print(f"    NL:      {r.nl[:80]}...")
            print(f"    Type:    {r.failure_type.value}")
            print(f"    Details: {r.failure_details[:120]}")
            if r.execution_error:
                print(f"    Error:   {r.execution_error[:120]}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    if not EVAL_RESULTS_PATH.exists():
        print(f"ERROR: {EVAL_RESULTS_PATH} not found.")
        print("Run the harness first:  python -m src.harness.runner")
        return

    raw_results = json.loads(EVAL_RESULTS_PATH.read_text())
    print(f"Loaded {len(raw_results)} results from {EVAL_RESULTS_PATH}")

    pairs = []
    for r in raw_results:
        pairs.append({
            "query_id": r["query_id"],
            "nl": r["nl"],
            "difficulty": r["difficulty"],
            "ground_truth_sql": r["ground_truth_sql"],
            "generated_sql": r["generated_sql"],
            "backend": r["backend"],
        })

    # Run the judge
    print(f"Judging {len(pairs)} query pairs...")
    results = judge_batch(pairs)

    # Compute scorecard
    scorecard = compute_scorecard(results)
    print_scorecard(scorecard)

    # Save
    output = {
        "total_queries": scorecard.total_queries,
        "total_correct": scorecard.total_correct,
        "overall_accuracy": scorecard.overall_accuracy,
        "accuracy_by_backend": scorecard.accuracy_by_backend,
        "accuracy_by_difficulty": scorecard.accuracy_by_difficulty,
        "failure_distribution": scorecard.failure_distribution,
        "results": [r.model_dump() for r in scorecard.results],
    }
    for r in output["results"]:
        if not isinstance(r["failure_type"], str):
            r["failure_type"] = r["failure_type"].value

    SCORED_OUTPUT_PATH.write_text(json.dumps(output, indent=2, default=str) + "\n")
    print(f"\nSaved scored results to {SCORED_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
