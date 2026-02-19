"""Generate NL+SQL evaluation pairs across 4 difficulty tiers using an LLM.

Reads the schema profile, asks Claude Sonnet to produce questions + ground-truth
SQL, validates every query by executing against DuckDB, retries failures.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import duckdb
import litellm
from dotenv import load_dotenv
from pydantic import BaseModel

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "tpcds.db"
SCHEMA_PATH = ROOT / "data" / "schema_profile.md"
OUTPUT_PATH = ROOT / "data" / "eval_set.json"

# ── Model config ─────────────────────────────────────────────────────────────
MODEL = "anthropic/claude-sonnet-4-20250514"
TEMPERATURE = 0.7

# ── Difficulty tiers ─────────────────────────────────────────────────────────
TIERS: dict[str, dict[str, str | int]] = {
    "easy": {
        "count": 5,
        "description": (
            "Single-table queries. Simple SELECT with WHERE, COUNT, SUM, AVG, "
            "MIN, MAX, GROUP BY, or ORDER BY. No joins."
        ),
    },
    "medium": {
        "count": 5,
        "description": (
            "2-3 table joins. INNER JOIN or LEFT JOIN with filtering and "
            "aggregation. May use HAVING or simple CASE expressions."
        ),
    },
    "hard": {
        "count": 5,
        "description": (
            "Subqueries and window functions. Correlated subqueries, EXISTS/IN "
            "with subselects, ROW_NUMBER/RANK/DENSE_RANK, LAG/LEAD, or "
            "PARTITION BY. 2-4 tables."
        ),
    },
    "very_hard": {
        "count": 5,
        "description": (
            "CTEs with complex aggregation. Multiple WITH clauses, nested CTEs, "
            "multi-level aggregation, UNION/INTERSECT/EXCEPT, complex CASE "
            "logic, or date arithmetic. 3+ tables."
        ),
    },
}


class EvalQuery(BaseModel):
    nl: str
    sql: str
    difficulty: str


# ── Prompt construction ──────────────────────────────────────────────────────

def _build_prompt(
    schema_md: str,
    tier_name: str,
    tier_desc: str,
    count: int,
    existing: list[str],
) -> str:
    """Build the LLM prompt for one tier of query generation."""
    avoid_block = ""
    if existing:
        avoid_block = (
            "\n\nAVOID generating questions similar to these already-generated ones:\n"
            + "\n".join(f"- {q}" for q in existing)
        )

    return f"""You are a SQL evaluation expert. Given the database schema below, generate exactly {count} natural language questions with corresponding ground-truth SQL queries.

DIFFICULTY TIER: {tier_name}
{tier_desc}

RULES:
1. SQL must be valid DuckDB SQL (PostgreSQL-compatible dialect).
2. Every query must be a SELECT statement (no DDL/DML).
3. Use ONLY tables and columns that exist in the schema below.
4. Each question should test a DIFFERENT aspect of SQL knowledge.
5. Questions must be phrased as a business user would ask them — no SQL jargon.
6. SQL must return at least 1 row when run against TPC-H scale factor 1 data.
7. Prefer concrete filters (specific dates, amounts, names) that will match TPC-H data.
   - Valid date range: 1992-01-01 to 1998-12-31
   - Nation names: uppercase (e.g., 'FRANCE', 'GERMANY', 'UNITED STATES')
   - Region names: uppercase (e.g., 'EUROPE', 'ASIA', 'AMERICA')
   - Market segments: 'AUTOMOBILE', 'BUILDING', 'FURNITURE', 'HOUSEHOLD', 'MACHINERY'
8. Do NOT use LIMIT unless the question specifically asks for "top N".
{avoid_block}

SCHEMA:
{schema_md}

Respond with EXACTLY {count} entries in this JSON format (no other text):
```json
[
  {{
    "nl": "What is the average account balance of customers in the AUTOMOBILE segment?",
    "sql": "SELECT AVG(c_acctbal) AS avg_balance FROM customer WHERE c_mktsegment = 'AUTOMOBILE'"
  }}
]
```"""


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    """Call the LLM via litellm and return the text response."""
    response = litellm.completion(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def _parse_response(raw: str) -> list[dict]:
    """Extract a JSON array from the LLM response text."""
    # Try to find JSON between ```json ... ``` fences first
    fence_match = re.search(r"```json\s*\n?(.*?)```", raw, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        # Fall back to finding the outermost [ ... ]
        bracket_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if bracket_match:
            text = bracket_match.group(0)
        else:
            raise ValueError("No JSON array found in LLM response")
    return json.loads(text)


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_sql(sql: str, con: duckdb.DuckDBPyConnection) -> tuple[bool, str]:
    """Execute SQL against DuckDB. Return (success, error_message)."""
    try:
        result = con.execute(sql).fetchall()
        if len(result) == 0:
            return False, "Query returned 0 rows"
        return True, ""
    except Exception as e:
        return False, str(e)


# ── Main generation loop ─────────────────────────────────────────────────────

def generate_eval_set(
    max_retries_per_tier: int = 3,
) -> list[EvalQuery]:
    """Generate 20 validated NL+SQL pairs across 4 difficulty tiers."""
    schema_md = SCHEMA_PATH.read_text()
    con = duckdb.connect(str(DB_PATH), read_only=True)

    all_queries: list[EvalQuery] = []
    all_nl: list[str] = []  # track NL questions for dedup

    try:
        for tier_name, tier_cfg in TIERS.items():
            target = tier_cfg["count"]
            desc = tier_cfg["description"]
            tier_queries: list[EvalQuery] = []

            print(f"\n{'='*60}")
            print(f"Tier: {tier_name.upper()} — need {target} valid queries")
            print(f"{'='*60}")

            for attempt in range(1, max_retries_per_tier + 1):
                needed = target - len(tier_queries)
                if needed <= 0:
                    break

                print(f"\n  Attempt {attempt}: requesting {needed} queries...")
                prompt = _build_prompt(
                    schema_md, tier_name, desc, needed, all_nl,
                )

                try:
                    raw = _call_llm(prompt)
                except Exception as e:
                    print(f"  LLM call failed: {e}")
                    continue

                try:
                    items = _parse_response(raw)
                except (ValueError, json.JSONDecodeError) as e:
                    print(f"  Parse failed: {e}")
                    continue

                for item in items:
                    if len(tier_queries) >= target:
                        break

                    nl = item.get("nl", "").strip()
                    sql = item.get("sql", "").strip()
                    if not nl or not sql:
                        print(f"    SKIP: empty nl/sql")
                        continue

                    # Ensure it ends with semicolon-free SQL (DuckDB is fine either way)
                    sql = sql.rstrip(";")

                    ok, err = _validate_sql(sql, con)
                    if ok:
                        eq = EvalQuery(nl=nl, sql=sql, difficulty=tier_name)
                        tier_queries.append(eq)
                        all_nl.append(nl)
                        print(f"    ✓ [{len(tier_queries)}/{target}] {nl[:70]}...")
                    else:
                        print(f"    ✗ FAILED: {err}")
                        print(f"      SQL: {sql[:100]}...")

            if len(tier_queries) < target:
                print(
                    f"\n  WARNING: Only got {len(tier_queries)}/{target} "
                    f"valid queries for tier '{tier_name}'"
                )

            all_queries.extend(tier_queries)

    finally:
        con.close()

    return all_queries


def save_eval_set(queries: list[EvalQuery], path: Path) -> None:
    """Write the eval set to JSON."""
    data = [q.model_dump() for q in queries]
    path.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    load_dotenv(ROOT / ".env", override=True)

    print("Generating evaluation query set...")
    print(f"  Model: {MODEL}")
    print(f"  Database: {DB_PATH}")
    print(f"  Schema: {SCHEMA_PATH}")

    queries = generate_eval_set()

    save_eval_set(queries, OUTPUT_PATH)

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE — {len(queries)} validated queries saved to {OUTPUT_PATH}")
    print(f"{'='*60}")

    tier_counts = {}
    for q in queries:
        tier_counts[q.difficulty] = tier_counts.get(q.difficulty, 0) + 1
    for tier, count in sorted(tier_counts.items()):
        print(f"  {tier}: {count}")

    # Print full eval set
    print(f"\n{'='*60}")
    print("FULL EVAL SET")
    print(f"{'='*60}")
    for i, q in enumerate(queries, 1):
        print(f"\n--- [{i}] {q.difficulty.upper()} ---")
        print(f"NL:  {q.nl}")
        print(f"SQL: {q.sql}")


if __name__ == "__main__":
    main()
