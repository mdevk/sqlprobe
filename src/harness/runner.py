"""Eval harness: send each NL question to 3 LLM backends, collect generated SQL.

For each of the 20 queries in eval_set.json × 3 backends (60 total calls):
  1. Build a prompt with the TPC-H DDL + natural language question.
  2. Call the LLM via litellm, asking for SQL only.
  3. Extract the SQL from the response.
  4. Save {query_id, nl, difficulty, ground_truth_sql, generated_sql, backend}
     to data/eval_results.json.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import duckdb
import litellm
from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "tpcds.db"
EVAL_SET_PATH = ROOT / "data" / "eval_set.json"
OUTPUT_PATH = ROOT / "data" / "eval_results.json"

# ── Backend definitions ──────────────────────────────────────────────────────
BACKENDS: dict[str, str] = {
    "gpt-4o": "gpt-4o",
    "claude-sonnet": "anthropic/claude-sonnet-4-20250514",
    "gemini-flash": "gemini/gemini-2.0-flash-lite",
}

SLEEP_BETWEEN_CALLS = 2   # seconds, between calls
MAX_RETRIES = 3           # retries per LLM call on rate-limit errors
INITIAL_BACKOFF = 5       # seconds, first retry wait (doubles each retry)


# ── Schema DDL extraction ────────────────────────────────────────────────────

def _get_schema_ddl(db_path: Path) -> str:
    """Extract CREATE TABLE DDL for every table in the database."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = con.execute("SHOW TABLES").fetchall()
        ddl_parts: list[str] = []
        for (table_name,) in tables:
            row = con.execute(
                f"SELECT sql FROM duckdb_tables() WHERE table_name = '{table_name}'"
            ).fetchone()
            if row:
                ddl_parts.append(row[0] + ";")
        return "\n\n".join(ddl_parts)
    finally:
        con.close()


# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a SQL expert. You are given a database schema (DuckDB / PostgreSQL dialect) \
and a natural language question. Return ONLY a single SQL SELECT statement that \
answers the question. No explanation, no markdown fences, no comments — just the \
raw SQL."""


def _build_user_prompt(schema_ddl: str, nl_question: str) -> str:
    return f"""\
DATABASE SCHEMA:
{schema_ddl}

QUESTION:
{nl_question}

SQL:"""


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_backend(model: str, schema_ddl: str, nl_question: str) -> str:
    """Call a single LLM backend with retry + exponential backoff."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = litellm.completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": _build_user_prompt(schema_ddl, nl_question)},
                ],
                temperature=0.0,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            last_err = e
            is_rate_limit = "RateLimitError" in type(e).__name__ or "rate" in str(e).lower()
            if is_rate_limit and attempt < MAX_RETRIES:
                wait = INITIAL_BACKOFF * (2 ** attempt)
                print(f"RATE LIMITED (retry {attempt + 1}/{MAX_RETRIES} in {wait}s)...", end=" ", flush=True)
                time.sleep(wait)
            else:
                raise
    raise last_err  # type: ignore[misc]


def _extract_sql(raw: str) -> str:
    """Extract SQL from LLM response, stripping markdown fences if present."""
    # Try to find SQL inside ```sql ... ``` fences
    fence_match = re.search(r"```(?:sql)?\s*\n?(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if fence_match:
        sql = fence_match.group(1).strip()
    else:
        sql = raw.strip()

    # Remove any trailing semicolons (DuckDB is fine either way)
    sql = sql.rstrip(";").strip()

    # If the response has multiple lines starting with explanation, take only
    # the part that starts with SELECT/WITH
    lines = sql.split("\n")
    sql_start = None
    for i, line in enumerate(lines):
        stripped = line.strip().upper()
        if stripped.startswith(("SELECT", "WITH")):
            sql_start = i
            break

    if sql_start is not None and sql_start > 0:
        sql = "\n".join(lines[sql_start:])

    return sql.rstrip(";").strip()


# ── Main runner ──────────────────────────────────────────────────────────────

def run_eval() -> list[dict]:
    """Run all 20 queries × 3 backends and return the results list."""
    eval_set = json.loads(EVAL_SET_PATH.read_text())
    schema_ddl = _get_schema_ddl(DB_PATH)

    print(f"Loaded {len(eval_set)} queries from {EVAL_SET_PATH}")
    print(f"Backends: {list(BACKENDS.keys())}")
    print(f"Total LLM calls: {len(eval_set) * len(BACKENDS)}")
    print(f"Schema DDL length: {len(schema_ddl)} chars")
    print()

    results: list[dict] = []
    total_calls = len(eval_set) * len(BACKENDS)
    call_num = 0

    for query_idx, query in enumerate(eval_set):
        nl = query["nl"]
        gt_sql = query["sql"]
        difficulty = query["difficulty"]

        print(f"\n{'─'*60}")
        print(f"[{query_idx + 1}/{len(eval_set)}] ({difficulty}) {nl[:70]}...")

        for backend_name, model_id in BACKENDS.items():
            call_num += 1
            print(f"  [{call_num}/{total_calls}] {backend_name}...", end=" ", flush=True)

            generated_sql = ""
            error = ""
            try:
                raw = _call_backend(model_id, schema_ddl, nl)
                generated_sql = _extract_sql(raw)
                print(f"OK ({len(generated_sql)} chars)")
            except Exception as e:
                error = str(e)
                generated_sql = f"-- ERROR: {error}"
                print(f"FAILED: {error[:80]}")

            results.append({
                "query_id": query_idx,
                "nl": nl,
                "difficulty": difficulty,
                "ground_truth_sql": gt_sql,
                "generated_sql": generated_sql,
                "backend": backend_name,
            })

            # Rate-limit sleep (skip after the very last call)
            if call_num < total_calls:
                time.sleep(SLEEP_BETWEEN_CALLS)

    return results


def main() -> None:
    load_dotenv(ROOT / ".env", override=True)

    print("="*60)
    print("SQLProbe Eval Harness — Real LLM Backends")
    print("="*60)

    results = run_eval()

    # Save raw results
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\n{'='*60}")
    print(f"Saved {len(results)} results to {OUTPUT_PATH}")
    print(f"{'='*60}")

    # Quick summary
    from collections import Counter
    by_backend = Counter(r["backend"] for r in results)
    for backend, count in sorted(by_backend.items()):
        print(f"  {backend}: {count} calls")


if __name__ == "__main__":
    main()
