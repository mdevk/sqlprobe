"""SQL Judge comparator: the core differentiator.

Three-level comparison:
  1. Execute both SQLs, compare result DataFrames.
  2. If results differ, parse ASTs with sqlglot and classify WHY.
  3. Determine semantic equivalence for correct-but-different SQL.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp

from src.judge.taxonomy import EvalResult, FailureType

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "tpcds.db"

FLOAT_ATOL = 0.01


# ── Public API ───────────────────────────────────────────────────────────────

def judge(
    query_id: int,
    nl: str,
    difficulty: str,
    ground_truth_sql: str,
    generated_sql: str,
    backend: str,
    db_path: Path = DB_PATH,
) -> EvalResult:
    """Run the full judge pipeline on a single query pair."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return _judge_impl(
            con, query_id, nl, difficulty,
            ground_truth_sql, generated_sql, backend,
        )
    finally:
        con.close()


def judge_batch(
    pairs: list[dict],
    db_path: Path = DB_PATH,
) -> list[EvalResult]:
    """Judge multiple pairs sharing a single DB connection."""
    con = duckdb.connect(str(db_path), read_only=True)
    results = []
    try:
        for p in pairs:
            result = _judge_impl(
                con,
                query_id=p["query_id"],
                nl=p["nl"],
                difficulty=p["difficulty"],
                ground_truth_sql=p["ground_truth_sql"],
                generated_sql=p["generated_sql"],
                backend=p["backend"],
            )
            results.append(result)
    finally:
        con.close()
    return results


# ── Internal implementation ──────────────────────────────────────────────────

def _judge_impl(
    con: duckdb.DuckDBPyConnection,
    query_id: int,
    nl: str,
    difficulty: str,
    ground_truth_sql: str,
    generated_sql: str,
    backend: str,
) -> EvalResult:
    """Core judge logic: execute → compare → classify."""

    base = dict(
        query_id=query_id,
        nl=nl,
        difficulty=difficulty,
        ground_truth_sql=ground_truth_sql,
        generated_sql=generated_sql,
        backend=backend,
    )

    # ── Step 0: Execute ground truth ─────────────────────────────────────
    gt_df, gt_err = _execute_sql(con, ground_truth_sql)
    if gt_err:
        return EvalResult(
            **base,
            is_correct=False,
            failure_type=FailureType.RUNTIME_ERROR,
            failure_details=f"Ground truth SQL failed: {gt_err}",
            execution_error=gt_err,
        )

    # ── Step 1: Execute generated SQL ────────────────────────────────────
    gen_df, gen_err = _execute_sql(con, generated_sql)
    if gen_err:
        # Distinguish syntax errors from runtime errors
        ft = FailureType.SYNTAX_ERROR if _is_syntax_error(gen_err) else FailureType.RUNTIME_ERROR
        return EvalResult(
            **base,
            is_correct=False,
            failure_type=ft,
            failure_details=gen_err,
            execution_error=gen_err,
            ground_truth_rows=len(gt_df),
        )

    # ── Step 2: DataFrame comparison ─────────────────────────────────────
    match_result = _compare_dataframes(gt_df, gen_df)

    if match_result == "exact":
        return EvalResult(
            **base,
            is_correct=True,
            failure_type=FailureType.CORRECT,
            ground_truth_rows=len(gt_df),
            generated_rows=len(gen_df),
        )

    if match_result == "equivalent":
        # Results match but SQL is different
        return EvalResult(
            **base,
            is_correct=True,
            failure_type=FailureType.CORRECT_DIFFERENT,
            failure_details="Same results, different SQL structure",
            ground_truth_rows=len(gt_df),
            generated_rows=len(gen_df),
        )

    # ── Step 3: Results differ — classify via AST ────────────────────────
    ast_diff = _compute_ast_diff(ground_truth_sql, generated_sql)
    failure_type = _classify_failure(ast_diff, gt_df, gen_df)

    details_parts = []
    if failure_type == FailureType.PARTIAL_MATCH:
        overlap = _row_overlap_pct(gt_df, gen_df)
        details_parts.append(f"Row overlap: {overlap:.1f}%")
    for category, diff in ast_diff.items():
        if diff["added"] or diff["removed"]:
            details_parts.append(
                f"{category}: +{diff['added']} -{diff['removed']}"
            )

    return EvalResult(
        **base,
        is_correct=False,
        failure_type=failure_type,
        failure_details="; ".join(details_parts) if details_parts else "Results differ",
        ast_diff=ast_diff,
        ground_truth_rows=len(gt_df),
        generated_rows=len(gen_df),
    )


# ── SQL execution ────────────────────────────────────────────────────────────

def _execute_sql(
    con: duckdb.DuckDBPyConnection, sql: str,
) -> tuple[pd.DataFrame, str]:
    """Execute SQL and return (dataframe, error_string)."""
    try:
        result = con.execute(sql)
        df = result.fetchdf()
        return df, ""
    except duckdb.ParserException as e:
        return pd.DataFrame(), f"SYNTAX: {e}"
    except Exception as e:
        return pd.DataFrame(), str(e)


def _is_syntax_error(err: str) -> bool:
    return err.startswith("SYNTAX:") or "Parser Error" in err


# ── DataFrame comparison ─────────────────────────────────────────────────────

def _compare_dataframes(gt_df: pd.DataFrame, gen_df: pd.DataFrame) -> str:
    """Compare two DataFrames ignoring column names and row order.

    Returns:
        "exact"      — identical values
        "equivalent" — same shape + values within tolerance
        "different"  — results don't match
    """
    if gt_df.empty and gen_df.empty:
        return "exact"

    # Normalize: strip column names, sort rows, reset index
    gt_norm = _normalize_df(gt_df)
    gen_norm = _normalize_df(gen_df)

    # Quick shape check
    if gt_norm.shape != gen_norm.shape:
        return "different"

    # Try exact comparison first (fast path)
    try:
        pd.testing.assert_frame_equal(
            gt_norm, gen_norm,
            check_names=False,
            check_dtype=False,
            check_exact=True,
        )
        return "exact"
    except AssertionError:
        pass

    # Try with float tolerance
    try:
        pd.testing.assert_frame_equal(
            gt_norm, gen_norm,
            check_names=False,
            check_dtype=False,
            atol=FLOAT_ATOL,
            rtol=0,
        )
        return "equivalent"
    except AssertionError:
        return "different"


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a DataFrame for comparison: rename columns, sort, reset index."""
    # Rename columns to positional integers (ignore name differences)
    normalized = df.copy()
    normalized.columns = range(len(normalized.columns))

    # Convert all columns to comparable types
    for col in normalized.columns:
        # Coerce numeric-looking columns to float for tolerance comparison
        try:
            normalized[col] = pd.to_numeric(normalized[col], errors="ignore")
        except (TypeError, ValueError):
            pass

    # Sort by all columns to make row order irrelevant
    sort_cols = list(normalized.columns)
    try:
        normalized = normalized.sort_values(by=sort_cols, ignore_index=True)
    except TypeError:
        # If columns aren't sortable (mixed types), convert to string first
        for col in sort_cols:
            normalized[col] = normalized[col].astype(str)
        normalized = normalized.sort_values(by=sort_cols, ignore_index=True)

    return normalized.reset_index(drop=True)


def _row_overlap_pct(gt_df: pd.DataFrame, gen_df: pd.DataFrame) -> float:
    """Compute what percentage of ground-truth rows appear in generated results."""
    if gt_df.empty:
        return 100.0 if gen_df.empty else 0.0

    gt_norm = _normalize_df(gt_df)
    gen_norm = _normalize_df(gen_df)

    # Convert rows to tuples for set intersection
    gt_rows = set(gt_norm.itertuples(index=False, name=None))
    gen_rows = set(gen_norm.itertuples(index=False, name=None))

    if not gt_rows:
        return 0.0
    overlap = len(gt_rows & gen_rows)
    return (overlap / len(gt_rows)) * 100


# ── AST comparison via sqlglot ───────────────────────────────────────────────

def _parse_sql(sql: str) -> exp.Expression | None:
    """Parse SQL with sqlglot, return None on failure."""
    try:
        return sqlglot.parse_one(sql, dialect="duckdb")
    except sqlglot.errors.ParseError:
        return None


def _extract_tables(tree: exp.Expression) -> set[str]:
    """Extract normalized table names from a query AST."""
    tables = set()
    for node in tree.find_all(exp.Table):
        name = node.name.lower()
        if name:
            tables.add(name)
    return tables


def _extract_joins(tree: exp.Expression) -> list[str]:
    """Extract join representations (table + ON condition)."""
    joins = []
    for node in tree.find_all(exp.Join):
        # Capture the join target table and kind
        table_node = node.find(exp.Table)
        table_name = table_node.name.lower() if table_node else "?"
        kind = node.kind or "JOIN"
        joins.append(f"{kind} {table_name}")
    return sorted(joins)


def _extract_columns(tree: exp.Expression) -> set[str]:
    """Extract column references (table.column or just column)."""
    columns = set()
    for node in tree.find_all(exp.Column):
        col_name = node.name.lower()
        table = node.table.lower() if node.table else ""
        if table:
            columns.add(f"{table}.{col_name}")
        else:
            columns.add(col_name)
    return columns


def _extract_aggregations(tree: exp.Expression) -> list[str]:
    """Extract aggregate function calls."""
    aggs = []
    for node in tree.find_all(exp.AggFunc):
        aggs.append(type(node).__name__.upper())
    return sorted(aggs)


def _extract_where(tree: exp.Expression) -> str:
    """Extract normalized WHERE clause as string."""
    where = tree.find(exp.Where)
    return where.sql(dialect="duckdb").lower() if where else ""


def _extract_order(tree: exp.Expression) -> list[str]:
    """Extract ORDER BY expressions."""
    orders = []
    order_node = tree.find(exp.Order)
    if order_node:
        for ordered in order_node.find_all(exp.Ordered):
            orders.append(ordered.sql(dialect="duckdb").lower())
    return orders


def _compute_ast_diff(
    ground_truth_sql: str, generated_sql: str,
) -> dict:
    """Parse both SQLs and compute per-category diffs.

    Returns a dict like:
    {
        "tables":       {"added": [...], "removed": [...]},
        "joins":        {"added": [...], "removed": [...]},
        "columns":      {"added": [...], "removed": [...]},
        "aggregations": {"added": [...], "removed": [...]},
        "where":        {"added": [...], "removed": [...]},
        "order":        {"added": [...], "removed": [...]},
    }
    """
    gt_tree = _parse_sql(ground_truth_sql)
    gen_tree = _parse_sql(generated_sql)

    if gt_tree is None or gen_tree is None:
        # Can't parse — return empty diff
        return {
            cat: {"added": [], "removed": []}
            for cat in ("tables", "joins", "columns", "aggregations", "where", "order")
        }

    gt_tables = _extract_tables(gt_tree)
    gen_tables = _extract_tables(gen_tree)

    gt_joins = _extract_joins(gt_tree)
    gen_joins = _extract_joins(gen_tree)

    gt_cols = _extract_columns(gt_tree)
    gen_cols = _extract_columns(gen_tree)

    gt_aggs = _extract_aggregations(gt_tree)
    gen_aggs = _extract_aggregations(gen_tree)

    gt_where = _extract_where(gt_tree)
    gen_where = _extract_where(gen_tree)

    gt_order = _extract_order(gt_tree)
    gen_order = _extract_order(gen_tree)

    return {
        "tables": {
            "added": sorted(gen_tables - gt_tables),
            "removed": sorted(gt_tables - gen_tables),
        },
        "joins": {
            "added": sorted(set(gen_joins) - set(gt_joins)),
            "removed": sorted(set(gt_joins) - set(gen_joins)),
        },
        "columns": {
            "added": sorted(gen_cols - gt_cols),
            "removed": sorted(gt_cols - gen_cols),
        },
        "aggregations": {
            "added": sorted(set(gen_aggs) - set(gt_aggs)),
            "removed": sorted(set(gt_aggs) - set(gen_aggs)),
        },
        "where": {
            "added": [gen_where] if gen_where and gen_where != gt_where else [],
            "removed": [gt_where] if gt_where and gen_where != gt_where else [],
        },
        "order": {
            "added": sorted(set(gen_order) - set(gt_order)),
            "removed": sorted(set(gt_order) - set(gen_order)),
        },
    }


def _classify_failure(
    ast_diff: dict,
    gt_df: pd.DataFrame,
    gen_df: pd.DataFrame,
) -> FailureType:
    """Determine the primary failure type from AST diff and result shapes.

    Priority order matches the most impactful structural errors first.
    """
    # Check for partial match first (results partially overlap)
    overlap = _row_overlap_pct(gt_df, gen_df)
    is_partial = 0 < overlap < 100

    # Count diffs per category
    table_diffs = len(ast_diff["tables"]["added"]) + len(ast_diff["tables"]["removed"])
    join_diffs = len(ast_diff["joins"]["added"]) + len(ast_diff["joins"]["removed"])
    col_diffs = len(ast_diff["columns"]["added"]) + len(ast_diff["columns"]["removed"])
    agg_diffs = len(ast_diff["aggregations"]["added"]) + len(ast_diff["aggregations"]["removed"])
    where_diffs = len(ast_diff["where"]["added"]) + len(ast_diff["where"]["removed"])
    order_diffs = len(ast_diff["order"]["added"]) + len(ast_diff["order"]["removed"])

    # Classify by the most significant structural difference
    # (priority: table > join > column > aggregation > filter > order)
    if table_diffs > 0:
        return FailureType.WRONG_TABLE
    if join_diffs > 0:
        return FailureType.WRONG_JOIN
    if agg_diffs > 0:
        return FailureType.WRONG_AGGREGATION
    if col_diffs > 0:
        return FailureType.WRONG_COLUMN
    if where_diffs > 0:
        return FailureType.WRONG_FILTER

    # Only row-order issues with identical data sets (same shape, different order)
    if order_diffs > 0 and gt_df.shape == gen_df.shape:
        return FailureType.WRONG_ORDER

    # Partial overlap but no clear AST difference identified
    if is_partial:
        return FailureType.PARTIAL_MATCH

    # Catch-all: results differ but no AST diff found (aliasing, expressions, etc.)
    return FailureType.WRONG_FILTER
