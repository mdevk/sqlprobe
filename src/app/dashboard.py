"""SQLProbe Evaluation Dashboard.

Streamlit app that visualises scored_results.json:
  1. Metric row — per-backend overall accuracy as big numbers
  2. Scorecard table — accuracy % by backend x difficulty tier
  3. Stacked bar chart — failure-type distribution by backend
  4. Drill-down — select any query, see side-by-side comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SCORED_PATH = ROOT / "data" / "scored_results.json"

# ── Difficulty display order ─────────────────────────────────────────────────
DIFFICULTY_ORDER = ["easy", "medium", "hard", "very_hard"]

# ── Failure types and colours ────────────────────────────────────────────────
FAILURE_COLORS = {
    "CORRECT": "#2ecc71",
    "CORRECT_DIFFERENT": "#27ae60",
    "PARTIAL_MATCH": "#f39c12",
    "WRONG_TABLE": "#e74c3c",
    "WRONG_JOIN": "#c0392b",
    "WRONG_COLUMN": "#e67e22",
    "WRONG_AGGREGATION": "#d35400",
    "WRONG_FILTER": "#8e44ad",
    "WRONG_ORDER": "#2980b9",
    "SYNTAX_ERROR": "#7f8c8d",
    "RUNTIME_ERROR": "#95a5a6",
}


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> dict:
    if not SCORED_PATH.exists():
        st.error(f"Scored results not found at `{SCORED_PATH}`.\n\nRun `python -m src.judge.scorer` first.")
        st.stop()
    return json.loads(SCORED_PATH.read_text())


def results_to_df(data: dict) -> pd.DataFrame:
    rows = []
    for r in data["results"]:
        rows.append({
            "query_id": r["query_id"],
            "nl": r["nl"],
            "difficulty": r["difficulty"],
            "ground_truth_sql": r["ground_truth_sql"],
            "generated_sql": r["generated_sql"],
            "backend": r["backend"],
            "is_correct": r["is_correct"],
            "failure_type": r["failure_type"],
            "failure_details": r["failure_details"],
            "execution_error": r["execution_error"],
            "ground_truth_rows": r["ground_truth_rows"],
            "generated_rows": r["generated_rows"],
        })
    return pd.DataFrame(rows)


# ── Section 1: Metric row ───────────────────────────────────────────────────

def render_metrics(df: pd.DataFrame) -> None:
    backends = sorted(df["backend"].unique())
    cols = st.columns(len(backends))
    for col, backend in zip(cols, backends):
        subset = df[df["backend"] == backend]
        acc = subset["is_correct"].mean()
        total = len(subset)
        correct = int(subset["is_correct"].sum())
        col.metric(
            label=backend,
            value=f"{acc:.0%}",
            delta=f"{correct}/{total} correct",
        )


# ── Section 2: Scorecard table ──────────────────────────────────────────────

def render_scorecard(df: pd.DataFrame) -> None:
    st.subheader("Accuracy by Backend × Difficulty")

    backends = sorted(df["backend"].unique())

    # Build a table: rows = difficulty tiers + overall, cols = backends
    rows = []
    for diff in DIFFICULTY_ORDER:
        if diff not in df["difficulty"].values:
            continue
        row = {"Difficulty": diff}
        for backend in backends:
            subset = df[(df["backend"] == backend) & (df["difficulty"] == diff)]
            if len(subset) > 0:
                acc = subset["is_correct"].mean()
                correct = int(subset["is_correct"].sum())
                total = len(subset)
                row[backend] = f"{acc:.0%} ({correct}/{total})"
            else:
                row[backend] = "—"
        rows.append(row)

    # Overall row
    overall = {"Difficulty": "**OVERALL**"}
    for backend in backends:
        subset = df[df["backend"] == backend]
        acc = subset["is_correct"].mean()
        correct = int(subset["is_correct"].sum())
        total = len(subset)
        overall[backend] = f"{acc:.0%} ({correct}/{total})"
    rows.append(overall)

    table_df = pd.DataFrame(rows)
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )


# ── Section 3: Failure distribution chart ────────────────────────────────────

def render_failure_chart(df: pd.DataFrame) -> None:
    st.subheader("Failure Type Distribution by Backend")

    # Only look at non-correct results for the failure chart
    # but include CORRECT so the full picture is visible
    chart_data = (
        df.groupby(["backend", "failure_type"])
        .size()
        .reset_index(name="count")
    )

    # Determine all failure types in the data for colour mapping
    all_types = sorted(chart_data["failure_type"].unique(), key=lambda x: list(FAILURE_COLORS.keys()).index(x) if x in FAILURE_COLORS else 99)

    color_domain = all_types
    color_range = [FAILURE_COLORS.get(ft, "#bdc3c7") for ft in all_types]

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("backend:N", title="Backend", sort=sorted(df["backend"].unique())),
            y=alt.Y("count:Q", title="Query Count", stack="zero"),
            color=alt.Color(
                "failure_type:N",
                title="Failure Type",
                scale=alt.Scale(domain=color_domain, range=color_range),
                sort=all_types,
            ),
            tooltip=["backend", "failure_type", "count"],
        )
        .properties(height=400)
    )

    st.altair_chart(chart, use_container_width=True)


# ── Section 4: Query drill-down ──────────────────────────────────────────────

def render_drilldown(df: pd.DataFrame) -> None:
    st.subheader("Query Drill-Down")

    # Build a list of unique queries (by query_id + nl)
    queries = (
        df[["query_id", "nl", "difficulty"]]
        .drop_duplicates(subset=["query_id", "nl"])
        .sort_values("query_id")
    )

    options = [
        f"[{row.query_id}] ({row.difficulty}) {row.nl[:80]}..."
        for _, row in queries.iterrows()
    ]
    if not options:
        st.info("No queries to display.")
        return

    selected = st.selectbox("Select a query:", options, index=0)

    # Parse out the query_id
    selected_id = int(selected.split("]")[0].replace("[", ""))

    query_results = df[df["query_id"] == selected_id].sort_values("backend")

    if query_results.empty:
        st.warning("No results for this query.")
        return

    first = query_results.iloc[0]
    st.markdown(f"**Natural Language:** {first['nl']}")
    st.markdown(f"**Difficulty:** `{first['difficulty']}`")
    st.markdown("**Ground Truth SQL:**")
    st.code(first["ground_truth_sql"], language="sql")

    # Show each backend's result side-by-side
    backends = sorted(query_results["backend"].unique())
    cols = st.columns(len(backends))

    for col, backend in zip(cols, backends):
        row = query_results[query_results["backend"] == backend].iloc[0]
        with col:
            st.markdown(f"#### {backend}")

            if row["is_correct"]:
                st.success(f"✓ {row['failure_type']}")
            else:
                st.error(f"✗ {row['failure_type']}")

            st.markdown("**Generated SQL:**")
            st.code(row["generated_sql"], language="sql")

            if row["failure_details"]:
                st.markdown(f"**Details:** {row['failure_details']}")

            if row["execution_error"]:
                st.markdown(f"**Error:** `{row['execution_error'][:200]}`")

            st.caption(
                f"GT rows: {row['ground_truth_rows']} · "
                f"Gen rows: {row['generated_rows']}"
            )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="SQLProbe Dashboard",
        page_icon="🔍",
        layout="wide",
    )

    st.title("SQLProbe — Text-to-SQL Evaluation Dashboard")
    st.caption(
        "Comparing LLM-generated SQL against ground truth on real schemas. "
        "Academic benchmarks show 86-91% accuracy; enterprise-realistic drops to 21%."
    )

    data = load_data()
    df = results_to_df(data)

    # Section 1: Metric row
    render_metrics(df)

    st.divider()

    # Section 2: Scorecard table
    render_scorecard(df)

    st.divider()

    # Section 3: Failure chart
    render_failure_chart(df)

    st.divider()

    # Section 4: Drill-down
    render_drilldown(df)


if __name__ == "__main__":
    main()
