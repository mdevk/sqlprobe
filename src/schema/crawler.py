"""Schema crawler: reads INFORMATION_SCHEMA from a DuckDB database and returns a SchemaProfile."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import duckdb
from pydantic import BaseModel, Field


class ColumnInfo(BaseModel):
    name: str
    data_type: str
    is_nullable: bool
    is_primary_key: bool = False


class ForeignKey(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str


class TableProfile(BaseModel):
    name: str
    row_count: int
    columns: list[ColumnInfo]
    primary_key: list[str] = Field(default_factory=list)


class SchemaProfile(BaseModel):
    database_path: str
    table_count: int
    total_columns: int
    tables: list[TableProfile]
    foreign_keys: list[ForeignKey]
    naming_ambiguity_score: float = Field(
        description="0.0 (no ambiguity) to 1.0 (high ambiguity). "
        "Measures how many column base-names appear in multiple tables, "
        "making it harder for LLMs to pick the right table."
    )


def crawl_schema(db_path: Path, schema: str = "main") -> SchemaProfile:
    """Connect to a DuckDB database and extract its full schema profile."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return _build_profile(con, db_path, schema)
    finally:
        con.close()


def _build_profile(con: duckdb.DuckDBPyConnection, db_path: Path, schema: str) -> SchemaProfile:
    columns_by_table = _fetch_columns(con, schema)
    row_counts = _fetch_row_counts(con, columns_by_table.keys())
    primary_keys = _infer_primary_keys(columns_by_table)
    foreign_keys = _infer_foreign_keys(columns_by_table, primary_keys)

    tables: list[TableProfile] = []
    for table_name, cols in sorted(columns_by_table.items()):
        pk_cols = primary_keys.get(table_name, [])
        for col in cols:
            if col.name in pk_cols:
                col.is_primary_key = True
        tables.append(TableProfile(
            name=table_name,
            row_count=row_counts.get(table_name, 0),
            columns=cols,
            primary_key=pk_cols,
        ))

    total_columns = sum(len(t.columns) for t in tables)
    ambiguity = _compute_naming_ambiguity(columns_by_table)

    return SchemaProfile(
        database_path=str(db_path),
        table_count=len(tables),
        total_columns=total_columns,
        tables=tables,
        foreign_keys=foreign_keys,
        naming_ambiguity_score=round(ambiguity, 3),
    )


def _fetch_columns(
    con: duckdb.DuckDBPyConnection, schema: str
) -> dict[str, list[ColumnInfo]]:
    """Query INFORMATION_SCHEMA.columns and group by table."""
    rows = con.execute(
        """
        SELECT table_name, column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = ?
        ORDER BY table_name, ordinal_position
        """,
        [schema],
    ).fetchall()

    result: dict[str, list[ColumnInfo]] = defaultdict(list)
    for table_name, col_name, dtype, nullable in rows:
        result[table_name].append(ColumnInfo(
            name=col_name,
            data_type=dtype,
            is_nullable=nullable == "YES",
        ))
    return dict(result)


def _fetch_row_counts(
    con: duckdb.DuckDBPyConnection, table_names: object
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for table in table_names:
        row = con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
        counts[table] = row[0] if row else 0
    return counts


def _infer_primary_keys(
    columns_by_table: dict[str, list[ColumnInfo]],
) -> dict[str, list[str]]:
    """Heuristic: first column ending in 'key' whose base name contains a table-name substring.

    TPC-H convention: customer -> c_custkey, orders -> o_orderkey, part -> p_partkey.
    Junction tables like partsupp get composite PKs from their constituent FK columns.
    """
    # Known tables with composite PKs
    composite_pks: dict[str, list[str]] = {
        "partsupp": ["ps_partkey", "ps_suppkey"],
        "lineitem": ["l_orderkey", "l_linenumber"],
    }

    pks: dict[str, list[str]] = {}
    for table, cols in columns_by_table.items():
        if table in composite_pks:
            pks[table] = composite_pks[table]
            continue
        # Find the PK: first column ending in "key" whose base contains a
        # substring of the table name (e.g. "custkey" contains "cust" from "customer")
        for col in cols:
            if not col.name.endswith("key"):
                continue
            base = col.name.split("_", 1)[1] if "_" in col.name else col.name
            # Strip trailing "key" to get the entity hint: custkey -> cust
            entity = base.removesuffix("key")
            if entity and entity in table:
                pks[table] = [col.name]
                break
    return pks


def _infer_foreign_keys(
    columns_by_table: dict[str, list[ColumnInfo]],
    primary_keys: dict[str, list[str]],
) -> list[ForeignKey]:
    """Infer FKs by matching column base-names across tables to known PKs."""
    # Build a map: base_name -> (table, column) for single-column PKs only.
    # Composite PKs (junction tables) are FK sources, not targets.
    pk_lookup: dict[str, tuple[str, str]] = {}
    for table, pk_cols in primary_keys.items():
        if len(pk_cols) != 1:
            continue
        base = pk_cols[0].split("_", 1)[1] if "_" in pk_cols[0] else pk_cols[0]
        pk_lookup[base] = (table, pk_cols[0])

    fks: list[ForeignKey] = []
    for table, cols in columns_by_table.items():
        for col in cols:
            if not col.name.endswith("key"):
                continue
            base = col.name.split("_", 1)[1] if "_" in col.name else col.name
            if base in pk_lookup:
                target_table, target_col = pk_lookup[base]
                if target_table != table:
                    fks.append(ForeignKey(
                        from_table=table,
                        from_column=col.name,
                        to_table=target_table,
                        to_column=target_col,
                    ))
    return sorted(fks, key=lambda fk: (fk.from_table, fk.from_column))


def _compute_naming_ambiguity(
    columns_by_table: dict[str, list[ColumnInfo]],
) -> float:
    """Fraction of column base-names that appear in more than one table.

    A "base name" strips the single-letter TPC-H prefix (e.g. c_name -> name,
    l_quantity -> quantity). High ambiguity means an LLM has many similarly-named
    columns across tables and must correctly pick the right one.
    """
    base_to_tables: dict[str, set[str]] = defaultdict(set)
    for table, cols in columns_by_table.items():
        for col in cols:
            base = col.name.split("_", 1)[1] if "_" in col.name else col.name
            base_to_tables[base].add(table)

    if not base_to_tables:
        return 0.0
    ambiguous = sum(1 for tables in base_to_tables.values() if len(tables) > 1)
    return ambiguous / len(base_to_tables)


def render_profile_markdown(profile: SchemaProfile) -> str:
    """Render a SchemaProfile as a readable markdown document."""
    lines: list[str] = []
    lines.append("# Schema Profile")
    lines.append("")
    lines.append(f"**Database:** `{profile.database_path}`")
    lines.append(f"**Tables:** {profile.table_count}")
    lines.append(f"**Total columns:** {profile.total_columns}")
    lines.append(f"**Naming ambiguity score:** {profile.naming_ambiguity_score}")
    lines.append("")

    lines.append("## Tables")
    lines.append("")
    for table in profile.tables:
        lines.append(f"### {table.name} ({table.row_count:,} rows)")
        lines.append("")
        if table.primary_key:
            lines.append(f"**Primary key:** {', '.join(table.primary_key)}")
            lines.append("")
        lines.append("| Column | Type | Nullable | PK |")
        lines.append("|--------|------|----------|----|")
        for col in table.columns:
            nullable = "YES" if col.is_nullable else "NO"
            pk = "PK" if col.is_primary_key else ""
            lines.append(f"| {col.name} | {col.data_type} | {nullable} | {pk} |")
        lines.append("")

    if profile.foreign_keys:
        lines.append("## Foreign Keys")
        lines.append("")
        lines.append("| From | To |")
        lines.append("|------|----|")
        for fk in profile.foreign_keys:
            lines.append(f"| {fk.from_table}.{fk.from_column} | {fk.to_table}.{fk.to_column} |")
        lines.append("")

    lines.append("## Naming Ambiguity Analysis")
    lines.append("")
    lines.append(
        f"Score: **{profile.naming_ambiguity_score}** "
        f"(0.0 = no ambiguity, 1.0 = high ambiguity)"
    )
    lines.append("")
    lines.append(
        "This measures what fraction of column base-names (after stripping "
        "table prefixes) appear in multiple tables. Higher scores mean LLMs "
        "must work harder to disambiguate which table a column belongs to."
    )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    db = Path(__file__).resolve().parents[2] / "data" / "tpcds.db"
    profile = crawl_schema(db)
    md = render_profile_markdown(profile)
    print(md)

    out = Path(__file__).resolve().parents[2] / "data" / "schema_profile.md"
    out.write_text(md)
    print(f"\nSaved to {out}")
