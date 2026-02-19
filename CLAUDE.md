# SQLProbe -- Text-to-SQL Evaluation Platform

## What This Is
Evaluate how well LLMs generate SQL against real enterprise schemas.
The insight: academic benchmarks show 86-91% accuracy; enterprise-realistic benchmarks drop to 21%.
We measure that gap on YOUR schema.

## Architecture
- Python 3.12, DuckDB (local), sqlglot (SQL parsing/AST), litellm (LLM router), Streamlit (UI)
- TPC-H via DuckDB's built-in `INSTALL tpch; LOAD tpch; CALL dbgen(sf=1);` for prototype
- All source in src/, tests in tests/, eval data in data/

## Module Map
- src/schema/crawler.py -- reads INFORMATION_SCHEMA, outputs SchemaProfile (Pydantic)
- src/schema/profile.py -- renders SchemaProfile as markdown for LLM context
- src/generator/query_gen.py -- generates NL+SQL eval pairs at 4 difficulty tiers
- src/harness/backends.py -- litellm adapter per LLM backend (GPT-4o, Claude Sonnet, Gemini Flash)
- src/harness/runner.py -- orchestrates: for each query x backend, generate SQL, run judge
- src/judge/comparator.py -- THE MOAT: execute both SQLs, compare results, AST diff via sqlglot
- src/judge/taxonomy.py -- failure classification: WRONG_TABLE, WRONG_JOIN, WRONG_COLUMN, WRONG_AGGREGATION, WRONG_FILTER, WRONG_ORDER, SYNTAX_ERROR, RUNTIME_ERROR, PARTIAL_MATCH, CORRECT_DIFFERENT
- src/judge/scorer.py -- aggregate results into per-backend, per-difficulty scorecard
- src/app/dashboard.py -- Streamlit UI: scorecard table, failure charts, query drill-down

## Critical: SQL Judge is the moat
- comparator.py does 3 levels: (1) execute + DataFrame compare, (2) sqlglot AST parse + normalize, (3) semantic equivalence
- Always parse with sqlglot dialect="duckdb"
- Float comparison uses atol=0.01
- Sort DataFrames before comparing (ignore row order)
- Ignore column name differences (check values only)

## Conventions
- Pydantic models for all data structures (EvalQuery, EvalResult, ScoreCard, SchemaProfile)
- No classes where functions suffice
- Type hints everywhere
- pytest for tests, especially test_comparator.py
- Use pathlib, not os.path

## LLM API Keys
- Set OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY as env vars
- litellm handles routing -- never call provider SDKs directly

## Build Order
1. Load TPC-H into DuckDB, verify INFORMATION_SCHEMA works
2. Schema crawler + profile
3. Query generator (20 validated NL+SQL pairs)
4. Eval harness with 2+ backends
5. SQL Judge v1 (result comparison + basic AST diff)
6. Streamlit scorecard
