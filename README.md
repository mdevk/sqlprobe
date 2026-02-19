# SQLProbe: How Well Does AI Actually Write SQL?

LLMs score 86-91% on [Spider 1.0](https://yale-lily.github.io/spider), the standard text-to-SQL benchmark. Then [Spider 2.0](https://spider2-sql.github.io/) dropped with real enterprise schemas and that number fell to 21%. SQLProbe lets you measure that gap on your own schema.

## Results

TPC-H, 20 queries, 3 models:

| Model | Easy | Medium | Hard | Very Hard | Overall |
|---|---|---|---|---|---|
| Claude Sonnet | 100% | 60% | 40% | 40% | **60%** |
| Gemini Flash | 100% | 40% | 20% | 0% | **40%** |
| GPT-4o | 100% | 40% | 0% | 0% | **35%** |

Every model nails simple queries. The wheels come off at joins and subqueries.

![SQLProbe Dashboard](docs/dashboard-screenshot.png)

## Quick Start

```bash
git clone https://github.com/your-org/sqlprobe.git
cd sqlprobe
python -m venv .venv && source .venv/bin/activate
pip install duckdb pydantic litellm sqlglot pandas streamlit python-dotenv

# API keys
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
EOF

python -m src.schema.crawler       # load TPC-H, crawl schema
python -m src.generator.query_gen  # generate 20 eval queries via LLM
python -m src.harness.runner       # run 60 LLM calls (20 queries x 3 backends)
python -m src.judge.scorer         # judge results, produce scorecard
streamlit run src/app/dashboard.py # open dashboard
```

## How It Works

```
src/
  schema/crawler.py      # reads INFORMATION_SCHEMA → SchemaProfile
  generator/query_gen.py # LLM-generates NL+SQL pairs across 4 difficulty tiers
  harness/runner.py      # sends questions to GPT-4o, Claude, Gemini via litellm
  judge/
    comparator.py        # executes both SQLs, compares results, diffs ASTs
    taxonomy.py          # 11 failure types (WRONG_TABLE, WRONG_JOIN, etc.)
    scorer.py            # aggregates into per-backend scorecard
  app/dashboard.py       # Streamlit dashboard
```

The judge is the interesting part. It doesn't just tell you pass/fail — it parses both the ground truth and generated SQL with [sqlglot](https://github.com/tobymao/sqlglot), diffs the ASTs, and classifies *why* it broke: wrong table, wrong join, wrong aggregation, etc.

## Roadmap

- [ ] Snowflake / Databricks connectors
- [ ] TPC-DS support (25 tables, 429 columns)
- [ ] CI-based regression tracking
- [ ] Custom schema upload

## License

MIT
