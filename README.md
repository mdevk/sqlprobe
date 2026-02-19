# SQLProbe: How Well Does AI Actually Write SQL?

AI models hit 86-91% on [Spider 1.0](https://yale-lily.github.io/spider) — the go-to text-to-SQL benchmark. [Spider 2.0](https://spider2-sql.github.io/) came out with real enterprise schemas and that dropped to 21%.

SQLProbe runs that test on your schema.

## Results

TPC-H, 20 queries, 3 models:

| Model | Easy | Medium | Hard | Very Hard | Overall |
|---|---|---|---|---|---|
| Claude Sonnet | 100% | 60% | 40% | 40% | **60%** |
| Gemini Flash | 100% | 40% | 20% | 0% | **40%** |
| GPT-4o | 100% | 40% | 0% | 0% | **35%** |

Simple queries are fine. Joins and subqueries are where it falls apart.

![SQLProbe Dashboard](docs/dashboard-screenshot.png)

## Quick Start

```bash
git clone https://github.com/your-org/sqlprobe.git
cd sqlprobe
python -m venv .venv && source .venv/bin/activate
pip install duckdb pydantic litellm sqlglot pandas streamlit python-dotenv
```

Add API keys:

```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AI...
EOF
```

Run it:

```bash
python -m src.schema.crawler       # load TPC-H, crawl schema
python -m src.generator.query_gen  # generate eval queries
python -m src.harness.runner       # call 3 LLMs on each query
python -m src.judge.scorer         # score results
streamlit run src/app/dashboard.py # dashboard at localhost:8501
```

## How It Works

1. **Schema crawler** reads your DB's `INFORMATION_SCHEMA`
2. **Query generator** uses an LLM to write NL questions + ground-truth SQL at 4 difficulty levels
3. **Eval harness** sends each question to GPT-4o, Claude, and Gemini via [litellm](https://github.com/BerriAI/litellm)
4. **SQL judge** runs both the ground truth and generated SQL, compares the results, then uses [sqlglot](https://github.com/tobymao/sqlglot) to diff the ASTs and figure out *what* went wrong — wrong table, wrong join, wrong aggregation, etc.
5. **Dashboard** shows the scorecard, failure breakdown, and per-query drill-down

## Roadmap

- [ ] Snowflake / Databricks connectors
- [ ] TPC-DS (25 tables, 429 columns)
- [ ] CI regression tracking
- [ ] Custom schema upload

## License

MIT
