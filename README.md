# PseuDRAGON

Schema-centric compliance NLP for closed-network pseudonymization.

PseuDRAGON interprets database column names, SQL comments, and locally
provided policy or legal documents. It combines local LLM inference,
authority-aware retrieval, expert review, deterministic reuse of approved
corrections, and executable transformation generation.

The repository contains only generic program code, empty document
placeholders, and scripts that generate synthetic sample data. It does not
include production records, organization-specific policies, learned
heuristics, model responses, audit logs, or credentials.

## Workflow

1. Detect candidate PII columns from schema metadata.
2. Retrieve locally indexed policy and legal evidence.
3. Propose a pseudonymization method and executable code.
4. Let an authorized reviewer accept or correct each proposal.
5. Reuse approved corrections through deterministic rules and feedback.

Generated classifications, legal bases, and code are review proposals. They
must be validated by qualified personnel before use.

## Requirements

- Python 3.10 or newer
- A local OpenAI-compatible LLM endpoint
- Local access to the embedding models configured in `.env`

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Configuration

Create a local configuration file:

```bash
cp .env.example .env
```

`.env` is intentionally ignored by Git. Do not commit API keys, database
credentials, private endpoints, or internal file paths.

The retrieval score is:

```text
score = alpha * semantic_similarity + (1 - alpha) * authority
```

`SEMANTIC_SIMILARITY_WEIGHT` supplies alpha and can also be adjusted for an
individual run in the web interface. Source-authority defaults are separately
configurable:

```text
approved expert feedback  1.0
institutional policy      0.8
applicable national law   0.6
international guidance    0.4
```

These values are deployment settings, not a universal legal ranking. Each
deployment must curate applicable sources and validate the configuration.

## Local documents

Place only documents that are approved for the deployment environment under:

```text
resources/legal_documents/
├── institutional_policy/
├── national_law/
└── international_regulation/
```

Documents and generated embedding caches are ignored by Git. The empty text
files in these directories are placeholders only.

## Synthetic sample database

Generate the local sample database:

```bash
python resources/sample_db/setup_test_db_duckdb_for_1st_loop.py
```

The generated DuckDB file is ignored by Git and contains synthetic records.
To use another database, set `USE_SAMPLE_DB=false` and provide connection
settings in the local `.env`.

## Run

```bash
python PseuDRAGON.py
```

Open `http://127.0.0.1:5000`.

## Project structure

```text
PseuDRAGON.py
config/
pseudragon/
├── database/
├── domain/
├── heuristics/
├── llm/
├── logging/
├── rag/
├── reporting/
├── stages/
└── validation/
resources/
├── legal_documents/
├── prompts/
└── sample_db/
tests/
web_interface/
```

## Tests

The regression test verifies environment-defined and per-run retrieval
weights:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## Data and privacy

- No production data is bundled.
- Runtime outputs, logs, feedback, heuristics, legal documents, embeddings,
  and local databases are excluded by `.gitignore`.
- The included sample-data generator creates synthetic records.
- Reviewers remain responsible for legal applicability, transformation
  correctness, and deployment approval.

## License

MIT
