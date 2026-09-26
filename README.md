# PseuDRAGON

Schema-centric compliance NLP for closed-network financial-data pseudonymization.

PseuDRAGON interprets column names, descriptions and table context with a local
language model. Experts review the proposed field labels, supporting passages
and treatments. Approved decisions are retained for reuse, while selected
transformations are assembled into inspectable Python programs.

Release: **2026.09.26**. The default model is **gpt-oss-120b**, served locally
through an OpenAI-compatible endpoint (`gpt-oss:120b` in Ollama).

## Workflow

1. **Identify fields.** Apply stored rules and approved judgments where available;
   use the model for the remaining schema fields.
2. **Propose treatment with evidence.** Search institutional policy, then national
   law, then international guidance. Within each category, BM25 retrieves
   candidate passages and the model checks relevance. Stop at the first category
   with relevant evidence; otherwise continue to the next category.
3. **Review and retain decisions.** Experts correct field labels, inspect the
   evidence and select treatments. A changed label regenerates treatment
   proposals. Approved selections and corrections support subsequent reuse.
4. **Generate programs.** Check supported code/parameter contracts, preserve
   missing inputs, and generate transformations from the selected policies.

Source order is a configurable workflow scope, not a legal-precedence rule.
Relevance does not establish legal applicability. The interface retains the
source trace and leaves treatment approval to the reviewer. Empty retrieval
preserves an empty traceable result rather than forcing a non-PII label.

## Install

Use Python 3.10 or newer, preferably in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

For a closed-network deployment, provision dependencies and the model inside
that network before starting the application. The default source-order mode
uses a lexical index and does not load a dense embedding model. Optional legacy
modes require the embedding models configured in `.env`.

## Configure and run

1. Start a local OpenAI-compatible model server. `.env.example` uses
   `http://127.0.0.1:11434/v1` and the Ollama model name `gpt-oss:120b`.
2. Put approved PDF documents in the corresponding local directories:

   ```text
   resources/legal_documents/
   ├── institutional_policy/
   ├── national_law/
   └── international_regulation/
   ```

3. Generate synthetic sample records, then start the interface:

   ```bash
   python resources/sample_db/setup_test_db_duckdb_for_1st_loop.py
   python PseuDRAGON.py
   ```

4. Open **http://127.0.0.1:5001**.

The sample generator creates nine tables with schema descriptions and synthetic
rows. It is a runnable demonstration, not the original evaluation dataset or
an independent reproduction of the paper's reported scores. The second-loop
generator replaces the same local sample database; run it only when that is
intended. To use another supported database, set `USE_SAMPLE_DB=false` and
configure the connection in `.env`.

The web interface binds to loopback by default and has no authentication layer.
It is intended for a trusted local session. An organizational deployment needs
its own access controls. Generated programs run selected Python snippets:
review them before execution; syntax and contract checks are not a sandbox.

## Retrieval configuration

`RAG_RETRIEVAL_MODE=source_order` is the default. It does not read or combine
numeric source weights. `TOP_K_RESULTS` controls candidates per category.
The runtime profile displays the active mode and source trace.

`legacy` and `lexical` remain available for compatibility with earlier
experiments. Only those modes use their corresponding ranking parameters;
they are not the default release workflow. See the optional settings in
`.env.example`.

## Treatment contracts

- Recognized code forms are checked against their method and parameters.
- Edited custom code requires an explicit method/parameter declaration; the
  declaration remains marked as unverified when outside the supported forms.
- Missing values are preserved before applying a scalar transformation.
- The bounded display-rule compiler accepts explicit supplied requirements or
  supported account-display table syntax. Compiled alternatives retain exact
  source records and remain pending expert selection.
- `ENCRYPT` programs require `PSEUDRAGON_ENCRYPTION_KEY` from deployment-managed
  secret storage. The generated program neither creates nor prints the key.
- Generated programs use installed dependencies and never install packages.

## Tests

Run synthetic, offline regression tests:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 LLM_AUDIT_ENABLED=false \
  python -m unittest discover -s tests -p 'test_*.py' -v
```

The tests cover ordered retrieval and fallback, invalid relevance responses,
concurrent retrieval traces, compatibility modes, runtime UI rendering,
method/parameter validation, display-rule provenance and missing-value
preservation. Install Node.js to include the JavaScript rendering check;
without it, that check is explicitly skipped. Tests do not call a model API.

## Code style

Python follows Black with four-space indentation, double-quoted strings and a
100-character target line width. JavaScript, CSS and HTML follow the checked-in
Prettier configuration. Use UTF-8 and LF line endings throughout.

Write comments and docstrings in concise English. Describe current behavior,
invariants and design reasons. Put change history in Git commits, not in code
comments. Use Google-style `Args`, `Returns` and `Raises` sections where useful.
Keep schema descriptions, model prompts, generated-code templates and runtime
messages intact when making formatting-only changes.

```bash
python -m pip install -r requirements-dev.txt
npm ci
python -m black .
npm run format
python -m black --check .
npm run format:check
```

## Included and excluded files

This repository contains runtime code, prompt templates, synthetic sample-data
generators, generic regression fixtures and configuration examples. It excludes
production records, internal policy documents, evaluation responses, learned
decision stores, local databases, audit logs, credentials and manuscript files.
Runtime audit records can contain supplied metadata and model output; keep
them inside the deployment environment.

Release files and commit attribution omit personal author and institution
identifiers. A named GitHub account and its public activity can still identify
repository ownership; this URL is not an anonymous review service.

## License

MIT; see [LICENSE](LICENSE).
