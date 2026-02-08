# PseuDRAGON 🐉

**PseuD**onymizing **RAG** is **ON** for Security Constrained Financial Industries

An LLM + RAG-powered framework that automates the entire pseudonymization pipeline—from PII detection to executable code generation—for constrained industries data.

---

## What does it do?

Feed it a database schema, and it will:
1. Find which columns contain personal information (Stage 1)
2. Generate processing policies for each column (Stage 2)
3. Let domain experts review and tweak the policies (Stage 3)
4. Spit out ready-to-run Python code (Stage 4)

It references legal documents (GDPR, local privacy laws, etc.) via RAG to back up its decisions with actual legal grounds.

---

## Installation

```bash
cd code
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the `code/` directory:

```bash
# OpenAI API key (required)
OPENAI_API_KEY=sk-...

# Use sample DB (recommended for first run)
USE_SAMPLE_DB=true

# Embedding settings
EMBEDDING_PROVIDER=local
LEGAL_EMBEDDING_MODEL=woong0322/ko-legal-sbert-finetuned
FEEDBACK_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Enable RAG
RAG_ENABLED=true
HEURISTICS_ENABLED=true
```

> Heads up: First run takes a while because it downloads the embedding models

---

## Running

```bash
cd code
python PseuDRAGON.py
```

Then open `http://localhost:5000` in your browser.

---

## Project Structure

```
code/
├── PseuDRAGON.py              # Entry point
├── config/config.py           # Configuration
├── pseudragon/
│   ├── pipeline.py            # Pipeline orchestrator
│   ├── stages/                # 4-stage processing
│   │   ├── stage1_pii_detection.py
│   │   ├── stage2_policy_synthesis.py
│   │   ├── stage3_hitl_refinement.py
│   │   └── stage4_code_generation.py
│   ├── rag/                   # RAG system
│   │   ├── retriever.py       # Legal document retrieval
│   │   └── expert_preference_manager.py  # Expert feedback learning
│   └── domain/policy_dsl.py   # Policy DSL definitions
├── resources/
│   ├── prompts/               # LLM prompts
│   ├── legal_documents/       # Legal PDFs
│   └── sample_db/             # Test DuckDB
└── web_interface/
    └── app.py                 # Flask web UI
```

---

## Key Features

### PII Detection (Stage 1)
- Heuristic pattern matching (fast first-pass filtering)
- LLM-based semantic analysis
- RAG context from legal documents
- Learns from previous expert feedback

### Policy Synthesis (Stage 2)
Supported actions:
- `KEEP` - Leave as-is
- `DELETE` - Remove the column
- `HASH` - Hash the values
- `MASK` - Apply masking (e.g., John D**)
- `TOKENIZE` - Replace with tokens
- `GENERALIZE` - Generalize values (e.g., age → age range)
- `ENCRYPT` - AES-256 encryption

### Human-in-the-Loop Refinement (Stage 3)
- Real-time policy editing
- 5 consistency validations
- Automatic audit logging

### Code Generation (Stage 4)
- Template-based generation (no LLM hallucination)
- Outputs production-ready Python code

---

## Database Connection

To use your own database instead of the sample:

```bash
# .env file
USE_SAMPLE_DB=false
DB_TYPE=oracle  # supports: oracle, mariadb, mysql

# For Oracle
DB_HOST=your-db-host
DB_PORT=1521
DB_SERVICE=your-service
DB_USER=username
DB_PASSWORD=password
```

Uncomment the relevant driver in `requirements.txt`:
```
cx_Oracle>=8.3.0      # Oracle
pymysql>=1.1.0        # MySQL/MariaDB
```

---

## Configuration

Tweak these in `config/config.py`:

```python
# Change LLM models
STAGE_1_LLM = "gpt-4o"  # or "gpt-4o-mini", etc.
STAGE_2_LLM = "gpt-4o"

# RAG parameters
CHUNK_SIZE = 200       # Chunk size (words)
TOP_K_RESULTS = 2      # Number of retrieved docs

# Batch processing
LLM_BATCH_SIZE = 8     # Columns processed in parallel
```

---

## Adding Legal Documents

Drop PDF files into `code/resources/legal_documents/` and they'll be indexed automatically.

```
legal_documents/
├── national_law/              # Domestic laws
├── international_regulation/  # GDPR, etc.
└── institutional_policy/      # Internal policies
```

> First run generates embeddings (takes time), subsequent runs use cached `.pkl` files

---

## Troubleshooting

**Q: Embedding model download fails**
- Check your network, or switch to `EMBEDDING_PROVIDER=openai`

**Q: OpenAI API errors**
- Verify your API key. If you're hitting rate limits, try reducing `LLM_BATCH_SIZE`

**Q: Out of memory**
- Set `PROMPT_MODE=lite` to reduce token usage

---

## References
- General legal embeddings: [nlpaueb/legal-bert-base-uncased](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
- Korean legal embeddings: [ko-legal-sbert-finetuned](https://huggingface.co/woong0322/ko-legal-sbert-finetuned)
- General embeddings: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

---

## License

MIT
