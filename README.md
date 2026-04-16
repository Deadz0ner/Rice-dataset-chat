# Rice EXIM Dataset Chat

Full-stack AI chat interface over an Indian rice export/import (EXIM) Excel dataset. Users upload an `.xlsx` file, ask natural-language questions, and receive answers grounded strictly in the dataset — no hallucinations.

Built with a **Retrieval-Augmented Generation (RAG)** pipeline: FastAPI backend, React + Vite frontend, sentence-transformers for embeddings, FAISS for vector search, and Groq (Llama 3.3 70B) for generation.

## Features

- Upload any `.xlsx` dataset and query it in natural language
- RAG pipeline: chunking → embedding → FAISS vector search → grounded LLM generation
- Answers cite specific dataset rows with relevance scores
- Explicit refusal when evidence is insufficient (no hallucination)
- Tiered data strategy: address fields in metadata only, not in embeddings
- Index-once architecture: embeddings and FAISS index built on first query, reused for subsequent queries
- Swappable LLM provider via `.env` (Groq, OpenAI, Together, Ollama — any OpenAI-compatible endpoint)

## Project structure

```text
.
├── backend
│   ├── app
│   │   ├── api            # FastAPI routes (health, datasets, chat)
│   │   ├── core           # Config (pydantic-settings), logging
│   │   ├── models         # Data models
│   │   ├── schemas        # Pydantic request/response schemas
│   │   ├── services       # RAG pipeline layers
│   │   │   ├── dataset_service.py       # Excel ingestion, cleaning, dedup
│   │   │   ├── chunking_service.py      # Row → semantic text chunks
│   │   │   ├── embedding_service.py     # sentence-transformers (all-MiniLM-L6-v2)
│   │   │   ├── vector_store_service.py  # FAISS IndexFlatIP
│   │   │   ├── prompt_service.py        # Grounded prompt construction
│   │   │   ├── llm_service.py           # OpenAI-compatible chat completion
│   │   │   └── rag_pipeline.py          # End-to-end orchestration
│   │   └── utils
│   ├── data               # Uploaded datasets
│   ├── .env.example
│   ├── Dockerfile
│   ├── requirements.txt
│   └── requirements-rag.txt
├── frontend
│   ├── src
│   │   ├── components     # ChatInput, MessageBubble, UploadPanel, LoadingDots
│   │   ├── hooks          # useAutoScroll
│   │   ├── pages          # ChatPage
│   │   ├── services       # API client (fetch wrappers)
│   │   └── types          # TypeScript interfaces
│   ├── .env.example
│   ├── Dockerfile
│   └── package.json
├── approach.md            # Detailed design document
├── docker-compose.yml
└── README.md
```

## Local setup

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-rag.txt
cp .env.example .env
# Edit .env with your API key and provider settings
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend base URL: `http://localhost:8000`

### Frontend

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Frontend URL: `http://localhost:5173`

## Docker setup

```bash
docker compose up --build
```

## API

### `GET /api/health`

Returns `{"status": "ok"}`.

### `POST /api/datasets/upload`

Uploads an Excel file. Cleans, normalizes, deduplicates, and loads into memory.

### `GET /api/datasets/summary`

Returns dataset metadata: file name, row/column counts, column names, sample rows.

### `POST /api/chat`

Accepts:

```json
{
  "message": "Which countries received basmati rice exports?"
}
```

Returns:

```json
{
  "answer": "The countries that received basmati rice exports are:\n* Turkey\n* Egypt\n* United Arab Emirates\n* Australia\n* Oman",
  "grounded": true,
  "sources": [
    {
      "row_id": "row-11521",
      "preview": "Product Description: INDIAN BASMATI RICE...",
      "score": 0.72,
      "metadata": { "foreign_country": "TURKEY", "quantity": 77.76, ... }
    }
  ],
  "note": null
}
```

## RAG pipeline

See [approach.md](approach.md) for the full design document covering each layer:

1. **Dataset ingestion** — Excel → cleaned DataFrame (17k raw → 15.7k deduped rows)
2. **Chunking** — rows → labeled semantic text (tiered: addresses in metadata only)
3. **Embeddings** — text → 384-dim vectors (all-MiniLM-L6-v2, L2-normalized)
4. **Vector store** — FAISS IndexFlatIP, cosine similarity, ~2ms search
5. **Prompt grounding** — numbered evidence rows + refusal rules
6. **LLM generation** — Groq/Llama 3.3 70B, ~1.5s per query

## Configuration

All settings via `.env` (see `.env.example`):

| Variable | Description | Example |
|----------|-------------|---------|
| `LLM_PROVIDER` | Provider label | `groq` |
| `OPENAI_API_KEY` | API key for the provider | `gsk_...` |
| `OPENAI_BASE_URL` | API base URL (blank = OpenAI) | `https://api.groq.com/openai/v1` |
| `OPENAI_MODEL` | Model name | `llama-3.3-70b-versatile` |
| `VECTOR_BACKEND` | Vector store type | `faiss` |
| `DEFAULT_DATASET_PATH` | Auto-load dataset on startup | `./data/March Month EXIM 2024.xlsx` |
