# Rice Dataset Chat Scaffold

Interview-quality full-stack scaffold for a Claude-like chat interface over an Excel rice dataset using FastAPI, React, and a learning-friendly RAG architecture.

## What is implemented

- React + Vite chat UI with message bubbles, loading states, upload flow, auto-scroll, and responsive layout
- FastAPI backend with modular routing, schemas, services, config, logging, and CORS
- Excel upload and dataframe loading with Pandas
- Placeholder RAG service boundaries for chunking, embeddings, vector indexing, retrieval, prompt grounding, and LLM generation
- Dockerfiles plus `docker-compose.yml`

## What is intentionally left as TODO

- Row chunking strategy
- Embedding model integration
- FAISS or Chroma indexing internals
- Similarity search thresholds
- Grounding prompt refinement
- Hallucination-prevention heuristics and refusal logic

These learning sections are already scaffolded with function signatures, docstrings, examples, and extension notes.

## Project structure

```text
.
├── backend
│   ├── app
│   │   ├── api
│   │   ├── core
│   │   ├── models
│   │   ├── schemas
│   │   ├── services
│   │   └── utils
│   ├── .env.example
│   ├── Dockerfile
│   └── requirements.txt
├── frontend
│   ├── src
│   │   ├── components
│   │   ├── hooks
│   │   ├── pages
│   │   ├── services
│   │   └── types
│   ├── .env.example
│   ├── Dockerfile
│   └── package.json
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
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend base URL: `http://localhost:8000`

Optional future RAG packages:

```bash
pip install -r requirements-rag.txt
```

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

## API overview

### `GET /api/health`

Returns service health status.

### `POST /api/datasets/upload`

Uploads an Excel file and loads it into memory.

### `GET /api/datasets/summary`

Returns current dataset metadata including row count, columns, and sample rows.

### `POST /api/chat`

Accepts:

```json
{
  "message": "Which rice variety has the highest yield?"
}
```

Returns a grounded response shape:

```json
{
  "answer": "I could not find enough grounded evidence in the dataset to answer that question.",
  "grounded": false,
  "sources": [],
  "note": "Implement retrieval thresholds and fallback rules in the RAG pipeline."
}
```

## Next implementation steps

1. Implement `chunk_excel_rows()` in `backend/app/services/chunking_service.py`
2. Add a real embedding provider in `backend/app/services/embedding_service.py`
3. Build FAISS or Chroma indexing in `backend/app/services/vector_store_service.py`
4. Improve prompt grounding in `backend/app/services/prompt_service.py`
5. Replace the LLM placeholder in `backend/app/services/llm_service.py`
6. Add tests once the retrieval pipeline behavior is defined

## Notes

- Current chat answers are scaffold responses until RAG TODOs are implemented.
- The design is dependency-injection friendly and keeps provider choices swappable.
- The backend currently stores the loaded dataframe in memory for simplicity.
- If you are on Python 3.9, keep your `pip` updated before installing dependencies.
