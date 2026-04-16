# RAG Pipeline — Approach Document

## What this project is

An AI chat interface over an Excel-based rice/EXIM export dataset. Users upload an `.xlsx` file, then ask natural-language questions like _"Which countries did we export Basmati to in March?"_ — the system retrieves relevant rows from the dataset and generates an answer grounded strictly in that data. No hallucinations.

The architecture is a classic **Retrieval-Augmented Generation (RAG)** pipeline:

```
User question
    → Embed the question into a vector
    → Search a vector index for the most similar dataset rows
    → Feed those rows + the question to an LLM as context
    → LLM answers using only the provided evidence
```

The stack: **FastAPI** backend, **React + Vite** frontend, **Pandas** for data processing, **FAISS** for vector storage, and **Groq (Llama 3.3 70B)** for generation.

---

## Layer 1 — Dataset Ingestion (`DatasetService`)

**File:** `backend/app/services/dataset_service.py`

### What it does

Takes a raw Excel file and produces a clean, RAG-ready pandas DataFrame. This is the foundation — every downstream layer trusts that the data coming out of here is consistent and noise-free.

### The raw data problem

The source file (`March Month EXIM 2024.xlsx`) has 17,321 rows and 23 columns of Indian rice export records. In its raw state it has:

- **Inconsistent column names** — `Product Description`, `Value_FC`, `Foreign Country` (mixed casing, spaces, underscores)
- **Null values** — `Exporter City` is null for ~3,800 rows, `Exporter Address2` for ~9,700
- **Excel artifacts** — carriage return markers like `_x000D_` embedded in address strings
- **Double spaces** — `"VILLAGE CHABBA  TARN TARAN ROAD"` with extra internal whitespace
- **Exact duplicate rows** — ~1,600 rows are perfect copies (likely Excel export artifacts)
- **Useless columns** — `s` is just a spreadsheet serial number

### What the cleaning does, step by step

| Step | Before | After | Why it matters for RAG |
|------|--------|-------|----------------------|
| Column normalization | `Product Description` | `product_description` | Consistent snake_case so downstream code doesn't handle mixed naming |
| Drop serial column | `s: 10063020` | (removed) | Not real data — just a spreadsheet row counter |
| String cleanup | `"143001_x000D_"` | `"143001"` | Excel carriage-return junk would become noise tokens in embeddings |
| Whitespace collapse | `"CHABBA  TARN"` | `"CHABBA TARN"` | Double spaces are inconsistent and waste embedding tokens |
| Null filling | `Exporter City: NaN` | `Exporter City: ""` | Prevents the literal string `"nan"` from appearing in serialized chunks |
| Numeric null filling | `NaN` | `0` | Ensures numeric columns are always valid floats |
| Row ID injection | (none) | `row-0`, `row-1`, ... | Every row gets a stable, citable identity for the frontend source cards |
| Deduplication | 17,321 rows | 15,685 rows | Duplicate rows waste vector store space and skew retrieval scores |

### Concrete example

**Raw Excel row:**
```
s:                  10063020
Product Description: 1121 EXTRA LONG BASMATI SELLA RICE PACKI
Exporter:           SUPPLE TEK INDUSTRIES PRIVATE LIMITED
Exporter Address2:  143001_x000D_
Exporter City:      <null>
...
```

**After DatasetService:**
```
row_id:              row-0
product_description: 1121 EXTRA LONG BASMATI SELLA RICE PACKI
exporter:            SUPPLE TEK INDUSTRIES PRIVATE LIMITED
exporter_address2:   143001
exporter_city:       (empty string)
...
```

### The contract to the next layer

- 15,685 rows, 23 columns
- Zero null values anywhere
- Every row has a unique `row_id`
- No duplicate rows
- Clean, consistent snake_case column names
- No Excel artifacts

---

## Layer 2 — Chunking (`ChunkingService`)

**File:** `backend/app/services/chunking_service.py`

### Why this layer exists

Embedding models operate on **text**, not on tabular data. A raw dataframe cell like `100.0` in a `quantity` column is meaningless to an embedding model — it's just a number. But the string `"Quantity: 100.0"` carries semantic information: the model now knows this number represents a quantity, and can place it in the right region of vector space near other quantity-related concepts.

This is the single most important design decision in the pipeline. If the chunks are bad, no amount of better embeddings, fancier vector stores, or smarter prompts will fix retrieval quality. **Garbage text in → garbage vectors out → garbage retrieval → garbage answers.**

### Design decisions

**One row = one chunk.** The dataset consists of independent export transaction records. There's no narrative flow between rows, no paragraphs to split. Each row is a self-contained fact, so each row becomes one document.

**Tiered importance — content vs metadata.** This is the key architectural choice. Not every column belongs in the embedding text:

- **`content`** (embedded into a vector) — only high-signal business fields: product description, exporter name, buyer, countries, ports, quantities, rates, dates, IEC, bill number. These are what 95% of user queries target.
- **`metadata`** (stored alongside the vector, never embedded) — everything in content *plus* address fields (`exporter_address1`, `exporter_address2`, `exporter_city`). Available for filtering and citation, but invisible to the embedding model.

Why not embed addresses? A chunk like `"Product Description: BASMATI RICE | Exporter Address: VILLAGE CHABBA TARN TARAN ROAD, 143001"` pushes the embedding vector toward generic location/address space. For the query `"top exporters to Iran"`, that address text is pure noise — it dilutes the match. But for `"exporters from Tarn Taran"`, the address IS the answer. The solution: handle location queries at the retrieval layer via **metadata filtering** rather than hoping vector similarity catches address substrings in chunk text.

**Labeled serialization.** Instead of dumping values as `"IRAN, 100.0, MTS, Sea"`, we prefix each with a human-readable label: `"Foreign Country: IRAN | Quantity: 100.0 | Unit: MTS | Mode: Sea"`. This matters because embedding models (trained on natural language text) understand labeled pairs far better than raw value lists.

**Skip empty values.** If `exporter_city` is empty for a row, we don't include `"Exporter City: "` in the chunk. Empty labels add noise tokens that dilute the embedding without contributing information.

**Deliberate field ordering.** The most semantically important fields (product description, exporter, buyer, country) come first in the serialized text. Many embedding models weight the beginning of input more heavily, so front-loading key facts improves retrieval relevance.

**Pipe-delimited format.** A simple `" | "` separator is unambiguous, doesn't conflict with commas in values (addresses, product names), and is visually parseable by both humans and models.

**Framework-agnostic.** No LangChain, no LlamaIndex — just pandas and plain dicts. This keeps the chunking logic testable, debuggable, and swappable without pulling in heavy dependencies.

### How serialization works

For each DataFrame row, the service:

1. Iterates over a predefined column → label mapping (in priority order)
2. Checks if the column exists in the row and has a non-empty value
3. Formats it as `"Label: value"`
4. Joins all parts with `" | "`
5. Packages the result with `row_id`, the text as `content`, and the original values as `metadata`

### Concrete example

**Cleaned DataFrame row (from DatasetService):**
```python
{
    "row_id": "row-0",
    "product_description": "1121 EXTRA LONG BASMATI SELLA RICE PACKI",
    "exporter": "SUPPLE TEK INDUSTRIES PRIVATE LIMITED",
    "exporter_address1": "VILLAGE CHABBA TARN TARAN ROAD,",
    "exporter_address2": "143001",
    "exporter_city": "",           # ← empty, will be skipped
    "buyer": "TO THE ORDER---.-------------------",
    "foreign_country": "IRAN",
    "foreign_port": "Bandar Abbas",
    "indian_port": "Kandla",
    "quantity": 100.0,
    "unit": "MTS",
    "rate_fc": 998.37,
    "value_fc": 99837.0,
    "fob": 8207000.0,
    "rate": 82070.0,
    "currency": "USD",
    "mode": "Sea",
    "date": "01-Mar-24",
    "month": "MARCH",
    "year": 2024,
    "iec": "1213002877",
    "bill_number": 7969610
}
```

**After ChunkingService:**
```python
{
    "row_id": "row-0",
    "content": "Product Description: 1121 EXTRA LONG BASMATI SELLA RICE PACKI | Exporter: SUPPLE TEK INDUSTRIES PRIVATE LIMITED | Buyer: TO THE ORDER---.------------------- | Foreign Country: IRAN | Foreign Port: Bandar Abbas | Indian Port: Kandla | Quantity: 100.0 | Unit: MTS | Rate FC: 998.37 | Value FC: 99837.0 | FOB: 8207000.0 | Rate: 82070.0 | Currency: USD | Mode: Sea | Date: 01-Mar-24 | Month: MARCH | Year: 2024 | IEC: 1213002877 | Bill Number: 7969610",
    "metadata": {
        "product_description": "1121 EXTRA LONG BASMATI SELLA RICE PACKI",
        "exporter": "SUPPLE TEK INDUSTRIES PRIVATE LIMITED",
        "foreign_country": "IRAN",
        "quantity": 100.0,
        "exporter_address1": "VILLAGE CHABBA TARN TARAN ROAD,",  # ← in metadata, NOT in content
        "exporter_address2": "143001",                            # ← in metadata, NOT in content
        ...
    }
}
```

Notice two things:
1. **No address text in `content`** — `"VILLAGE CHABBA TARN TARAN ROAD,"` and `"143001"` are absent from the embedded string. The embedding vector stays focused on business facts.
2. **Address IS in `metadata`** — available for the retrieval layer to filter on when a user asks a location-specific question like `"exporters from Tarn Taran"`.

### Another example — Row 2 (Yemen export, has exporter_city)

**content (what gets embedded):**
```
Product Description: INDIAN LONG GRAIN BASMATI SELLA RICE,PKD |
Exporter: BANKE BIHARI JI EXPORTS |
Buyer: JASSAR FOR TRADING |
Foreign Country: YEMEN, DEMOCRATIC |
Foreign Port: Hodeidah | Indian Port: Kandla |
Quantity: 120.0 | Unit: MTS |
Rate FC: 955.3 | Value FC: 114636.0 |
FOB: 9423000.0 | Rate: 78525.0 |
Currency: USD | Mode: Sea |
Date: 01-Mar-24 | Month: MARCH | Year: 2024 |
IEC: 3316500090 | Bill Number: 7981555
```

**metadata (stored, not embedded):**
```python
{
    "exporter_address1": "Home Number 1120, Sector 6 Urban Estate, Karnal C",
    "exporter_city": "KARNAL",   # ← searchable via metadata filter, not vector similarity
    ...all other fields...
}
```

`"KARNAL"` does NOT appear in content — but when a user asks `"which city exported the most basmati"`, the retrieval layer can filter on `metadata["exporter_city"]` directly.

### Output stats

- **15,685 documents** produced (one per cleaned row, zero skipped)
- **Average chunk length: ~432 characters** (down from ~511 before removing address text)
- **14,184 documents** carry address metadata (the rest have empty address fields)
- Every document has a `row_id`, `content` (the text to embed), and `metadata` (full column values for citation + filtering)

---

## Layer 3 — Embeddings (`EmbeddingService`)

**File:** `backend/app/services/embedding_service.py`

### What it does

Takes the `content` string from each chunked document and converts it into a 384-dimensional numerical vector using the `all-MiniLM-L6-v2` sentence-transformer model. These vectors are what make semantic search possible — texts that mean similar things end up as vectors that are geometrically close in 384-dimensional space.

### Why sentence-transformers locally, not an API?

The `.env` uses Groq for the LLM, but Groq does not serve embedding endpoints. Rather than adding another API dependency (OpenAI embeddings, Cohere, etc.), we run `all-MiniLM-L6-v2` locally via CPU. It's free, fast enough for this dataset size, and removes an external dependency from the critical path.

### How it works

1. **Lazy model loading** — the model downloads from HuggingFace on first use and stays in memory for subsequent calls (singleton). This keeps app startup fast when embeddings aren't needed (health checks, dataset upload).
2. **`create_embeddings(documents)`** — takes the ordered list of content strings, runs them through the model in a single batch, returns a numpy array of shape `(N, 384)`. Order is preserved: `vectors[i]` is the embedding for `documents[i]`. This is critical because the next layer (FAISS) maps vectors to documents by positional index.
3. **`embed_query(query)`** — embeds a single user question using the same model and normalization settings, returning a 1-D array of shape `(384,)`. Using the same model for both documents and queries ensures they live in the same vector space.
4. **Normalization** — all vectors are L2-normalized (`normalize_embeddings=True`), so cosine similarity reduces to a simple dot product. This makes FAISS search faster and scoring more interpretable.

### Why this layer is separate from vector indexing

Embedding (text → vector) and indexing (vector → searchable data structure) are independent concerns. Keeping them in separate services means:
- You can swap the embedding model (MiniLM → OpenAI → Cohere) without touching FAISS code.
- You can swap the vector store (FAISS → Chroma → Pinecone) without re-implementing embedding logic.
- Each layer is independently testable.

### Performance on this dataset

| Metric | Value |
|--------|-------|
| Documents embedded | 15,685 |
| Model load time | ~13s (first call, from HF cache) |
| Embedding time | ~278s (CPU, single batch) |
| Output shape | `(15685, 384)` |
| Vector norm | 1.0 (L2-normalized) |
| Query embed time | ~0.02s |

### Retrieval quality sanity check

Query: `"basmati rice exports to Iran"`

| Rank | Row | Score | Content preview |
|------|-----|-------|-----------------|
| 1 | row-15079 | 0.708 | INDIAN BASMATI SELLA RICE, PACKED IN (25... |
| 2 | row-14609 | 0.705 | INDIAN BASMATI SELLA RICE, PACKED IN (25... |
| 3 | row-15080 | 0.700 | INDIAN BASMATI SELLA RICE, PACKED IN (5K... |
| 4 | row-12876 | 0.696 | INDIAN BASMATI SELLA RICE, PACKED IN 25... |
| 5 | row-15194 | 0.696 | INDIAN LONG GRAIN RICE (BASMATI), PACKED... |

All top results are semantically relevant — basmati rice exports. The tiered chunking strategy (addresses in metadata only) keeps the embedding focused on the business-relevant fields.

---

## Layer 4 — Vector Store (`VectorStoreService`)

**File:** `backend/app/services/vector_store_service.py`

### What it does

Takes the numpy array of document embeddings from `EmbeddingService` and builds a FAISS index that supports instant similarity search. Given a query vector, returns the top-k most similar document chunks with their cosine similarity scores.

### Why FAISS and why `IndexFlatIP`

FAISS is the standard library for dense-vector nearest-neighbour search. It runs entirely in-process (no external server), supports millions of vectors, and works natively with numpy arrays.

We use `IndexFlatIP` (inner product) specifically because our embeddings are L2-normalized. For unit-length vectors, inner product equals cosine similarity:

```
cos(a, b) = (a · b) / (||a|| × ||b||) = a · b    when ||a|| = ||b|| = 1
```

This is exact brute-force search with no approximation — perfectly suited for ~15k vectors × 384 dimensions where the search takes ~2ms.

### Why positional mapping is critical

FAISS stores raw float arrays — it knows nothing about `row_id`, `content`, or `metadata`. When a search returns "position 42 scored 0.71", the service maps position 42 back to the original chunk dict via a parallel `_documents` list stored at index-build time. If this positional mapping broke (reordering, skipping), FAISS would return the wrong dataset rows — silently producing incorrect grounded answers.

### How it works

1. **`build_index(documents, embeddings)`** — creates a `faiss.IndexFlatIP` with the embedding dimension, adds all vectors, and stores the parallel document list. For 15,685 vectors this takes ~11ms.
2. **`search(query_vector, top_k=5)`** — reshapes the query vector into FAISS's expected 2-D format, runs nearest-neighbour search, maps FAISS position indices back to chunk documents, and returns them with scores. Search takes ~2ms.

### Performance

| Metric | Value |
|--------|-------|
| Index build time | 0.011s for 15,685 vectors |
| Search latency | ~0.002s per query |
| Index type | `IndexFlatIP` (exact, no approximation) |
| Score range | −1 to 1 (cosine similarity) |

### Search quality examples

| Query | #1 Score | #1 Result |
|-------|----------|-----------|
| "basmati rice exports to Iran" | 0.708 | INDIAN BASMATI SELLA RICE, PACKED IN (25... |
| "exports from Kandla port" | 0.526 | 1121 GOLDEN SELLA BASMATI RICE... via Kandla |
| "highest value shipment in March 2024" | 0.460 | 546 BOXES OF INDIAN BASMATI RICE... |

---

## Layer 5 — Prompt Grounding (`PromptService`)

**File:** `backend/app/services/prompt_service.py`

### What it does

Takes the user's question and the retrieved evidence rows from FAISS, and assembles a structured prompt that constrains the LLM to answer only from the provided data. This is the core anti-hallucination mechanism.

### Prompt structure

The prompt has two parts:

**System instructions** — set once, tell the model its role and rules:
- You are a data analyst answering from an EXIM dataset
- Use ONLY the evidence rows — no outside knowledge
- If evidence is insufficient, explicitly refuse rather than guess
- Cite exact figures from the data
- Note when aggregations cover only the top matching rows, not the full dataset

**User message** — carries the actual evidence and question:
- Numbered evidence blocks: `[1] row-15079 (relevance: 0.71)` followed by the full content text
- The question at the end, with a final grounding instruction

### Why numbered evidence rows

Each row is formatted as `[1] row_id (relevance: score)` followed by its content. This lets the LLM:
- Reference specific rows in its answer ("According to row-15079...")
- Weight higher-relevance rows more heavily
- Makes the output auditable — you can trace any claim back to a specific dataset row

### Example output

For the query `"which countries received basmati rice exports?"` with 5 retrieved rows, the prompt is ~3,300 chars and looks like:

```
[System instructions — grounding rules]

EVIDENCE ROWS (5 rows retrieved):
---
[1] row-11521 (relevance: 0.72)
Product Description: INDIAN BASMATI RICE... | Foreign Country: TURKEY | ...

[2] row-6392 (relevance: 0.72)
Product Description: BASMATI RICE... | Foreign Country: EGYPT | ...

[3] row-4013 (relevance: 0.72)
Product Description: BASMATI RICE... | Foreign Country: UNITED ARAB EMIRATES | ...
...
---

QUESTION: which countries received basmati rice exports?

Answer based strictly on the evidence rows above.
```

---

## Layer 6 — LLM Generation (`LLMService`)

**File:** `backend/app/services/llm_service.py`

### What it does

Takes the structured chat messages from the prompt service (system + user) and sends them to a Groq chat completion endpoint. Returns the model's text response.

### Implementation

Uses the `openai` Python SDK pointed at Groq's OpenAI-compatible API (`https://api.groq.com/openai/v1`). The provider is configured entirely via `.env` — changing `OPENAI_BASE_URL` and `OPENAI_MODEL` switches to any OpenAI-compatible endpoint (OpenAI, Together, Ollama, vLLM).

- **Model:** `llama-3.3-70b-versatile` (via Groq)
- **Temperature:** 0.1 (low — prefer deterministic, grounded answers over creative ones)
- **Max tokens:** 1024
- Accepts both a list of chat messages (proper path) and a flat string (legacy fallback)
- Logs: provider/model, message count, char count, response time, token usage

### Performance

| Metric | Value |
|--------|-------|
| Response time | ~1.2–1.7s per query |
| Prompt tokens | ~1,100 (system + 5 evidence rows + question) |
| Completion tokens | ~30–50 |

---

## Pipeline integration (`RAGPipelineService`)

**File:** `backend/app/services/rag_pipeline.py`

The pipeline was refactored to fix a critical performance issue: the initial implementation re-chunked, re-embedded, and re-indexed the entire dataset on **every single chat message**. Now:

- **`_ensure_index()`** — builds the index once on the first query, then short-circuits on subsequent queries. Re-indexing only happens after `invalidate_index()` is called (triggered by a new dataset upload).
- **`invalidate_index()`** — called by the dataset upload route to force re-indexing when new data arrives.
- **"No rows" refusal** — returns `"Not found in provided dataset."` when retrieval finds nothing.
- **Source previews** — truncated to 200 chars to keep API responses compact.

---

## The full pipeline flow

```
Excel file
    ↓
[DatasetService] — clean, normalize, deduplicate, add row_id
    ↓
Clean DataFrame (15,685 rows × 23 columns, zero nulls)
    ↓
[ChunkingService] — serialize each row into labeled text
    ↓
15,685 documents [{row_id, content, metadata}]
    ↓
[EmbeddingService] — convert text → 384-dim vectors (all-MiniLM-L6-v2)
    ↓
15,685 vectors (384-dimensional, L2-normalized)
    ↓
[VectorStoreService] — index vectors in FAISS IndexFlatIP (~11ms)
    ↓
FAISS index (15,685 vectors, searchable)
    ↓
    ↓  ← User asks a question
    ↓
[EmbeddingService] — embed the question (same model)
    ↓
Query vector (384-dim)
    ↓
[VectorStoreService] — cosine similarity search top-k (~2ms)
    ↓
Retrieved rows [{row_id, content, metadata, score}]
    ↓
[PromptService] — build grounded prompt with numbered evidence + refusal rules
    ↓
"Answer ONLY from these rows: ..."
    ↓
[LLMService] — generate answer via Groq/Llama 3.3 70B (~1.5s)
    ↓
ChatResponse {answer, grounded, sources[]}
    ↓
Frontend renders answer + source cards with row_ids and scores
```

---

## Current status

| Layer | File | Status |
|-------|------|--------|
| Dataset ingestion | `dataset_service.py` | Done |
| Chunking | `chunking_service.py` | Done |
| Embeddings | `embedding_service.py` | Done — all-MiniLM-L6-v2, 384-dim, L2-normalized |
| Vector store | `vector_store_service.py` | Done — FAISS IndexFlatIP, ~2ms search |
| Prompt grounding | `prompt_service.py` | Done — numbered evidence, refusal rules, grounding constraints |
| LLM generation | `llm_service.py` | Done — Groq/Llama 3.3 70B, ~1.5s per query |
| Pipeline orchestration | `rag_pipeline.py` | Done — index-once, query-many, invalidation on upload |
| Frontend chat UI | `frontend/src/` | Done |
| API routes | `backend/app/api/routes/` | Done |
