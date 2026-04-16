# ---- Stage 1: Build frontend ----
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python backend + built frontend ----
FROM python:3.12-slim
WORKDIR /app

# Install CPU-only PyTorch first (avoids downloading ~2GB of CUDA libs)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python deps
COPY backend/requirements.txt backend/requirements-rag.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-rag.txt

# Copy backend code
COPY backend/ ./

# Copy built frontend into backend/static so FastAPI serves it
COPY --from=frontend-build /app/frontend/dist ./static

# Download the embedding model at build time so it's cached in the image
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create data dir
RUN mkdir -p ./data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
