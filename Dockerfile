# ── Stage 1: install dependencies ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY backend/ backend/
COPY frontend/ frontend/

EXPOSE 8000 8501

# Start FastAPI and Streamlit side-by-side
CMD sh -c "\
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 & \
  streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false \
  & wait"
