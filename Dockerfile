# ============================================================
# NIFTY Quant Lab — Dockerfile (Multi-stage)
# ============================================================
# Stages:
#   base      — shared Python + system dependencies
#   api       — FastAPI server target
#   dashboard — Streamlit dashboard target
#   dev       — development image with all tools
# ============================================================

# ── Base stage
FROM python:3.12-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source
COPY nifty_quant_lab/ ./nifty_quant_lab/
COPY main.py .

# Create runtime directories
RUN mkdir -p /app/logs /app/reports/output /app/data/storage

# Non-root user
RUN useradd -m -u 1000 nql && chown -R nql:nql /app
USER nql

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── API stage
FROM base AS api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "nifty_quant_lab.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--log-level", "info"]

# ── Dashboard stage
FROM base AS dashboard

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "nifty_quant_lab/dashboard/app.py", \
     "--server.port", "8501", \
     "--server.address", "0.0.0.0", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]

# ── Dev stage (includes testing tools)
FROM base AS dev

USER root
RUN pip install --no-cache-dir pytest pytest-asyncio pytest-cov ruff black ipython
USER nql

CMD ["bash"]
