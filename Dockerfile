# =============================================================================
# GraphSentry Production Dockerfile
# Heterogeneous Graph Neural Network Engine for Anti-Money Laundering
# =============================================================================

FROM python:3.11-slim AS builder

WORKDIR /app

# Install minimal OS dependencies for compilation and network healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to minimize container footprint
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final Production Stage
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install curl for container health probes
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages and binaries from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create unprivileged application service user for container security
RUN groupadd -r graphsentry && useradd -r -g graphsentry graphsentry

# Copy application codebase and artifacts
COPY src/ /app/src/
COPY data/ /app/data/
COPY dashboard/ /app/dashboard/
COPY artifacts/ /app/artifacts/

# Grant appropriate permissions
RUN chown -R graphsentry:graphsentry /app

USER graphsentry

# Environment settings
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    HOST=0.0.0.0

EXPOSE 8000

# Container health probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch ASGI server with uvicorn
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
