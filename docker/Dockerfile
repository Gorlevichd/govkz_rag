FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY config.yaml backend.py serve.py create_index.py eval.py ./
COPY docker ./docker
COPY src ./src
COPY evals ./evals

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/chroma /app/models /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

CMD ["streamlit", "run", "serve.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
