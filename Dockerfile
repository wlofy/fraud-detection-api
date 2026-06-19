# Render-friendly deploy image. Local dev still uses docker-compose.yml +
# requirements.txt (which include Kafka, matplotlib, pytest, full torch).
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Torch CPU-only wheel keeps the image small enough for Render's free tier.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY pyproject.toml ./
COPY src ./src
COPY web ./web

# Install our package so `fraud.api` resolves from `src/` layout.
RUN pip install --no-deps -e .

# Artifact + data dirs that train.py and data.py expect.
RUN mkdir -p artifacts data

EXPOSE 8000

CMD uvicorn fraud.api:app --host 0.0.0.0 --port ${PORT:-8000}
