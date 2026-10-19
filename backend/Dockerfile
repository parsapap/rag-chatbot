FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

WORKDIR /app

# Copied before the code so this layer stays cached when only code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ ./app/

RUN useradd --create-home appuser && chown appuser:appuser /app
USER appuser

CMD ["python", "-m", "app.cli.hello"]
