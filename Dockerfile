# Cronus Trading Bot - production image
#
# Two use modes from the same image:
#   - Live signal bot (default CMD): polls for new bars and broadcasts to Telegram
#   - One-off jobs (training/backtesting): docker run --rm cronus-bot python scripts/train.py
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by matplotlib / lightgbm / catboost at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p logs models reports data/processed \
    && useradd --create-home --shell /bin/bash bot \
    && chown -R bot:bot /app
USER bot

# Healthcheck: confirms the interpreter and core imports still work
HEALTHCHECK --interval=5m --timeout=10s --start-period=30s \
    CMD python -c "import src.telegram_bot.bot" || exit 1

ENTRYPOINT ["python"]
CMD ["scripts/live_signal.py", "--forever"]
