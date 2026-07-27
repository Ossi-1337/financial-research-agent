FROM python:3.14-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.14-slim AS runtime

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FRA_ALLOW_REMOTE_BIND=true \
    FRA_HOME=/data

RUN groupadd --gid 10001 fra \
    && useradd --uid 10001 --gid fra --create-home --shell /usr/sbin/nologin fra \
    && mkdir -p /data \
    && chown fra:fra /data

COPY --from=builder /wheels /wheels

RUN python -m pip install --no-cache-dir --no-index --find-links=/wheels \
        /wheels/financial_research_agent-*.whl \
    && rm -rf /wheels

USER fra
WORKDIR /app

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=12 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/status', timeout=5).read()"

CMD ["financial-research-agent", "serve", "--host", "0.0.0.0", "--port", "8000"]
