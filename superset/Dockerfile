FROM apache/superset:6.0.0

USER root

RUN uv pip install --python /app/.venv/bin/python --no-cache "clickhouse-connect>=0.13.0"

USER superset
