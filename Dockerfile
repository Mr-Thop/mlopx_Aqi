FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MLFLOW_DISABLE_AGENT_HINT=1 GIT_PYTHON_REFRESH=quiet MLOPS_HOME=/app RUN_DIR=/data
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY mlops ./mlops
COPY sample ./sample
RUN useradd -m app && mkdir -p /data && chown app /data
USER app
EXPOSE 8000 8502
CMD ["uvicorn", "mlops.serve:app", "--host", "0.0.0.0", "--port", "8000"]
