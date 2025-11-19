FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" "./"

RUN uv sync --locked

COPY "src/serve.py" "./src/"
COPY  "models/best_model.pkl" "./models/"

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "9696"]