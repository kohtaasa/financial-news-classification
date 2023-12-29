FROM --platform=linux/amd64 python:3.10-slim

WORKDIR /app

COPY . .

USER root

RUN apt-get update

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --without expt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless", "true", "--server.fileWatcherType", "none", "--browser.gatherUsageStats", "false"]
