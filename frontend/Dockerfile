# 먼작귀 에이전트 — 화면 (Streamlit). 로직은 agent-app 에 HTTP 로만 묻는다.
FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py ./
COPY .streamlit/ ./.streamlit/
COPY img/ ./img/
RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3).status == 200 else 1)"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
