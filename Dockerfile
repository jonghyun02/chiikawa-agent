# 먼작귀 에이전트 — 실행 이미지 (agent-app 과 mcp-server 가 같은 이미지를 쓴다, command 로 구분)
#   agent-app  : uvicorn server:app          (기본 CMD)
#   mcp-server : python mcp_server.py         (MCP_TRANSPORT=http)
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    # 임베딩 모델을 이미지 안에 저장 → 컨테이너 시작 때 내려받지 않는다
    FASTEMBED_CACHE_PATH=/app/.cache/fastembed

WORKDIR /app

# 1) 의존성 먼저 — 코드만 바뀌면 이 레이어는 캐시를 다시 쓴다 (Day5 레이어 캐시)
COPY requirements.txt .
RUN pip install -r requirements.txt

# 2) 다국어 임베딩 모델 미리 받기 (rag_module.EMBED_MODEL_NAME 과 같아야 한다). 컨테이너에서는 질문 임베딩만 한다
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('google/embeddinggemma-300m')"

# 3) 코드와 데이터 (원본 수집 파일 data/raw 는 넣지 않는다 — .dockerignore)
#    rag_vectors.npy: 문서 벡터를 PC 의 Intel GPU 로 미리 계산한 것 (컨테이너 CPU 로 하면 17분) — scripts/build_vectors.py
COPY *.py ./
COPY data/chiikawa.db data/rag_chunks.jsonl data/characters.json data/rag_vectors.npy data/rag_vectors.json ./data/

# 4) root 가 아닌 사용자로 실행. /data 는 시청 목록·대화 체크포인트를 쓰는 볼륨 자리
RUN useradd --create-home --uid 1000 app \
 && mkdir -p /data \
 && chown -R app:app /app /data
USER app

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=6 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
