"""
먼작귀 에이전트 — RAG 모듈 (Day3 RAG + Day3 심화 하이브리드 검색)
================================================================
검색 대상은 scripts/build_data.py 가 만든 data/rag_chunks.jsonl 이다.
  - 소제목 청크 : "57화 · 제초 검정" 의 공식 줄거리 + 등장인물   (AniBox 공식 영상 설명란)
  - 캐릭터 청크 : 이름·원어 이름·공식 소개·성우               (나무위키 등장인물 문서)
  - 영상 줄거리·장면 청크 : 화별 결말까지의 줄거리, 장면별 시각·장소·행동·대사·바로가기 주소
                           (영상 캡처를 보고 만든 기록, scripts/describe_scenes.py)

[Day5 HR 예제와 달라진 점]
  1) 청킹: 글자 수(350자) 대신 의미 단위(소제목 1개 = 청크 1개). Day5 이슈 6 "글자 수 청킹의 한계" 대응
  2) 임베딩: 영어 전용 bge-small 대신 다국어 모델(embeddinggemma). Day5 이슈 2 "언어 불일치" 대응
     장면 질문 65개로 잰 모델 비교(scripts/eval_search.py): MiniLM 75% · e5-large 88% · embeddinggemma 91% (1등 적중)
     문서 벡터는 이 PC 의 Intel GPU 로 미리 계산해 둔다(scripts/build_vectors.py) → CPU 17분 대신 1분, Docker 안에서도 바로 색인
  3) 재색인: 컬렉션 이름에 데이터 해시를 붙여, 데이터가 바뀌면 자동으로 새로 색인. Day5 이슈 1 대응
  4) 하이브리드: 의미 검색(Dense, Qdrant) + 키워드 검색(글자 2-gram)을 RRF 로 합친다 (Day3 4부)
     → "하치와레" 같은 고유명사는 키워드가, "울면서 뭔가 먹는 화" 같은 장면 묘사는 의미 검색이 잡는다.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv(Path(__file__).parent / ".env")

CHUNKS_PATH = Path(os.environ.get("RAG_CHUNKS", Path(__file__).parent / "data" / "rag_chunks.jsonl"))
VECTORS_PATH = Path(os.environ.get("RAG_VECTORS", CHUNKS_PATH.with_name("rag_vectors.npy")))
# 한국어 문서라 다국어 임베딩 모델을 쓴다 (768차원, 약 1.2GB). Dockerfile 에서 미리 내려받는다.
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "google/embeddinggemma-300m")
# 모델 카드가 정한 질문·문서 앞머리 (없는 모델은 그대로 넣는다)
EMBED_PREFIX = {
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
    "google/embeddinggemma-300m": ("task: search result | query: ", "title: none | text: "),
}
RRF_K = 60                     # RRF 상수 (Day3 p.30: RRF(d) = Σ 1 / (k + rank))
CANDIDATES = 30                # RRF 로 합치기 전에 검색마다 가져올 후보 수
DENSE_WEIGHT = 0.5             # RRF 에서 의미 검색 순위의 무게 (키워드 = 1.0). scripts/eval_search.py 로 정한 값
UPSERT_BATCH = 256             # Qdrant 에 한 번에 넣을 점 개수 (서버 요청 한도 32MB 아래로)

_embedding_model = None
_client: QdrantClient | None = None
_chunks: list[dict] | None = None
_lock = threading.RLock()


# ── 청크 ─────────────────────────────────────────────────────────
def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """rag_chunks.jsonl 을 읽는다. 한 줄 = {id, kind, text, source, ep?, seg?} 하나."""
    global _chunks
    with _lock:
        if _chunks is None:
            if not path.exists():
                _chunks = []
            else:
                with path.open(encoding="utf-8") as f:
                    _chunks = [json.loads(line) for line in f if line.strip()]
        return _chunks


def collection_name(chunks: list[dict]) -> str:
    """데이터 내용 + 임베딩 모델 이름의 해시를 컬렉션 이름에 넣는다 → 문서나 모델이 바뀌면 새 컬렉션으로 자동 재색인."""
    digest = hashlib.sha1((EMBED_MODEL_NAME + "".join(c["text"] for c in chunks)).encode("utf-8")).hexdigest()[:10]
    return f"chiikawa_{digest}"


URL_RE = re.compile(r"\s*영상: https?://\S+")


def embed_text(chunk: dict) -> str:
    """임베딩에 넣을 글. 장면 청크 끝의 유튜브 주소는 뜻이 없는 글자라 뺀다.
    (주소가 붙은 장면 청크만 질문과 멀어져서, 주소 없는 캐릭터 소개 청크가 모든 질문에 1등으로 끼던 원인)"""
    return URL_RE.sub("", chunk["text"])


# ── 임베딩 · Qdrant ───────────────────────────────────────────────
def get_embedding_model():
    """FastEmbed 모델 지연 로딩 싱글턴 (첫 호출 때 한 번만 만든다)."""
    global _embedding_model
    with _lock:
        if _embedding_model is None:
            from fastembed import TextEmbedding
            _embedding_model = TextEmbedding(model_name=EMBED_MODEL_NAME)
        return _embedding_model


def get_client() -> QdrantClient:
    """QDRANT_HOST 가 있으면 서버 모드(compose 의 qdrant 컨테이너), 없으면 로컬 파일 모드."""
    global _client
    with _lock:
        if _client is None:
            host = os.environ.get("QDRANT_HOST")
            if host:
                _client = QdrantClient(host=host, port=int(os.environ.get("QDRANT_PORT", "6333")))
            else:
                _client = QdrantClient(path=os.environ.get("QDRANT_PATH",
                                                           str(Path(__file__).parent / "qdrant_data")))
        return _client


def precomputed_vectors(chunks: list[dict], name: str):
    """scripts/build_vectors.py 가 가속기로 미리 만든 문서 벡터. 지금 청크·모델(= 컬렉션 이름)과 맞을 때만 쓴다."""
    meta_path = VECTORS_PATH.with_suffix(".json")
    if not (VECTORS_PATH.exists() and meta_path.exists()):
        return None
    if json.loads(meta_path.read_text(encoding="utf-8")).get("collection") != name:
        return None                             # 청크나 모델이 바뀌었다 → 직접 임베딩
    import numpy as np
    vectors = np.load(VECTORS_PATH)
    return vectors if len(vectors) == len(chunks) else None


def ensure_index() -> str | None:
    """컬렉션이 없으면 임베딩 → 색인한다. 멱등이라 검색할 때마다 불러도 된다. 컬렉션 이름을 돌려준다.
    미리 만든 벡터(rag_vectors.npy)가 맞으면 임베딩을 건너뛴다."""
    chunks = load_chunks()
    if not chunks:
        return None
    name = collection_name(chunks)
    with _lock:
        client = get_client()
        if client.collection_exists(name):
            if client.count(name).count == len(chunks):
                return name
            client.delete_collection(name)      # 색인 도중 끊겨 덜 찬 컬렉션 → 지우고 다시 만든다
        vectors = precomputed_vectors(chunks, name)
        if vectors is None:
            doc_prefix = EMBED_PREFIX.get(EMBED_MODEL_NAME, ("", ""))[1]
            vectors = list(get_embedding_model().embed([doc_prefix + embed_text(c) for c in chunks]))
        client.create_collection(name, vectors_config=VectorParams(size=len(vectors[0]),
                                                                   distance=Distance.COSINE))
        # Qdrant 서버는 요청 하나가 32MB 를 넘으면 거절한다 (768차원 2,825개를 한 번에 보내면 46MB) → 나눠 보낸다
        for s in range(0, len(chunks), UPSERT_BATCH):
            client.upsert(name, points=[PointStruct(id=i, vector=vectors[i].tolist(), payload={"idx": i})
                                        for i in range(s, min(s + UPSERT_BATCH, len(chunks)))])
        return name


# ── 검색 ─────────────────────────────────────────────────────────
def dense_search(query: str, top_k: int = CANDIDATES) -> list[int]:
    """의미 검색: 질문 벡터와 코사인 유사도가 높은 청크 번호를 순서대로 돌려준다."""
    name = ensure_index()
    if name is None:
        return []
    query_prefix = EMBED_PREFIX.get(EMBED_MODEL_NAME, ("", ""))[0]
    qv = list(get_embedding_model().embed([query_prefix + query]))[0]
    points = get_client().query_points(collection_name=name, query=qv.tolist(), limit=top_k).points
    return [p.payload["idx"] for p in points]


def _bigrams(text: str) -> Counter:
    """공백·문장부호를 뺀 글자 2-gram. 형태소 분석기 없이도 '하치와레가'와 '하치와레'를 같이 잡는다."""
    s = "".join(ch for ch in text.lower() if ch.isalnum())
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


QUESTION_WORDS = ("어떤", "무슨", "알려줘", "알려", "뭐야", "누구야", "있어", "보여줘", "해줘", "나오는", "나온",
                  "장면", "찾아줘", "몇 화", "하는", "거")

_keyword_index: tuple[int, list[Counter], Counter, list[float]] | None = None


def keyword_index(chunks: list[dict]) -> tuple[list[Counter], Counter, list[float]]:
    """청크별 2-gram, 2-gram 이 나온 청크 수(df), 길이 보정값을 한 번만 계산해 둔다 (질문마다 다시 세면 1초 걸림)."""
    global _keyword_index
    with _lock:
        if _keyword_index is None or _keyword_index[0] != id(chunks):
            grams = [_bigrams(c["text"]) for c in chunks]
            df = Counter(g for cg in grams for g in cg)
            norm = [math.sqrt(1 + sum(cg.values()) / 50) for cg in grams]
            _keyword_index = (id(chunks), grams, df, norm)
        return _keyword_index[1:]


def keyword_search(query: str, top_k: int = CANDIDATES) -> list[int]:
    """키워드 검색: 질문의 2-gram 이 청크에 얼마나 겹치는지(길이 보정) 점수를 매겨 상위 청크 번호를 돌려준다.
    BM25 의 직관(드문 단어가 많이 겹칠수록 높게)을 단순화한 것 — 희귀 2-gram 일수록 가중치를 높게 준다.
    '어떤', '알려줘' 같은 질문용 낱말은 줄거리 문장과 우연히 겹치므로 먼저 뺀다(불용어)."""
    chunks = load_chunks()
    for w in QUESTION_WORDS:
        query = query.replace(w, " ")
    q = _bigrams(query)
    if not q or not chunks:
        return []
    grams, df, norm = keyword_index(chunks)
    n = len(chunks)
    scored = []
    for i, cg in enumerate(grams):
        s = sum(math.log(1 + n / df[g]) for g in q if g in cg)
        if s > 0:
            scored.append((s / norm[i], i))
    return [i for _, i in sorted(scored, reverse=True)[:top_k]]


def rrf_merge(*rankings: list[int], k: int = RRF_K,
              weights: tuple[float, ...] | None = None) -> list[tuple[int, float]]:
    """RRF(Reciprocal Rank Fusion): 점수 대신 순위만으로 여러 검색 결과를 합친다 (Day3 p.30).
    weights 를 주면 검색마다 무게를 다르게 준다 (가중 RRF)."""
    weights = weights or (1.0,) * len(rankings)
    score: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, idx in enumerate(ranking, start=1):
            score[idx] = score.get(idx, 0.0) + w / (k + rank)
    return sorted(score.items(), key=lambda x: x[1], reverse=True)


# 랏코의 더빙판 이름 '해달'은 '위로 해달라고'·'일하게 해달라는' 같은 동사에도 들어 있다. 뒤에 이 글자가 오면 이름이 아니다
NAME_NOT_FOLLOWED_BY = {"해달": "라래랍랬"}


def mentions(text: str, name: str) -> bool:
    """text 에 캐릭터 이름이 이름으로 들어 있는가 (다른 낱말의 일부로만 나오면 아니다)."""
    stop = NAME_NOT_FOLLOWED_BY.get(name, "")
    for m in re.finditer(re.escape(name), text):
        nxt = text[m.end():m.end() + 1]
        if not nxt or nxt not in stop:
            return True
    return False


def expand_names(query: str) -> str:
    """질문에 나온 캐릭터의 다른 이름을 덧붙인다 (하치와레 ↔ 가르마, 우사기 ↔ 토끼 …).
    화면·답변은 원어 이름을 쓰지만 장면 기록 본문은 더빙판 이름으로 적혀 있어서, 한쪽 이름만으로는 키워드가 안 겹친다."""
    extra = []
    for c in load_chunks():
        names = c.get("names", []) if c["kind"] == "character" else []
        if any(mentions(query, n) for n in names):
            extra += [n for n in names if n not in query and n not in extra]
    return " ".join([query, *extra])


def named_characters(query: str, limit: int = 2) -> list[int]:
    """질문에 이름(더빙판·원어)이 그대로 나온 캐릭터의 소개 청크 번호.
    장면 청크가 2천 개를 넘으면서 '하치와레는 어떤 캐릭터야'의 소개 청크가 상위에서 밀려나서,
    이름이 정확히 나온 캐릭터의 소개는 검색 점수와 상관없이 근거에 넣는다(개체 연결)."""
    q = query.replace(" ", "")
    hits = [i for i, c in enumerate(load_chunks())
            if c["kind"] == "character" and any(mentions(q, n.replace(" ", "")) for n in c.get("names", []))]
    return hits[:limit]


def search(query: str, top_k: int = 4) -> list[dict]:
    """하이브리드 검색 결과 상위 top_k 개 청크를 {text, source, ep, seg, score} 로 돌려준다.
    질문에 이름이 나온 캐릭터의 소개 청크는 순위 밖이어도 맨 뒤 자리에 넣는다."""
    chunks = load_chunks()
    query = expand_names(query)
    merged = rrf_merge(dense_search(query), keyword_search(query), weights=(DENSE_WEIGHT, 1.0))[:top_k]
    have = {i for i, _ in merged}
    pinned = [i for i in named_characters(query) if i not in have][:max(0, top_k - 1)]
    if pinned:
        merged = merged[:top_k - len(pinned)] + [(i, 0.0) for i in pinned]
    return [{**chunks[i], "score": round(s, 4)} for i, s in merged]
