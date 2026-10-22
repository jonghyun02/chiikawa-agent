# 먼작귀 에이전트

먼작귀(치이카와) 캐릭터가 몇 화에 나오는지, 어떤 장면이 몇 화인지 찾아 주고, 시청 목록에 담기 전에 사용자 승인을 받는 에이전트입니다.
Agentic AI 과정 Day 1~7에서 배운 ReAct·MCP·RAG·오케스트레이션·LangGraph·Docker를 한 시스템에 적용했습니다.

> **공개본 안내** — 저작권이 있는 자료는 이 저장소에 넣지 않았습니다.
> 캐릭터 그림(`frontend/img/`), 공식 영상 설명·줄거리, 장면 기록과 대사(`data/`, `docs/`)가 빠져 있어서
> 이 상태로는 `docker compose up` 과 `test_offline.py` 가 바로 돌지 않습니다.
> 데이터를 만드는 순서는 [`data/README.md`](data/README.md) 를 보세요. API 키(`.env`)도 포함되어 있지 않습니다 (`.env.example` 참고).

## 구조

```
브라우저 :8501
   │
frontend (Streamlit)  ── 캐릭터 목록 탭 · 대화 탭 · 승인/거절 버튼 · 내 시청 목록
   │  POST /chat · POST /resume · GET /characters · GET /watchlist/{id}
agent-app (FastAPI + LangGraph) :8000
   │  ├─ route   : 라우터 LLM (needs_rag / needs_tool, 구조화 출력)
   │  ├─ rag     : 하이브리드 검색 (Qdrant 의미 검색 + 2-gram 키워드, RRF)
   │  ├─ agent ⇄ tools : ReAct 루프 (bind_tools + ToolNode)
   │  ├─ approval: 쓰기 도구는 interrupt() 로 멈추고 사람 승인 / deny: 다른 사용자 ID 차단
   │  ├─ answer → reflect ⟲ : 근거 기반 답변 + Reflection (재작성 1회 상한)
   │  └─ 체크포인터(SQLite, thread_id = session_id) : 대화 기억 · 승인 대기 상태 저장
   ├── MCP(HTTP) ──▶ mcp-server (FastMCP) :8001  ── SQLite: 화·소제목·캐릭터·등장 기록(읽기 전용) + 시청 목록
   └── REST ──────▶ qdrant :6333                 ── 줄거리·장면·캐릭터 소개 청크 2,825개 (embeddinggemma 벡터)
```

그래프 흐름 (`python graph.py` 를 실행하면 Mermaid 그림이 출력됩니다):

```
START → route ─┬→ rag ─┬→ agent ⇄ tools
               │       │    ├→ approval ⏸ → tools / agent
               ├───────┘    └→ deny → agent
               └──────────→ answer → [reflect ⟲ answer] → finalize → END
```

## 배운 기술이 쓰인 곳

| 배운 것 | 이 프로젝트 | 파일 |
|---|---|---|
| Day 1 ReAct 루프 | agent 노드가 도구를 고르고, ToolNode 결과를 보고 다시 판단 | `graph.py` |
| Day 1 프롬프트·구조화 출력 | 라우터 예시, 근거 기반 답변 규칙, `with_structured_output` | `graph.py` |
| Day 2 FastMCP 도구 서버 | 도구 8개 (읽기 6, 쓰기 2), SQLite, 읽기 전용 연결 | `mcp_server.py` |
| Day 2 최소 권한 | 데이터 DB `mode=ro`, SQL 파라미터 바인딩, 쓰기는 시청 목록만 | `mcp_server.py` |
| Day 3 RAG | 소제목·장면 단위 청킹, 다국어 임베딩(embeddinggemma), Qdrant | `rag_module.py` |
| Day 3 하이브리드 검색 | 의미 검색 + 키워드(2-gram) 검색을 가중 RRF 로 합침, 원어·더빙판 이름을 함께 넣어 찾음 | `rag_module.py` |
| Day 3 RAG 평가 | 장면 질문 65개로 검색 적중률 측정 → 모델·가중치 선택 | `scripts/eval_search.py` |
| Day 3 환각 방지 | 근거만 사용, 없으면 "제공된 정보로는 답할 수 없습니다" | `graph.py` |
| Day 4 라우터 · 조건부 분기 | needs_rag / needs_tool, RAG → 도구 순서 | `graph.py` |
| Day 4 Reflection Loop | reflect 노드 `Command(goto)`, 재작성 상한 | `graph.py` |
| Day 4 단기 메모리 | messages + 체크포인터 + `trim_messages` (최근 4턴), 턴마다 작업 메모 초기화 | `graph.py` |
| Day 5 Docker Compose | 4개 컨테이너, healthcheck, 볼륨, `.env` 분리, expose | `docker-compose.yml` |
| Day 5 FastAPI · Streamlit | lifespan 1회 초기화, `/chat`, 처리 과정 표시 | `server.py`, `frontend/app.py` |
| Day 6-7 LangChain | `ChatAnthropic`, `@tool`, `bind_tools`, `langchain-mcp-adapters` | `llm.py`, `tools.py` |
| Day 6-7 LangGraph | StateGraph, reducer, 조건부 엣지, `version="v2"` | `graph.py` |
| Day 6-7 Human-in-the-loop | `interrupt()` / `Command(resume)`, `/resume` 엔드포인트 | `graph.py`, `server.py` |
| Day 6-7 context_schema | user_id 를 State 가 아닌 `Runtime.context` 로 | `graph.py` |

## 데이터

| 자료 | 출처 | 만드는 법 |
|---|---|---|
| 화·소제목·공식 줄거리 | AniBox 공식 유튜브 재생목록 「[전편공개] 먼작귀」 설명란 (1~250화, 비공개 119·240화 제외) | `scripts/fetch_videos.py` → `data/raw/videos.jsonl` |
| 캐릭터 25명 (이름·공식 소개·성우) | 나무위키 「먼작귀/등장인물」과 개별 문서 (CC BY-NC-SA 2.0 KR) | `data/characters.json` |
| 등장 기록 | 공식 줄거리에서 캐릭터 이름 찾기(소제목 단위) + 장면 기록 + 검수한 인식 결과. 해시태그는 128개 영상에 같은 묶음이 붙어 있어 쓰지 않음 | `scripts/build_data.py` → `data/chiikawa.db` |
| 장면 기록 248화 · 2,352장면 | 공식 영상(오프닝·엔딩 제거) 1초 간격 캡처 + 더빙 대사 받아쓰기(Whisper, 246~250화는 Intel GPU)를 Claude Code 에이전트가 보고 기록. 화별 결말까지의 줄거리, 장면별 시각·장소·등장·행동·대사·검색어 | `C:/projects/yt-dlp/preprocess` 의 `transcribe.py`·`scene_sheets.py` → `scripts/describe_scenes.py --prepare` → 에이전트 → `--ingest` → `data/scenes/{화}.json` |
| 캐릭터 인식 (OWLv2) | 잘린 영상 1초마다 Intel GPU 로 캐릭터를 찾음(248화 약 30분). 장면 기록에 없던 등장 후보 118건을 잘라 직접 확인하니 맞은 것은 20건(17%) — 하얗거나 크림색인 둥근 캐릭터끼리, 회색 갑옷 씨끼리 구별을 못 함. 그래서 후보로만 쓰고 확인한 것만 더한다(`data/vision/review.json`, 문서에는 † 표시). 후보 확인 중에 장면 기록 오류(145~153화 '유두부 갑옷 씨' → 실제로는 초록 파자마 파티스)도 찾아 고침 | `C:/projects/yt-dlp/preprocess` 의 `char_refs.py`·`recognize.py`·`vision_review.py` → `scripts/import_vision.py` → `data/vision/{화}.json` |
| RAG 청크 2,825개 | 소제목 줄거리 · 캐릭터 소개 · 영상 줄거리 · 장면(유튜브 `&t=` 바로가기 포함) | `scripts/build_data.py` → `data/rag_chunks.jsonl` |
| RAG 문서 벡터 | 청크 2,825개의 embeddinggemma 벡터. PC 의 Intel GPU 로 미리 계산(43초, CPU 로는 17분) | `scripts/build_vectors.py` → `data/rag_vectors.npy` |
| 사람이 읽는 장면 문서 | 화마다 한 장 + 목차 (옵시디언) | `scripts/build_scene_docs.py` → `docs/scenes/` |

한계: 장면 기록은 자동 기록이라 틀린 곳이 있을 수 있습니다(3~23화는 Sonnet, 나머지는 Haiku가 작성하고 사람 검수는 안 함). 공개된 248화(1~250화, 비공개 119·240화 제외)에 모두 장면 기록이 있습니다.

## 장면 검색 평가

팬 말투 장면 질문 65개(그중 13개는 기록과 다른 낱말로 쓴 어려운 질문)로 정답 화를 찾는지 쟀습니다. 검색 결과 6개는 에이전트에 넘기는 수와 같습니다.

| 설정 | 1등이 정답 화 | 6개 안에 정답 화 | 어려운 질문 1등 / 6개 안 |
|---|---|---|---|
| 처음 (MiniLM, 같은 무게 RRF, 후보 10개) | 32% | 94% | 23% / 77% |
| 키워드 검색만 | 75% | 98% | 31% / 100% |
| MiniLM + 고친 점 | 75% | 95% | 46% / 85% |
| e5-large + 고친 점 | 88% | 98% | 62% / 100% |
| **embeddinggemma + 고친 점 (지금)** | **92%** | **98%** | **77% / 100%** |

고친 점은 네 가지입니다.

- 장면 청크의 유튜브 주소를 빼고 임베딩했습니다. 주소 때문에 캐릭터 소개 청크가 모든 질문에 1등으로 끼었습니다.
- 후보를 10개에서 30개로 늘렸습니다.
- 의미 검색의 무게를 0.5로 낮췄습니다.
- 원어 이름(하치와레)과 더빙판 이름(가르마)을 함께 넣어 찾습니다. 화면·답변은 원어 이름을 씁니다.

아직 못 찾는 질문이 하나 있습니다. "제초 검정 떨어져서 우는 장면"(61화)은 기록에 '제초 5급'이라고 적혀 있고, 제초 검정이 나오는 화도 57~64화, 107~110화로 많아서 못 찾습니다.

질문은 줄거리를 읽고 직접 썼기 때문에, 실제 팬 질문에서는 점수가 이보다 낮을 수 있습니다.

## 실행

### Docker Compose (발표용)

```bash
cp .env.example .env        # ANTHROPIC_API_KEY 채우기
docker compose up -d --build
```

- 화면: http://localhost:8501
- API 문서: http://localhost:8000/docs

### 로컬 (개발용)

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/import_vision.py                    # 캐릭터 인식 결과가 바뀌었을 때만
python scripts/build_data.py                       # 데이터가 바뀌었을 때만
python scripts/build_vectors.py                    # 청크가 바뀌었을 때만 (Intel GPU, --device NPU 도 됨)
python mcp_server.py                               # 터미널 1 (기본이 http :8001)
$env:MCP_SERVER_URL="http://127.0.0.1:8001/mcp"; uvicorn server:app --port 8000   # 터미널 2 (PowerShell)
streamlit run frontend/app.py                      # 터미널 3 → http://localhost:8501
```

터미널에서만 대화해 보려면 `python graph.py` (MCP 서버를 자식 프로세스로 띄웁니다).

## 테스트

```bash
pytest -q test_offline.py        # API 키 없이 21개: 데이터·MCP 도구·하이브리드 검색·라우팅·ReAct·승인·차단·Reflection·메모리·HTTP
python scripts/eval_search.py -v # API 키 없이 장면 질문 65개 검색 적중률 (틀린 질문 목록 포함)
python scripts/smoke_real.py     # 실제 모델로 시연 시나리오 7개 (API 키 필요, 비용 발생)
pytest -q test_frontend.py       # Streamlit AppTest 3개: 화면 로드·예시 질문 답변·승인 카드 → 실행 (agent-app 이 떠 있어야 함)
```

## 시연 질문

`python scripts/smoke_real.py` 로 실제 모델(Claude Haiku 4.5)에 돌린 결과(2026-10-22, 7개 모두 기대대로):

1. 토끼가 거대한 푸딩 위에서 뒹구는 장면 몇 화야? → 문서 → 1화 0:03 장면 바로가기
2. 마녀 나온 화 영상 링크랑 같이 알려줘 → 도구 (`find_character`) → 21·22화 + 처음 나오는 장면 링크
3. 파자마 파티스 나온 화 처음 5개만 영상 링크랑 같이 알려줘 → 도구 (`find_episodes`) → 50·127·128·133·144화
4. 치이카와랑 토끼는 나오는데 가르마는 안 나오는 화 3개만 알려줘 → 도구 (`find_episodes`, exclude)
5. 가르마가 꽃을 마이크 삼아 노래 불러주는 장면 찾아서 그 화 줄거리도 알려줘 → 문서 + 도구 → 199화
6. 그 화 내 시청 목록에 담아줘 → 대화 기억 + 쓰기 도구 → 승인 버튼
7. 학교 셔틀버스 시간표 알려줘 → "제공된 정보로는 답할 수 없습니다"
