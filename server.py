"""
먼작귀 에이전트 — FastAPI 서버 (langgraph_addon/step5_server.py + 승인 재개 엔드포인트)
====================================================================================
compose 의 agent-app 자리에 그래프를 끼운다. Day5 와 같은 방식으로 lifespan 에서 한 번만 준비한다.

  POST /chat            {session_id, message, user_id} → 답변, 또는 승인 대기(needs_approval)
  POST /resume          {session_id, user_id, approve}  → 멈춘 그래프를 승인/거절 값으로 이어 간다
  GET  /characters      캐릭터 목록 (MCP list_characters 를 코드가 직접 호출)
  GET  /watchlist/{id}  시청 목록 (MCP get_watchlist)
  GET  /history/{sid}   체크포인터에 저장된 대화 기록
  GET  /health          헬스체크 (compose healthcheck · 화면의 연결 배지)

session_id 가 그대로 LangGraph thread_id 가 된다 → 대화 기억과 승인 대기 상태가 세션별로 저장된다.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

import rag_module
import tools as T
from graph import build_graph, run_turn


class UTF8JSONResponse(JSONResponse):
    """charset=utf-8 을 명시해 Windows PowerShell 에서도 한글이 깨지지 않게 한다 (step5 와 같음)."""
    media_type = "application/json; charset=utf-8"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=500)
    user_id: str = Field(default="guest", min_length=1, max_length=40)   # 실서비스라면 인증 토큰에서 꺼낼 값


class ResumeRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    user_id: str = Field(default="guest", min_length=1, max_length=40)
    approve: bool


class ChatResponse(BaseModel):
    session_id: str
    status: str                     # "done" | "needs_approval"
    answer: str = ""
    approval: dict | None = None    # 승인 대기일 때 사람에게 보여 줄 내용 (interrupt payload)
    trace: list[str] = []           # 이번 턴에 지나간 노드 기록


def _to_response(session_id: str, out) -> ChatResponse:
    """GraphOutput(v2) → 응답. interrupts 가 있으면 승인 대기, 없으면 마지막 메시지가 답변."""
    trace = out.value.get("log", [])
    if out.interrupts:
        return ChatResponse(session_id=session_id, status="needs_approval",
                            approval=out.interrupts[0].value, trace=trace)
    return ChatResponse(session_id=session_id, status="done",
                        answer=str(out.value["messages"][-1].content), trace=trace)


async def _warm_up_rag(retries: int = 5) -> None:
    """서버 시작 때 RAG 색인을 미리 만든다 → 첫 질문이 느려지지 않는다.
    Qdrant 가 '시작'은 했지만 아직 '준비'가 안 됐을 수 있어 몇 번 다시 시도한다 (Day5: 시작 ≠ 준비)."""
    for i in range(retries):
        try:
            await asyncio.to_thread(rag_module.ensure_index)   # 동기 임베딩·Qdrant 호출은 스레드로 (Day5 이슈 4)
            return
        except Exception as e:
            print(f"[agent-app] RAG 색인 준비 실패 ({i + 1}/{retries}): {e!r}", flush=True)
            await asyncio.sleep(2)


def create_app(llm=None, mcp_tools=None, checkpointer=None, warm_up: bool | None = None) -> FastAPI:
    """llm / mcp_tools / checkpointer 를 주입할 수 있게 해 두어 test_offline.py 가 가짜로 바꿔 끼운다.
    warm_up 기본값: 실제 모델을 쓸 때(llm 이 None)만 시작 시 RAG 색인을 만든다."""
    warm_up = (llm is None) if warm_up is None else warm_up

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with AsyncExitStack() as stack:
            saver = checkpointer
            if saver is None:
                db = os.environ.get("CHECKPOINT_DB")
                if db:      # 서버를 재시작해도 대화·승인 대기 상태가 남는다
                    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
                    saver = await stack.enter_async_context(AsyncSqliteSaver.from_conn_string(db))
                else:
                    saver = InMemorySaver()
            model = llm
            if model is None:
                from llm import get_llm
                model = get_llm()
            tools = mcp_tools
            if tools is None and os.environ.get("MCP_SERVER_URL"):
                # HTTP(compose): 도구를 부를 때마다 새 MCP 세션을 연다. 세션 하나를 붙잡고 있으면
                # mcp-server 가 재시작된 뒤 옛 세션 ID 로 계속 실패하지만, 이 방식은 다음 호출부터 바로 복구된다.
                tools = await T.load_mcp_tools()
            elif tools is None:
                # stdio(로컬): 호출마다 프로세스를 띄우면 느리므로 세션 하나를 서버 수명 동안 연다.
                tools = await stack.enter_async_context(T.open_mcp_tools())
            app.state.tools = {t.name: t for t in tools}
            app.state.graph = build_graph(model, tools, checkpointer=saver)
            if warm_up:
                await _warm_up_rag()
            yield

    api = FastAPI(title="먼작귀 에이전트", lifespan=lifespan, default_response_class=UTF8JSONResponse)
    session_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()

    def lock_for(session_id: str) -> asyncio.Lock:
        """같은 세션의 요청은 한 번에 하나씩 (같은 체크포인트를 두 요청이 덮어쓰지 않게)."""
        lock = session_locks.get(session_id)
        if lock is None:
            lock = session_locks[session_id] = asyncio.Lock()
        return lock

    async def call_tool(name: str, args: dict):
        """화면용 조회는 모델을 거치지 않고 코드가 MCP 도구를 직접 부른다. 도구 서버 장애는 503."""
        try:
            return T.parse_mcp_result(await api.state.tools[name].ainvoke(args))
        except Exception as e:
            raise HTTPException(503, f"도구 서버 호출 실패: {e}") from e

    @api.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    async def pending_approval(session_id: str) -> bool:
        """승인 대기 여부는 snap.next 가 아니라 snap.interrupts 로 판단한다.
        노드가 예외로 죽은 턴도 next 가 남기 때문에, next 로 보면 그 세션이 영원히 '승인 대기'로 막힌다."""
        snap = await api.state.graph.aget_state({"configurable": {"thread_id": session_id}})
        return bool(snap.interrupts)

    async def run_safely(**kw) -> object:
        """모델 API 429·타임아웃이나 도구 서버 장애로 그래프가 죽어도 500 대신 읽을 수 있는 502 로 돌려준다.
        같은 세션에 새 질문을 보내면 실패한 턴은 버려지고 정상으로 이어진다."""
        try:
            return await run_turn(api.state.graph, **kw)
        except Exception as e:
            raise HTTPException(502, f"처리 중 오류가 났습니다. 잠시 뒤 다시 시도해 주세요. ({type(e).__name__})") from e

    @api.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        async with lock_for(req.session_id):
            if await pending_approval(req.session_id):   # 승인 대기 중이면 먼저 승인/거절을 받도록 알린다
                raise HTTPException(409, "승인을 기다리는 작업이 있습니다. 먼저 승인하거나 거절해 주세요.")
            out = await run_safely(text=req.message, thread_id=req.session_id, user_id=req.user_id)
        return _to_response(req.session_id, out)

    @api.post("/resume", response_model=ChatResponse)
    async def resume(req: ResumeRequest) -> ChatResponse:
        async with lock_for(req.session_id):
            if not await pending_approval(req.session_id):
                raise HTTPException(409, "승인을 기다리는 작업이 없습니다.")
            out = await run_safely(thread_id=req.session_id, user_id=req.user_id, resume=req.approve)
        return _to_response(req.session_id, out)

    @api.get("/characters")
    async def characters() -> list:
        return await call_tool("list_characters", {})

    @api.get("/watchlist/{user_id}")
    async def watchlist(user_id: str) -> list:
        return (await call_tool("get_watchlist", {"user_id": user_id}))["items"]

    @api.get("/history/{session_id}")
    async def history(session_id: str) -> dict:
        snap = await api.state.graph.aget_state({"configurable": {"thread_id": session_id}})
        msgs = snap.values.get("messages", []) if snap.values else []
        return {"session_id": session_id, "pending_approval": bool(snap.interrupts),
                "messages": [{"role": m.type, "content": str(m.content)} for m in msgs]}

    return api


# uvicorn server:app 이 찾는 모듈 전역 앱 (실제 LLM · MCP 사용)
app = create_app()
