"""
오프라인 테스트 — API 키·모델 다운로드·Docker 없이 데이터·MCP·RAG·그래프 배선을 검증한다
(langgraph_addon/test_offline.py 와 같은 전략)

  - LLM      → ScriptedLLM : 키워드로 정해진 라우팅·도구 호출·답변을 내는 가짜 모델
  - 임베딩   → 해시 기반 가짜 임베더 + 메모리 Qdrant
  - MCP 서버 → 진짜 mcp_server.py 를 stdio 자식 프로세스로 실행 (진짜 data/chiikawa.db 사용)

LLM 이 똑똑한지는 검증하지 않는다. 라우팅·ReAct 사이클·승인 멈춤/재개·권한 차단·Reflection 상한·
턴별 초기화·HTTP API 가 의도대로 동작하는지를 검증한다.

실행:  pytest -q test_offline.py
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from pathlib import Path

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from qdrant_client import QdrantClient

import graph as G
import rag_module
import tools as T


# ── 가짜 LLM ─────────────────────────────────────────────────────
def _text(m) -> str:
    return m.content if isinstance(m.content, str) else str(m.content)


def _last_human(messages) -> str:
    return next((_text(m) for m in reversed(messages) if getattr(m, "type", "") == "human"), "")


class ScriptedLLM(BaseChatModel):
    """키워드 규칙으로 응답하는 결정적 가짜 채팅 모델.
      invoke                      → answer 노드  (system 프롬프트 일부를 그대로 답변에 넣는다)
      bind_tools(...).invoke      → agent 노드   (질문 키워드로 도구 호출, ToolMessage 를 보면 끝)
      with_structured_output(...) → route / reflect 노드
    """
    reflect_first_fail: bool = False
    always_fail: bool = False
    critic_error: bool = False           # True 면 비평 호출이 예외를 던진다 (Reflection 폴백 검증용)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        if "터져" in _last_human(messages):          # 모델 API 장애(429·타임아웃 등) 흉내
            raise RuntimeError("fake API outage")
        system = next((_text(m) for m in messages if m.type == "system"), "")
        prefix = "개선된 답변: " if "[이전 초안에 대한 지적" in system else "초안 답변: "
        evidence = system.split("\n\n", 1)[1] if "\n\n" in system else system
        return ChatResult(generations=[ChatGeneration(message=AIMessage(prefix + evidence[:600]))])

    def _with_tools(self, messages) -> AIMessage:
        system = next((_text(m) for m in messages if m.type == "system"), "")
        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(f"도구 결과 정리: {_text(last)[:200]}")
        q = _last_human(messages)
        me = re.search(r'user_id 는 반드시 "([^"]+)"', system).group(1)
        other = re.search(r"user=(\w+)", q)
        uid = other.group(1) if other else me
        ep = int(m.group(1)) if (m := re.search(r"(\d+)화", q)) else 1
        if "담아" in q:
            call = {"name": "add_to_watchlist", "args": {"user_id": uid, "ep": ep}, "id": "c1"}
        elif "빼" in q:
            call = {"name": "remove_from_watchlist", "args": {"user_id": uid, "ep": ep}, "id": "c1"}
        elif "시청 목록" in q:
            call = {"name": "get_watchlist", "args": {"user_id": uid}, "id": "c1"}
        elif "같이" in q:
            call = {"name": "find_episodes", "args": {"characters": ["치이카와", "우사기"]}, "id": "c1"}
        elif "영상" in q:
            call = {"name": "get_episode", "args": {"ep": ep}, "id": "c1"}
        else:
            return AIMessage("도구가 필요 없습니다.")
        return AIMessage("", tool_calls=[call])

    def bind_tools(self, tools, **kw):
        return RunnableLambda(self._with_tools)

    def with_structured_output(self, schema, **kw):
        def run(messages):
            msgs = list(messages)
            joined = " ".join(_text(m) if hasattr(m, "content") else str(m) for m in msgs)
            if schema is G.RouteDecision:
                q = joined.split("[질문]")[-1]
                return G.RouteDecision(needs_rag=("어떤" in q or "장면" in q),
                                       needs_tool=any(k in q for k in ("같이", "담아", "빼", "시청 목록", "영상")),
                                       reasoning="keyword")
            if self.critic_error:
                raise RuntimeError("fake critic failure")
            ok = (not self.always_fail) and ((not self.reflect_first_fail) or ("개선된" in joined))
            return G.Critique(ok=ok, feedback="" if ok else "화 번호를 근거에서 인용하세요")
        return RunnableLambda(run)


# ── 공통 픽스처 ───────────────────────────────────────────────────
class FakeEmbedder:
    """특성 해싱 Bag-of-Words 임베딩 (64차원). 같은 낱말을 공유하는 문장끼리 가까워진다."""
    def embed(self, texts):
        for t in texts:
            v = np.zeros(64)
            for w in t.lower().split():
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
            yield v / (np.linalg.norm(v) or 1)


@pytest.fixture(autouse=True)
def fake_rag(monkeypatch):
    monkeypatch.setattr(rag_module, "_embedding_model", FakeEmbedder())
    monkeypatch.setattr(rag_module, "_client", QdrantClient(":memory:"))
    monkeypatch.setattr(rag_module, "VECTORS_PATH", Path("없음.npy"))     # 미리 만든 768차원 벡터 대신 가짜 임베딩


@pytest.fixture(scope="module")
def mcp_tools(tmp_path_factory):
    """진짜 MCP 서버를 stdio 로 띄운다. 시청 목록 DB 는 임시 파일로 바꿔 실제 기록을 건드리지 않는다."""
    os.environ["WATCHLIST_DB"] = str(tmp_path_factory.mktemp("w") / "watchlist.sqlite")
    os.environ.pop("MCP_SERVER_URL", None)
    return asyncio.run(T.load_mcp_tools())


def call(tools, tool_name, **args):
    tool = next(t for t in tools if t.name == tool_name)
    return T.parse_mcp_result(asyncio.run(tool.ainvoke(args)))


def make_app(mcp_tools, llm=None, **kw):
    return G.build_graph(llm or ScriptedLLM(), mcp_tools, checkpointer=InMemorySaver(), **kw)


def turn(app, text=None, *, thread="t1", user="guest", resume=None):
    return asyncio.run(G.run_turn(app, text, thread_id=thread, user_id=user, resume=resume))


# ── 데이터 · MCP 도구 ─────────────────────────────────────────────
def test_mcp_lists_eight_tools(mcp_tools):
    assert {t.name for t in mcp_tools} == {"list_characters", "find_character", "find_episodes", "get_episode",
                                           "search_titles", "get_watchlist", "add_to_watchlist",
                                           "remove_from_watchlist"}


def test_alias_resolves_to_dub_name(mcp_tools):
    """더빙판 이름 '가르마'로 찾아도 화면에 쓰는 이름 '하치와레'가 나온다."""
    r = call(mcp_tools, "find_character", name="가르마")
    assert r["name"] == "하치와레" and r["episode_count"] > 100


def test_unknown_character_is_error_data(mcp_tools):
    r = call(mcp_tools, "find_character", name="피카츄")
    assert "error" in r and r["candidates"]


def test_find_episodes_together_and_exclude(mcp_tools):
    """1화는 치이카와와 토끼가 함께 나오고 가르마는 안 나온다 (공식 줄거리 기준)."""
    both = call(mcp_tools, "find_episodes", characters=["치이카와", "우사기"], limit=50)
    assert 1 in [e["ep"] for e in both["episodes"]]
    no_hachi = call(mcp_tools, "find_episodes", characters=["치이카와", "토끼"], exclude=["가르마"], limit=50)
    assert 1 in [e["ep"] for e in no_hachi["episodes"]] and no_hachi["total"] < both["total"]


def test_get_episode_has_video_and_synopsis(mcp_tools):
    r = call(mcp_tools, "get_episode", ep=1)
    assert r["url"].startswith("https://www.youtube.com/watch?v=")
    assert r["segments"][0]["title"] == "단단한 푸딩" and "우사기" in r["segments"][0]["characters"]
    assert "error" in call(mcp_tools, "get_episode", ep=9999)


def test_watchlist_add_remove(mcp_tools):
    assert call(mcp_tools, "add_to_watchlist", user_id="tester", ep=57)["status"] == "added"
    assert call(mcp_tools, "add_to_watchlist", user_id="tester", ep=57)["status"] == "already"
    assert [w["ep"] for w in call(mcp_tools, "get_watchlist", user_id="tester")["items"]] == [57]
    assert call(mcp_tools, "remove_from_watchlist", user_id="tester", ep=57)["status"] == "removed"
    assert "error" in call(mcp_tools, "add_to_watchlist", user_id="tester", ep=9999)


# ── RAG ──────────────────────────────────────────────────────────
def test_rrf_merge_prefers_items_ranked_high_in_both():
    merged = [i for i, _ in rag_module.rrf_merge([1, 2, 3], [3, 1, 4])]
    assert merged[0] == 1 and set(merged) == {1, 2, 3, 4}


def test_hybrid_search_finds_character_chunk():
    """키워드 검색이 고유명사를 잡는다: 더빙판 '가르마'로 물어도 하치와레 소개 청크가 상위에 온다."""
    hits = rag_module.search("가르마는 어떤 캐릭터야", top_k=4)
    assert any(h["source"] == "등장인물: 하치와레" for h in hits)


# ── 그래프 ───────────────────────────────────────────────────────
def test_route_rag_only_skips_agent(mcp_tools):
    out = turn(make_app(mcp_tools), "토끼는 어떤 캐릭터야?")
    log = out.value["log"]
    assert log[0].startswith("route: 문서=True 도구=False")
    assert any(l.startswith("rag:") for l in log) and not any(l.startswith("agent") for l in log)


def test_react_loop_calls_mcp_then_answers(mcp_tools):
    """agent → tools → agent(끝) → answer: Day1 ReAct 사이클이 한 바퀴 돈다."""
    out = turn(make_app(mcp_tools), "치이카와랑 토끼 같이 나온 화 알려줘")
    log = out.value["log"]
    assert any("find_episodes" in l for l in log) and "agent: 도구 단계 끝" in log
    assert [m.type for m in out.value["work"]] == ["ai", "tool", "ai"]
    assert "find_episodes" in out.value["messages"][-1].content       # 도구 결과가 답변 근거로 들어갔다


def test_rag_then_tool_order(mcp_tools):
    """둘 다 필요하면 RAG 먼저, 그다음 도구 (Day4 순서 규칙)."""
    log = turn(make_app(mcp_tools), "울면서 먹는 장면 나온 1화 영상 보여줘").value["log"]
    names = [l.split(":")[0] for l in log]
    assert names.index("rag") < names.index("agent")


def test_approval_interrupt_then_approve(mcp_tools):
    app = make_app(mcp_tools)
    out = turn(app, "5화 시청 목록에 담아줘", thread="h1", user="alice")
    assert out.interrupts and out.interrupts[0].value["actions"][0]["tool"] == "add_to_watchlist"
    assert call(mcp_tools, "get_watchlist", user_id="alice")["count"] == 0          # 승인 전에는 실행 안 됨
    out = turn(app, thread="h1", user="alice", resume=True)
    assert out.interrupts == () and [w["ep"] for w in call(mcp_tools, "get_watchlist", user_id="alice")["items"]] == [5]


def test_approval_reject_and_non_bool(mcp_tools):
    for i, value in enumerate([False, "no"]):
        app = make_app(mcp_tools)
        turn(app, "7화 시청 목록에 담아줘", thread=f"r{i}", user="bob")
        out = turn(app, thread=f"r{i}", user="bob", resume=value)
        assert "거절" in str([m.content for m in out.value["work"] if m.type == "tool"])
    assert call(mcp_tools, "get_watchlist", user_id="bob")["count"] == 0


def test_other_user_is_denied_without_asking(mcp_tools):
    out = turn(make_app(mcp_tools), "user=mallory 9화 시청 목록에 담아줘", thread="d1", user="carol")
    assert out.interrupts == () and "deny: 다른 사용자 ID 호출 차단" in out.value["log"]
    assert call(mcp_tools, "get_watchlist", user_id="mallory")["count"] == 0


def test_reflection_rewrites_then_caps(mcp_tools):
    out = turn(make_app(mcp_tools, ScriptedLLM(reflect_first_fail=True)), "토끼는 어떤 캐릭터야?")
    assert out.value["messages"][-1].content.startswith("개선된 답변")
    out = turn(make_app(mcp_tools, ScriptedLLM(always_fail=True)), "토끼는 어떤 캐릭터야?", thread="c2")
    assert out.value["attempts"] == G.MAX_ATTEMPTS + 1                     # 상한에서 멈춘다


def test_reflect_failure_keeps_draft(mcp_tools):
    """비평 호출이 실패해도 초안을 버리지 않고 그대로 확정한다."""
    out = turn(make_app(mcp_tools, ScriptedLLM(critic_error=True)), "토끼는 어떤 캐릭터야?", thread="cf")
    assert out.value["messages"][-1].content.startswith("초안 답변")
    assert any(l.startswith("reflect: 비평 실패") for l in out.value["log"])


def test_resume_by_other_user_does_not_execute(mcp_tools):
    """alice 의 담기 요청이 멈춘 사이 eve 로 재개하면 실행하지 않는다."""
    app = make_app(mcp_tools)
    turn(app, "11화 시청 목록에 담아줘", thread="u1", user="alice2")
    out = turn(app, thread="u1", user="eve", resume=True)
    assert "approval: 사용자 불일치 → 실행 안 함" in out.value["log"]
    assert call(mcp_tools, "get_watchlist", user_id="alice2")["count"] == 0


def test_partial_index_is_rebuilt():
    """색인 도중 끊겨 덜 찬 컬렉션이 남아 있으면 지우고 다시 색인한다."""
    from qdrant_client.models import Distance, PointStruct, VectorParams
    chunks = rag_module.load_chunks()
    name = rag_module.collection_name(chunks)
    client = rag_module.get_client()
    client.create_collection(name, vectors_config=VectorParams(size=64, distance=Distance.COSINE))
    client.upsert(name, points=[PointStruct(id=0, vector=[1.0] + [0.0] * 63, payload={"idx": 0})])
    rag_module.ensure_index()
    assert client.count(name).count == len(chunks)


def test_small_talk_skips_reflect(mcp_tools):
    log = turn(make_app(mcp_tools), "고마워!").value["log"]
    assert not any(l.startswith(("rag", "agent", "reflect")) for l in log)


class SkipToolsLLM(ScriptedLLM):
    """도구를 건너뛰는 모델 — 대화에 비슷한 답이 쌓이면 실제 모델이 이렇게 한다. 호출을 강제(tool_choice)해야만 도구를 부른다."""
    ignore_force: bool = False          # True 면 강제해도 안 부른다

    def bind_tools(self, tools, **kw):
        if kw.get("tool_choice") == "any" and not self.ignore_force:
            return RunnableLambda(self._with_tools)
        return RunnableLambda(lambda messages: AIMessage("도구가 필요 없습니다."))


class LinkInventingLLM(ScriptedLLM):
    """근거에 없는 유튜브 주소를 지어내는 모델."""
    def _generate(self, messages, stop=None, run_manager=None, **kw):
        fake = AIMessage('1화 "먼작귀" https://www.youtube.com/watch?v=tJzrjFUNR5w&t=3s')
        return ChatResult(generations=[ChatGeneration(message=fake)])


def test_router_needs_tool_forces_first_tool_call(mcp_tools):
    """라우터가 도구 필요라고 했는데 모델이 건너뛰면 호출을 강제한다 ('우사기 나온 화'에서 도구 없이 답한 사고)."""
    out = turn(make_app(mcp_tools, SkipToolsLLM()), "치이카와랑 토끼 같이 나온 화 알려줘", thread="f1")
    assert any("find_episodes" in l for l in out.value["log"])


def test_no_evidence_is_not_invented(mcp_tools):
    """도구를 끝내 못 불렀으면 모델에게 답을 맡기지 않는다 (제목·영상 주소를 지어내던 사고)."""
    out = turn(make_app(mcp_tools, SkipToolsLLM(ignore_force=True)), "치이카와랑 토끼 같이 나온 화 알려줘", thread="f2")
    assert out.value["messages"][-1].content == G.NO_EVIDENCE
    assert "answer: 근거 없음 → 고정 답변" in out.value["log"]


def test_invented_video_link_is_blocked(mcp_tools):
    """근거(도구 결과)에 없는 영상 주소가 답변에 있으면 다시 쓰게 하고, 그래도 있으면 내보내지 않는다."""
    out = turn(make_app(mcp_tools, LinkInventingLLM()), "치이카와랑 토끼 같이 나온 화 알려줘", thread="f3")
    assert out.value["messages"][-1].content == G.NO_EVIDENCE
    assert any(l.startswith("answer: 근거에 없는 영상 주소") for l in out.value["log"])


def test_unsupported_links_compares_video_ids_only():
    evidence = "영상: https://www.youtube.com/watch?v=kzAfNg6pAvI&t=3s"
    assert G._unsupported_links("https://www.youtube.com/watch?v=kzAfNg6pAvI&t=99s", evidence) == []
    assert G._unsupported_links("https://www.youtube.com/watch?v=tJzrjFUNR5w", evidence) == ["tJzrjFUNR5w"]


def test_turn_memo_resets_and_history_kept(mcp_tools):
    """다음 턴에 지난 턴의 도구 기록(work)·log 가 남지 않고, 대화(messages)는 이어진다."""
    app = make_app(mcp_tools)
    turn(app, "치이카와랑 토끼 같이 나온 화 알려줘", thread="m1")
    out = turn(app, "고마워!", thread="m1")
    assert out.value["work"] == [] and not any("find_episodes" in l for l in out.value["log"])
    assert [m.type for m in out.value["messages"]] == ["human", "ai", "human", "ai"]


# ── HTTP API ─────────────────────────────────────────────────────
def test_http_api_chat_approval_resume(mcp_tools):
    from fastapi.testclient import TestClient
    import server
    api = server.create_app(llm=ScriptedLLM(), mcp_tools=mcp_tools, checkpointer=InMemorySaver())
    with TestClient(api) as c:
        assert c.get("/health").json() == {"status": "ok"}
        chars = c.get("/characters").json()
        assert chars[0]["name"] == "치이카와" and chars[0]["image"] == "chiikawa"
        r = c.post("/chat", json={"session_id": "s1", "message": "12화 시청 목록에 담아줘", "user_id": "dave"}).json()
        assert r["status"] == "needs_approval" and r["approval"]["actions"][0]["args"]["ep"] == 12
        assert c.post("/chat", json={"session_id": "s1", "message": "안녕", "user_id": "dave"}).status_code == 409
        r = c.post("/resume", json={"session_id": "s1", "user_id": "dave", "approve": True}).json()
        assert r["status"] == "done" and [w["ep"] for w in c.get("/watchlist/dave").json()] == [12]
        assert c.post("/resume", json={"session_id": "s1", "user_id": "dave", "approve": True}).status_code == 409
        assert c.get("/history/s1").json()["pending_approval"] is False
        # 모델 장애로 턴이 죽어도 502 로 알리고, 그 세션은 막히지 않는다 (다음 질문은 정상 처리)
        assert c.post("/chat", json={"session_id": "s2", "message": "터져라", "user_id": "dave"}).status_code == 502
        r = c.post("/chat", json={"session_id": "s2", "message": "고마워", "user_id": "dave"})
        assert r.status_code == 200 and r.json()["status"] == "done"
