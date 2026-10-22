"""
먼작귀 에이전트 — LangGraph 그래프 (step3 라우팅·Reflection + step4 ReAct·승인을 한 그래프로)
==========================================================================================

    START → route ─┬→ rag ─┬→ agent ⇄ tools          (ReAct 루프, Day1)
                   │       │    │ ├→ approval ⏸ → tools / agent   (쓰기 도구는 사람 승인, interrupt)
                   ├───────┘    │ └→ deny → agent                (다른 사용자 ID 로 부르면 차단)
                   │            ↓
                   └──────────→ answer → [reflect ⟲ answer] → finalize → END

  배운 것                          →  이 그래프에서
  ───────────────────────────────────────────────────────────────────────────────
  Day1 ReAct (생각→행동→관찰)        →  agent 노드(bind_tools) ⇄ ToolNode 사이클
  Day2 MCP 도구                      →  mcp_server.py 의 8개 도구를 langchain-mcp-adapters 로 변환
  Day3 RAG                           →  rag 노드(라우터가 정할 때) + search_docs 도구(agent 가 다시 찾을 때)
  Day4 라우터(needs_rag/needs_tool)  →  route 노드 + with_structured_output + 조건부 엣지
  Day4 Reflection Loop               →  reflect 노드의 Command(goto) + 재작성 상한
  Day4 단기 메모리                   →  messages + 체크포인터(thread_id) + trim_messages
  Day6-7 Human-in-the-loop          →  approval 노드의 interrupt() / Command(resume)
  Day6-7 context_schema             →  user_id 는 State 가 아니라 Runtime.context 로 (모델이 못 바꿈)

[순서 규칙 — 왜 RAG 다음에 도구인가] (Day4 p.18)
  "가르마가 기타 치는 화 영상 보여줘" 는 RAG 로 화 번호(예: 22화)를 먼저 찾아야 get_episode 의 인자가
  생긴다. 앞 단계 결과가 다음 단계 입력이 되므로 병렬이 아니라 순차로 둔다.

[State 의 두 메시지 목록]
  messages : 턴을 넘어 유지되는 대화 (사용자 질문 + 최종 답변만)
  work     : 이번 턴의 ReAct 기록 (agent 의 tool_calls + ToolMessage). route 가 매 턴 비운다.
             도구 호출 기록이 대화에 쌓이지 않아 다음 턴 프롬프트가 짧게 유지된다.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import (AIMessage, AnyMessage, RemoveMessage, SystemMessage, ToolMessage,
                                     trim_messages)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

import tools as T

MAX_ATTEMPTS = 1          # Reflection 재작성 최대 횟수 (답변 생성은 최대 2번)
MAX_TOOL_ROUNDS = 4       # 한 턴에 agent 가 도구를 부를 수 있는 최대 횟수 (무한 루프 방지)
MEMORY_WINDOW = 8         # 모델에 보여 줄 최근 메시지 수 (4턴)
RECURSION_LIMIT = 40      # LangGraph 전체 스텝 상한 — 위 상한들과 함께 이중 안전장치 (Day6-7 p.63)
TOOL_TEXT_LIMIT = 6000    # answer 에 넣을 도구 결과 최대 글자 수 (토큰 비용 상한)
SENSITIVE = {"add_to_watchlist", "remove_from_watchlist"}     # 사람 승인이 필요한 쓰기 도구
USER_ARG = "user_id"
VIDEO_ID_RE = re.compile(r"(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})")
NO_EVIDENCE = ("제공된 정보로는 답할 수 없습니다. 조회된 근거가 없어서 화 번호나 영상 주소를 지어내지 않았어요. "
               "한 번 더 물어봐 주세요.")

ROUTER_PROMPT = """먼작귀(치이카와) 애니메이션 도우미의 라우터입니다. 질문에 무엇이 필요한지만 판단하세요.
- needs_rag : 공식 줄거리·영상 장면 기록·캐릭터 소개 문서를 찾아야 하는가
  (예: "울면서 뭔가 먹는 화", "가르마가 기타 치는 화", "토끼는 어떤 캐릭터야?")
- needs_tool : 데이터베이스 도구가 필요한가
  (예: "가르마 나오는 화 다 알려줘", "가르마가 처음 나온 화", "토끼는 몇 화에 나왔어?",
   "치이카와랑 토끼 같이 나온 화", "57화 영상", "내 시청 목록", "그 화 담아줘")
- 누가 몇 화에 나왔는지, 처음·마지막으로 나온 화, 나온 횟수는 데이터베이스에만 있으므로 needs_tool=true.
- 장면 묘사로 화를 찾고 그 화의 영상·정보까지 원하면 둘 다 true.
- 인사·감사·방금 답을 다시 정리해 달라는 요청은 둘 다 false."""

AGENT_PROMPT = """당신은 먼작귀(치이카와) 애니메이션 도우미의 '도구 담당'입니다. 답에 필요한 정보를 도구로 모으세요.
- 캐릭터가 나온 화, 여러 캐릭터가 같이 나온 화, 누구는 빠진 화: find_episodes
- 캐릭터 프로필과 나온 화 목록: find_character (화마다 영상 주소 url 과 처음 나오는 장면 바로가기 scene_url 포함)
- 특정 화의 줄거리·장면 목록(시각·바로가기 주소)·등장인물: get_episode
- 소제목에 들어간 낱말: search_titles / 장면 묘사: search_docs
- 시청 목록: get_watchlist, add_to_watchlist, remove_from_watchlist. user_id 는 반드시 "{user_id}" 를 쓴다.
- "그 화", "두 번째 거" 같은 말은 이전 대화에서 화 번호를 찾아 쓴다.
- 이전 대화에 비슷한 질문의 답이 있어도 베끼지 않는다. 이번 질문의 화 번호·제목·영상 주소는 도구로 새로 조회한다.
- 캐릭터 이름은 원어(하치와레, 우사기)와 더빙판(가르마, 토끼) 어느 쪽으로 넣어도 된다.
- 필요한 정보를 다 모았으면 도구를 더 부르지 말고, 찾은 내용을 짧게 정리해 끝낸다."""

ANSWER_PROMPT = """당신은 먼작귀(치이카와) 애니메이션 도우미입니다. 아래 근거만 사용해 한국어로 답하세요.
1. [문서 검색 결과]와 [도구 조회 결과]에 있는 내용만 쓴다. 화 번호·소제목·이름을 지어내지 않는다.
2. 화를 알려 줄 때 근거에 영상 주소가 있으면 함께 적는다. 장면 바로가기(scene_url)가 있으면 그 주소를 쓴다.
   목록이 길면 앞의 10개와 전체 개수를 적는다.
3. 근거가 비어 있고 인사 같은 일반 대화도 아니면 "제공된 정보로는 답할 수 없습니다"라고 답한다.
4. 캐릭터는 도구 결과의 이름(name)으로 부른다: 하치와레, 우사기, 모몽가, 쿠리만쥬, 랏코, 카니, 파자마 파티즈.
   문서에 더빙판 이름(가르마, 토끼, 하늘다람쥐, 밤만쥬, 해달, 헌책방, 파자마 파티스)으로 적혀 있어도 같은 캐릭터이니
   위 이름으로 바꿔 쓴다. 일본어 글자는 쓰지 않는다.
5. 시청 목록을 바꿨는지는 도구 결과의 status 로만 판단한다. 거절되었거나 실행되지 않았으면 담지 않았다고 말한다.
6. 짧고 읽기 쉽게 쓴다."""


# ── 구조화 출력 스키마 ─────────────────────────────────────────────
class RouteDecision(BaseModel):
    """route 노드의 출력."""
    needs_rag: bool = Field(description="공식 줄거리·영상 장면 기록·캐릭터 소개 문서 검색이 필요한가")
    needs_tool: bool = Field(description="화·등장인물·시청 목록 데이터베이스 도구가 필요한가")
    reasoning: str = Field(description="판단 이유 한 줄")


class Critique(BaseModel):
    """reflect 노드의 출력 (LLM-as-a-judge)."""
    ok: bool = Field(description="초안이 근거와 일치하고, 근거에 없는 화 번호·이름을 지어내지 않았고, 질문에 답했는가")
    feedback: str = Field(description="부족하면 무엇을 고쳐야 하는지")


# ── State / Context ───────────────────────────────────────────────
LOG_RESET = "__log_reset__"


def add_log(old: list[str], new: list[str]) -> list[str]:
    """log reducer: 이어 붙이되, 첫 원소가 LOG_RESET 이면 이번 턴 기록으로 새로 시작한다 (step3 와 같음)."""
    if new and new[0] == LOG_RESET:
        return list(new[1:])
    return (old or []) + new


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]     # 대화 (턴을 넘어 유지)
    work: Annotated[list[AnyMessage], add_messages]         # 이번 턴 ReAct 기록 (route 가 비움)
    needs_rag: bool
    needs_tool: bool
    rag_chunks: str
    draft: str
    critique: str
    attempts: int
    tool_rounds: int
    log: Annotated[list[str], add_log]


@dataclass
class Ctx:
    """실행 맥락. 로그인 사용자 ID 처럼 모델이 바꾸면 안 되는 값은 State 가 아니라 여기에 둔다."""
    user_id: str = "guest"


# ── 도우미 함수 ───────────────────────────────────────────────────
def _text(content) -> str:
    """메시지 content 가 블록 리스트(Anthropic 형식)여도 글자만 뽑는다."""
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


def _last_user_text(state: State) -> str:
    for m in reversed(state.get("messages", [])):
        if m.type == "human":
            return _text(m.content)
    return ""


def _history(state: State) -> list[AnyMessage]:
    """단기 메모리 창: 저장은 전부, 모델에는 최근 MEMORY_WINDOW 개만 (Day4 '최근 N턴')."""
    return trim_messages(state.get("messages", []), max_tokens=MEMORY_WINDOW, token_counter=len,
                         strategy="last", start_on="human", include_system=False)


def _history_text(state: State, n: int = 4) -> str:
    """라우터에 넘길 이전 대화 요약 (마지막 질문은 뺀다)."""
    msgs = state.get("messages", [])[:-1][-n:]
    who = {"human": "사용자", "ai": "도우미"}
    return "\n".join(f"{who.get(m.type, m.type)}: {_text(m.content)[:300]}" for m in msgs) or "(없음)"


def _tool_results(state: State) -> list[ToolMessage]:
    return [m for m in state.get("work", []) if m.type == "tool"]


def _calls(state: State) -> list[dict]:
    work = state.get("work", [])
    return (getattr(work[-1], "tool_calls", None) or []) if work else []


def _foreign_calls(calls: list[dict], user_id: str) -> list[dict]:
    """user_id 인자가 현재 사용자와 다른 호출 (= 남의 시청 목록을 보거나 바꾸려는 시도)."""
    return [c for c in calls if USER_ARG in c.get("args", {}) and c["args"][USER_ARG] != user_id]


def _unsupported_links(draft: str, evidence: str) -> list[str]:
    """답변 초안의 유튜브 영상 번호 중 근거에 없는 것 (= 모델이 지어낸 주소). 시각(&t=)은 따지지 않는다."""
    known = set(VIDEO_ID_RE.findall(evidence))
    return sorted({v for v in VIDEO_ID_RE.findall(draft) if v not in known})


def _reply_all(calls: list[dict], text: str) -> list[ToolMessage]:
    """tool_calls 하나하나에 ToolMessage 로 답한다 (모델 API 의 짝 맞추기 규칙)."""
    return [ToolMessage(text, tool_call_id=c["id"], name=c["name"]) for c in calls]


def build_graph(llm, mcp_tools: list, *, reflect: bool = True, checkpointer=None, rag_tool=None):
    """그래프를 조립한다.

    llm          : LangChain 채팅 모델 (테스트에서는 가짜 모델)
    mcp_tools    : MCP 서버에서 가져온 Tool 목록
    reflect      : True 면 answer 뒤에 Reflection 루프를 끼운다
    checkpointer : thread_id 단위 State 저장 (대화 기억 + interrupt 재개에 필수)
    rag_tool     : RAG 검색 Tool (기본 tools.search_docs, 테스트에서 바꿔 끼울 수 있다)
    """
    rag_tool = rag_tool or T.search_docs
    all_tools = [rag_tool, *mcp_tools]
    router = llm.with_structured_output(RouteDecision)
    critic = llm.with_structured_output(Critique)
    model = llm.bind_tools(all_tools)
    # 라우터가 "도구 필요"라고 했는데 모델이 도구를 건너뛰는 일이 있어서(대화에 비슷한 답이 쌓였을 때 베낌), 첫 라운드는 호출을 강제한다
    forced_model = llm.bind_tools(all_tools, tool_choice="any")

    # ── 노드 ──────────────────────────────────────────────────────
    def route(state: State) -> dict:
        """[턴의 시작] 질문을 분류하고, 지난 턴의 작업 메모를 비운다."""
        q = _last_user_text(state)
        try:
            d: RouteDecision = router.invoke(
                [SystemMessage(ROUTER_PROMPT),
                 ("user", f"[이전 대화]\n{_history_text(state)}\n\n[질문]\n{q}")])
        except Exception as e:      # 형식 이탈 등 → 부작용 없는 쪽(문서 검색만)으로 폴백 (Day5 라우터 이슈 3)
            d = RouteDecision(needs_rag=True, needs_tool=False, reasoning=f"라우터 실패 → 폴백 ({e!r:.60})")
        return {"needs_rag": d.needs_rag, "needs_tool": d.needs_tool,
                "rag_chunks": "", "draft": "", "critique": "", "attempts": 0, "tool_rounds": 0,
                "work": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
                "log": [LOG_RESET, f"route: 문서={d.needs_rag} 도구={d.needs_tool} ({d.reasoning})"]}

    def rag(state: State) -> dict:
        """RAG 근거 수집. 모델이 아니라 코드가 검색 도구를 직접 부른다."""
        try:
            text = rag_tool.invoke({"query": _last_user_text(state)})
        except Exception as e:
            return {"rag_chunks": f"(문서 검색 실패: {e})", "log": [f"rag 실패: {e!r}"]}
        n = text.count("\n\n") + 1 if text and not text.startswith("관련 문서를") else 0
        return {"rag_chunks": text, "log": [f"rag: 청크 {n}개 검색"]}

    async def agent(state: State, runtime: Runtime[Ctx]) -> dict:
        """ReAct 의 '생각' 단계. 모델이 다음에 부를 도구를 고른다 (없으면 도구 단계 끝)."""
        system = AGENT_PROMPT.format(user_id=runtime.context.user_id)
        if state.get("rag_chunks"):
            system += f"\n\n[이미 찾은 문서 검색 결과]\n{state['rag_chunks']}"
        first_round = state.get("needs_tool") and not _tool_results(state)      # 도구 결과를 한 번도 못 받았다
        msg = await (forced_model if first_round else model).ainvoke(
            [SystemMessage(system), *_history(state), *state.get("work", [])])
        calls = getattr(msg, "tool_calls", None) or []
        note = ", ".join(f"{c['name']}({json.dumps(c['args'], ensure_ascii=False)})" for c in calls)
        return {"work": [msg], "tool_rounds": state.get("tool_rounds", 0) + (1 if calls else 0),
                "log": [f"agent: {note}" if calls else "agent: 도구 단계 끝"]}

    def gate(state: State, runtime: Runtime[Ctx]) -> Literal["tools", "approval", "deny", "answer"]:
        """agent 다음 목적지 (step4 의 gate 와 같은 우선순위)."""
        calls = _calls(state)
        if not calls or state.get("tool_rounds", 0) > MAX_TOOL_ROUNDS:
            return "answer"
        if _foreign_calls(calls, runtime.context.user_id):
            return "deny"
        return "approval" if any(c["name"] in SENSITIVE for c in calls) else "tools"

    def deny(state: State, runtime: Runtime[Ctx]) -> dict:
        """다른 사용자 ID 로 부른 호출은 실행하지 않고 이유를 알려 준다."""
        return {"work": _reply_all(_calls(state), f"권한 없음: 현재 사용자({runtime.context.user_id})의 "
                                                  "시청 목록만 보거나 바꿀 수 있습니다. 실행하지 않았습니다."),
                "log": ["deny: 다른 사용자 ID 호출 차단"]}

    def approval(state: State, runtime: Runtime[Ctx]) -> Command[Literal["tools", "agent"]]:
        """사람 승인. interrupt() 에서 멈췄다가 Command(resume=True/False) 로 이어진다.
        재개하면 이 노드가 처음부터 다시 돌기 때문에 interrupt() 앞에는 부작용을 두지 않는다."""
        calls = _calls(state)
        asked = [c for c in calls if c["name"] in SENSITIVE]
        decision = interrupt({"question": "시청 목록을 바꿀까요?",
                              "actions": [{"tool": c["name"], "args": c["args"]} for c in asked]})
        if _foreign_calls(calls, runtime.context.user_id):
            # 멈춘 사이에 다른 사용자로 재개한 경우: 승인 버튼을 누른 사람의 목록이 아니므로 실행하지 않는다
            return Command(goto="agent", update={"work": _reply_all(calls, "권한 없음: 승인한 사용자와 요청한 "
                                                                           "사용자가 다릅니다. 실행하지 않았습니다."),
                                                 "log": ["approval: 사용자 불일치 → 실행 안 함"]})
        if decision is True:            # 명시적인 True 만 승인 ("no" 같은 문자열을 승인으로 오해하지 않게)
            return Command(goto="tools", update={"log": ["approval: 승인 → 실행"]})
        return Command(goto="agent", update={"work": _reply_all(calls, "사용자가 실행을 거절했습니다. 실행하지 않았습니다."),
                                             "log": ["approval: 거절 → 실행 안 함"]})

    def answer(state: State) -> dict:
        """[합류 지점] 근거 + 최근 대화로 답변 초안을 쓴다 (근거 기반 생성, Day3 환각 방지)."""
        evidence = []
        if state.get("rag_chunks"):
            evidence.append(f"[문서 검색 결과]\n{state['rag_chunks']}")
        results = "\n".join(f"- {m.name}: {_text(m.content)}" for m in _tool_results(state))
        if results:
            evidence.append(f"[도구 조회 결과]\n{results[:TOOL_TEXT_LIMIT]}")
        n = state.get("attempts", 0) + 1
        if not evidence and (state.get("needs_rag") or state.get("needs_tool")):
            # 근거를 찾으러 갔는데 빈손이다. 모델에게 맡기면 그럴듯한 제목·영상 주소를 지어낸다 ('우사기 나온 화'에서 실제로 발생)
            return {"draft": NO_EVIDENCE, "attempts": n, "log": ["answer: 근거 없음 → 고정 답변"]}
        evidence_text = "\n\n".join(evidence)
        system = ANSWER_PROMPT + "\n\n" + (evidence_text or "(근거 없음)")
        if state.get("critique"):
            system += f"\n\n[이전 초안에 대한 지적 — 반영해서 다시 작성]\n{state['critique']}"
        draft = _text(llm.invoke([SystemMessage(system), *_history(state)]).content)
        bad = _unsupported_links(draft, evidence_text)
        if bad:         # 근거에 없는 영상 주소 → 한 번 다시 쓰게 하고, 그래도 있으면 지어낸 답을 내보내지 않는다
            note = (f"\n\n[이전 초안에 대한 지적 — 반영해서 다시 작성]\n근거에 없는 영상 주소({', '.join(bad)})를 썼다. "
                    "근거에 적힌 주소만 그대로 쓰고, 근거에 없으면 주소를 쓰지 않는다.")
            draft = _text(llm.invoke([SystemMessage(system + note), *_history(state)]).content)
            bad = _unsupported_links(draft, evidence_text)
            if bad:
                return {"draft": NO_EVIDENCE, "attempts": n, "log": [f"answer: 근거에 없는 영상 주소 {bad} → 고정 답변"]}
        return {"draft": draft, "attempts": n, "log": [f"answer: 초안 {n}"]}

    def after_answer(state: State) -> Literal["reflect", "finalize"]:
        """근거를 쓴 답변만 검사한다 (인사 같은 일반 대화는 비평 호출을 아낀다)."""
        used_evidence = bool(state.get("rag_chunks")) or bool(_tool_results(state))
        return "reflect" if (reflect and used_evidence) else "finalize"

    def reflect_node(state: State) -> Command[Literal["answer", "finalize"]]:
        """[Reflection] 초안이 근거와 맞는지 검사하고 다음 노드를 직접 고른다 (Day4 p.10)."""
        results = "\n".join(f"- {m.name}: {_text(m.content)}" for m in _tool_results(state))[:TOOL_TEXT_LIMIT]
        try:
            c: Critique = critic.invoke(
                [SystemMessage("초안이 근거와 일치하는지, 근거에 없는 화 번호·소제목·이름을 지어내지 않았는지, "
                               "질문에 답했는지 평가하세요. 질문의 범위를 바꾸지 마세요 "
                               "(예: '나온 화'를 '주인공으로 나온 화'로 좁히지 않는다). "
                               "근거와 맞고 질문에 답했으면 ok=true 입니다.\n"
                               f"[문서 검색 결과]\n{state.get('rag_chunks', '')}\n[도구 조회 결과]\n{results}"),
                 ("user", f"질문: {_last_user_text(state)}\n초안: {state['draft']}")])
        except Exception as e:          # 비평이 실패해도 이미 쓴 초안은 버리지 않는다
            return Command(update={"log": [f"reflect: 비평 실패 → 초안 그대로 ({type(e).__name__})"]},
                           goto="finalize")
        if c.ok or state["attempts"] > MAX_ATTEMPTS:
            return Command(update={"log": [f"reflect: 통과 (ok={c.ok}, 초안 {state['attempts']})"]}, goto="finalize")
        return Command(update={"critique": c.feedback, "log": [f"reflect: 다시 쓰기 — {c.feedback}"]},
                       goto="answer")

    def finalize(state: State) -> dict:
        """확정된 초안 한 개만 대화 기록에 남긴다."""
        return {"messages": [AIMessage(state["draft"])]}

    # ── 조건부 엣지 ───────────────────────────────────────────────
    def after_route(state: State) -> Literal["rag", "agent", "answer"]:
        if state["needs_rag"]:
            return "rag"            # 둘 다 필요해도 RAG 먼저 (순서 규칙)
        return "agent" if state["needs_tool"] else "answer"

    def after_rag(state: State) -> Literal["agent", "answer"]:
        return "agent" if state["needs_tool"] else "answer"

    g = StateGraph(State, context_schema=Ctx)
    for name, fn in [("route", route), ("rag", rag), ("agent", agent), ("deny", deny),
                     ("approval", approval), ("answer", answer), ("finalize", finalize)]:
        g.add_node(name, fn)
    # work 목록의 tool_calls 를 실행. handle_tool_errors=True: 도구 서버 장애 같은 예외도 그래프를 죽이지 않고
    # 오류 ToolMessage 로 바꿔 모델이 사용자에게 설명하게 한다 (Day5 "오류는 데이터로")
    g.add_node("tools", ToolNode(all_tools, messages_key="work", handle_tool_errors=True))
    g.add_edge(START, "route")
    g.add_conditional_edges("route", after_route, ["rag", "agent", "answer"])
    g.add_conditional_edges("rag", after_rag, ["agent", "answer"])
    g.add_conditional_edges("agent", gate, ["tools", "approval", "deny", "answer"])
    g.add_edge("tools", "agent")            # 사이클 = Day1 의 while 루프
    g.add_edge("deny", "agent")
    g.add_conditional_edges("answer", after_answer, ["reflect", "finalize"])
    g.add_node("reflect", reflect_node)     # 나가는 엣지는 Command 반환 타입으로 선언
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


async def run_turn(app, text: str | None = None, *, thread_id: str, user_id: str = "guest", resume=None):
    """한 번 호출. text 가 있으면 새 질문, 없으면 멈춘 그래프를 resume 값으로 이어 간다. GraphOutput(v2) 반환."""
    payload = Command(resume=resume) if text is None else {"messages": [("user", text)]}
    return await app.ainvoke(payload,
                             config={"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT},
                             context=Ctx(user_id=user_id), version="v2")


async def main() -> None:
    """터미널에서 대화해 보기:  python graph.py   (ANTHROPIC_API_KEY 필요)"""
    from llm import get_llm

    async with T.open_mcp_tools() as mcp_tools:
        app = build_graph(get_llm(), mcp_tools, checkpointer=InMemorySaver())
        print(app.get_graph().draw_mermaid())
        print("질문을 입력하세요 (빈 줄이면 끝).")
        while q := input("\n나> ").strip():
            out = await run_turn(app, q, thread_id="cli", user_id="guest")
            while out.interrupts:
                req = out.interrupts[0].value
                print("⏸ 승인 요청:", json.dumps(req["actions"], ensure_ascii=False))
                ok = input("승인할까요? (y/n) ").strip().lower() == "y"
                out = await run_turn(app, thread_id="cli", user_id="guest", resume=ok)
            print("도우미>", out.value["messages"][-1].content)
            for line in out.value.get("log", []):
                print("   ·", line)


if __name__ == "__main__":
    asyncio.run(main())
