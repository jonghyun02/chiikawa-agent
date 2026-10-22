"""
먼작귀 에이전트 — 도구 모음 (langgraph_addon/tools.py 와 같은 구조)
==================================================================
Day2 의 MCP 도구와 Day3 의 RAG 검색을 "LangChain 도구(Tool)"로 감싼다.
모델 입장에서는 출처(MCP 서버 / RAG)와 관계없이 같은 Tool 목록이다.

  search_docs     : RAG 하이브리드 검색 (@tool 로 감싼 rag_module.search)
  MCP 도구 8개    : mcp_server.py 의 @mcp.tool 들을 langchain-mcp-adapters 로 변환

[그래프에서 Tool 이 쓰이는 두 가지 방식]
  1) 코드가 직접 부르기 : rag 노드가 search_docs.invoke(...)  → "언제 검색할지"는 라우터가 정한다
  2) 모델이 고르게 하기 : agent 노드의 llm.bind_tools(tools) + ToolNode  → ReAct 루프 (Day1)
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools as _tools_from_session

import rag_module

SERVER_PATH = str(Path(__file__).parent / "mcp_server.py")
SERVER_NAME = "chiikawa"


@tool
def search_docs(query: str) -> str:
    """먼작귀 공식 줄거리(화·소제목별), 영상에서 기록한 줄거리·장면(시각·바로가기 주소), 캐릭터 소개 문서에서
    질문과 관련된 내용을 찾는다. (RAG)
    '울면서 뭔가 먹는 화'처럼 장면을 묘사한 질문이나, 캐릭터가 어떤 인물인지 묻는 질문에 쓴다."""
    hits = rag_module.search(query, top_k=6)
    if not hits:
        return "관련 문서를 찾지 못했습니다."
    # 청크 앞에 [출처] 를 붙여 모델이 답변에 근거를 인용할 수 있게 한다.
    return "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)


def mcp_connection() -> dict:
    """MCP_SERVER_URL 이 있으면 HTTP(compose), 없으면 stdio 자식 프로세스(로컬·테스트)."""
    url = os.environ.get("MCP_SERVER_URL")
    if url:
        return {SERVER_NAME: {"transport": "streamable_http", "url": url}}
    return {SERVER_NAME: {"transport": "stdio", "command": sys.executable, "args": [SERVER_PATH],
                          "env": {**os.environ, "MCP_TRANSPORT": "stdio"}}}


def _select(tools: list, names: set[str] | None) -> list:
    """names 화이트리스트로 도구를 고른다. 없는 이름을 요청하면 바로 알린다."""
    if names is None:
        return tools
    missing = set(names) - {t.name for t in tools}
    if missing:
        raise ValueError(f"MCP 서버에 없는 도구: {sorted(missing)} (서버 도구: {sorted(t.name for t in tools)})")
    return [t for t in tools if t.name in names]


async def load_mcp_tools(names: set[str] | None = None) -> list:
    """세션을 열어 두지 않는 방식: 도구를 부를 때마다 새 MCP 세션(stdio 면 새 프로세스)을 연다.
    테스트처럼 짧게 쓸 때 편하다. 시청 목록은 파일(WATCHLIST_DB)에 저장되므로 호출 사이에도 남는다."""
    client = MultiServerMCPClient(mcp_connection())
    return _select(await client.get_tools(), names)


@asynccontextmanager
async def open_mcp_tools(names: set[str] | None = None) -> AsyncIterator[list]:
    """MCP 세션 하나를 열어 둔 채로 도구를 제공한다 (서버에서는 lifespan 동안 유지).

        async with open_mcp_tools() as tools:
            app = build_graph(llm, tools, ...)
    """
    client = MultiServerMCPClient(mcp_connection())
    async with client.session(SERVER_NAME) as session:
        yield _select(await _tools_from_session(session, server_name=SERVER_NAME), names)


def parse_mcp_result(raw) -> dict | list | str:
    """MCP 도구 결과(content block 리스트 또는 문자열)를 파이썬 객체로 되돌린다."""
    if isinstance(raw, tuple):                      # (content, artifact) 형태로 오는 경우
        raw = raw[0]
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw
