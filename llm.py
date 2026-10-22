"""
먼작귀 에이전트 — 공통 LLM 팩토리 (langgraph_addon/llm.py 와 같은 방식)
=====================================================================
그래프 빌더는 모델을 인자로 받는다(의존성 주입). 그래서 test_offline.py 는 같은 자리에
가짜 모델(ScriptedLLM)을 꽂아 API 키 없이 그래프 배선을 검증한다.

그래프가 모델에 요구하는 LangChain BaseChatModel 인터페이스:
  - invoke / ainvoke            : 메시지 리스트 → AIMessage          (answer 노드)
  - bind_tools(tools)           : tool_calls 를 낼 수 있는 모델       (agent 노드, ReAct)
  - with_structured_output(cls) : Pydantic 객체를 돌려주는 모델      (route·reflect 노드)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

# 이 폴더의 .env 를 환경변수로 읽는다. 이미 설정된 환경변수는 덮어쓰지 않는다.
load_dotenv(Path(__file__).parent / ".env")

# 기본 모델: 빠르고 저렴한 Haiku. 환경변수 LLM_MODEL 로 교체할 수 있다.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def get_llm(model: str | None = None, temperature: float = 0.0) -> ChatAnthropic:
    """ANTHROPIC_API_KEY 환경변수가 필요하다.

    temperature=0.0 : 라우팅·도구 선택을 최대한 결정적으로(재현 가능하게) 만든다.
    max_tokens=1024 : 한 번의 응답 길이 상한 (화 목록이 길어도 잘리지 않을 만큼).
    """
    return ChatAnthropic(
        model=model or os.environ.get("LLM_MODEL", DEFAULT_MODEL),
        temperature=temperature,
        max_tokens=1024,
    )
