"""
화면 테스트 — Streamlit AppTest 로 frontend/app.py 를 실제 agent-app 에 붙여 돌린다 (Day5 화면 테스트)
================================================================================================
agent-app 이 떠 있어야 한다 (API_URL, 기본 http://localhost:8000). 없으면 건너뛴다.
실제 모델을 부르므로 ANTHROPIC_API_KEY 가 있는 서버에서만 의미가 있다.

실행:  pytest -q test_frontend.py
"""

from __future__ import annotations

import os
import uuid

import pytest
import requests
from streamlit.testing.v1 import AppTest

API = os.environ.get("API_URL", "http://localhost:8000")


def _api_up() -> bool:
    try:
        return requests.get(f"{API}/health", timeout=3).ok
    except requests.RequestException:
        return False


pytestmark = pytest.mark.skipif(not _api_up(), reason="agent-app 이 떠 있지 않음")


def new_app() -> AppTest:
    at = AppTest.from_file("frontend/app.py", default_timeout=180)
    at.session_state["session_id"] = uuid.uuid4().hex[:8]
    return at.run()


def test_screen_loads_characters_and_examples():
    at = new_app()
    assert not at.exception
    assert at.title[0].value == "먼작귀 에이전트"
    assert len([b for b in at.button if b.key and b.key.startswith("ask-")]) == 25     # 캐릭터 카드 25장
    assert any(b.key and b.key.startswith("ex-") for b in at.button)                  # 예시 질문 버튼


def test_example_question_gets_answer_with_trace():
    at = new_app()
    at.button(key="ex-우사기가 거대한 푸딩 위에서 뒹구는 장면 몇 화야?").click().run()
    assert not at.exception
    answer = at.session_state["messages"][-1]
    assert answer["role"] == "assistant" and "화" in answer["content"]
    assert any(line.startswith("route:") for line in answer["trace"])


def test_watchlist_request_shows_approval_then_executes():
    at = new_app()
    at.session_state["user_id"] = f"ui-{uuid.uuid4().hex[:6]}"
    at.chat_input[0].set_value("3화 시청 목록에 담아줘").run()
    assert at.session_state["pending"] is not None                 # 승인 카드가 떴다
    assert any(b.label == "승인" for b in at.button)
    next(b for b in at.button if b.label == "승인").click().run()
    assert at.session_state["pending"] is None
    assert "3화" in at.session_state["messages"][-1]["content"]
