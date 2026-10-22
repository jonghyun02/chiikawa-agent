"""
실제 모델로 시연 시나리오 돌려 보기 — Claude Haiku + 다국어 임베딩 + MCP(stdio)
==========================================================================
발표 전에 시연 질문이 기대대로 답하는지 확인한다. 승인 요청은 자동으로 '승인'한다.
ANTHROPIC_API_KEY 필요 (.env). 처음 실행하면 임베딩 모델(약 0.2GB)을 내려받는다.

실행:  python scripts/smoke_real.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

import tools as T  # noqa: E402
from graph import build_graph, run_turn  # noqa: E402
from llm import get_llm  # noqa: E402

SCENARIO = [
    "토끼가 거대한 푸딩 위에서 뒹구는 장면 몇 화야?",                      # 장면 → 화 (RAG)
    "마녀 나온 화 영상 링크랑 같이 알려줘",                                  # 캐릭터 → 화 + 장면 바로가기 (MCP)
    "파자마 파티스 나온 화 처음 5개만 영상 링크랑 같이 알려줘",              # 캐릭터 카드 버튼 질문
    "치이카와랑 토끼는 나오는데 가르마는 안 나오는 화 3개만 알려줘",        # 조합 조건 (MCP find_episodes)
    "가르마가 꽃을 마이크 삼아 노래 불러주는 장면 찾아서 그 화 줄거리도 알려줘",  # RAG → 도구
    "그 화 내 시청 목록에 담아줘",                                           # 대화 기억 + 승인
    "학교 셔틀버스 시간표 알려줘",                                           # 근거 없음 → 답하지 않음
]


async def main() -> None:
    async with T.open_mcp_tools() as mcp_tools:
        app = build_graph(get_llm(), mcp_tools, checkpointer=InMemorySaver())
        for q in SCENARIO:
            t0 = time.perf_counter()
            out = await run_turn(app, q, thread_id="smoke", user_id="demo")
            while out.interrupts:
                print("  ⏸ 승인 요청:", json.dumps(out.interrupts[0].value["actions"], ensure_ascii=False))
                out = await run_turn(app, thread_id="smoke", user_id="demo", resume=True)
            print(f"\nQ: {q}  ({time.perf_counter() - t0:.1f}초)")
            print("A:", out.value["messages"][-1].content)
            for line in out.value.get("log", []):
                print("   ·", line)


if __name__ == "__main__":
    asyncio.run(main())
