"""
먼작귀 에이전트 — Streamlit 화면 (Day5 HR 에이전트 frontend 와 같은 역할)
=======================================================================
화면은 Streamlit, 로직은 agent-app(FastAPI) 이 맡고 HTTP 로만 연결한다.

  머리말         : 제목 + 주요 캐릭터 그림 행렬
  대화 탭        : POST /chat → 답변(하치와레가 답하고 사용자는 치이카와)과 처리 과정 단계(라우팅·검색·도구·검토)
                   승인 대기(needs_approval)면 승인/거절 버튼 → POST /resume
  캐릭터 목록 탭 : GET /characters → 그림 카드. "나온 화 물어보기"를 누르면 대화 탭으로 넘어가 그 질문을 보낸다
  사이드바       : 연결 상태(GET /health), 사용자 ID, 새 대화
  색·글꼴은 frontend/.streamlit/config.toml 테마로 정한다.

실행:  streamlit run frontend/app.py      (API_URL 기본값 http://localhost:8000)
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path

import requests
import streamlit as st
from PIL import Image, ImageDraw

API = os.environ.get("API_URL", "http://localhost:8000")
IMG = Path(__file__).parent / "img"
CHAT_TAB, CHAR_TAB = "에이전트와 대화", "캐릭터 목록"
PARADE = ["chiikawa", "hachiware", "usagi", "momonga", "kurimanju", "rakko", "shisa", "furuhonya"]
EXAMPLES = [("usagi", "우사기가 거대한 푸딩 위에서 뒹구는 장면 몇 화야?"),
            ("hachiware", "하치와레가 꽃을 마이크 삼아 노래 불러주는 장면 찾아줘"),
            ("momonga", "모몽가가 야자열매 깨려다가 위로해달라고 떼쓰는 장면 몇 화야?"),
            ("pajama", "파자마 파티즈 나온 화 처음 5개만 영상 링크랑 같이 알려줘")]
# 처리 과정 로그 앞머리 → 단계 이름 (graph.py 노드 이름)
STEPS = {"route": ":material/alt_route: 질문 나누기", "rag": ":material/search: 문서 검색 (RAG)",
         "agent": ":material/build: 도구 사용 (MCP)", "approval": ":material/verified_user: 사람 승인",
         "deny": ":material/block: 차단", "answer": ":material/edit_note: 답변 쓰기",
         "reflect": ":material/fact_check: 스스로 검토 (Reflection)"}


# ── 그림 ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, max_entries=64)
def cutout(path: str) -> Image.Image:
    """흰 배경을 투명하게 하고 정사각형 가운데에 놓는다.
    네 모서리에서 이어진 흰 부분만 지워서, 테두리 안의 하얀 몸(치이카와 등)은 그대로 남는다."""
    im = Image.open(path).convert("RGBA")
    w, h = im.size
    for xy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        if min(im.getpixel(xy)[:3]) > 235:
            ImageDraw.floodfill(im, xy, (255, 255, 255, 0), thresh=60)
    side = max(w, h)
    canvas = Image.new("RGBA", (side, side), (255, 255, 255, 0))
    canvas.paste(im, ((side - w) // 2, (side - h) // 2), im)
    return canvas


def picture(name: str) -> Image.Image | None:
    path = IMG / f"{name}.webp"
    return cutout(str(path)) if path.exists() else None


def art(name: str, width: int, alt: str = "") -> None:
    """그림을 width 픽셀 칸에 원본 해상도 그대로 보여 준다.
    st.image(width=정수)는 그림을 그 픽셀 수로 줄여서 보내기 때문에, 화면 배율이 큰 노트북에서 흐려진다."""
    img = picture(name)
    if img is not None:
        with st.container(width=width):
            st.image(img, width="stretch", alt=alt or name)


st.set_page_config(page_title="먼작귀 에이전트", page_icon=picture("chiikawa"), layout="wide")
AVATAR = {"user": picture("chiikawa"), "assistant": picture("hachiware")}
# 대화 말풍선 옆 캐릭터 그림: 기본 2rem(32px)이 작아서 키운다. 그림은 정사각형 투명 배경이라 잘리지 않는다.
st.html("<style>img[data-testid='stChatMessageAvatarImage']"
        "{width:4.5rem;height:4.5rem;object-fit:contain;border-radius:0}</style>")


# ── API 호출 ─────────────────────────────────────────────────────
class ApiError(Exception):
    pass


def api(method: str, path: str, **kw):
    try:
        r = requests.request(method, f"{API}{path}", timeout=120, **kw)
    except requests.RequestException as e:
        raise ApiError(f"agent-app 에 연결하지 못했습니다: {e}") from e
    if r.status_code >= 400:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise ApiError(f"{r.status_code}: {detail}")
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def load_characters() -> list[dict]:
    return api("GET", "/characters")


def healthy() -> bool:
    try:
        return api("GET", "/health").get("status") == "ok"
    except ApiError:
        return False


# ── 세션 상태 ─────────────────────────────────────────────────────
ss = st.session_state
ss.setdefault("session_id", uuid.uuid4().hex[:8])
ss.setdefault("messages", [])          # [{"role", "content", "trace"}]
ss.setdefault("pending", None)         # 승인 대기 payload
ss.setdefault("queued", None)          # 예시 카드·캐릭터 카드에서 보낼 질문
ss.setdefault("tab", CHAT_TAB)         # 지금 보이는 탭 (st.tabs key)


def uid() -> str:
    """사용자 ID 입력값. 비워 두면 guest (위젯 값을 직접 덮어쓰면 Streamlit 오류가 나므로 읽을 때 정한다)."""
    return (ss.get("user_id") or "").strip() or "guest"


def send(text: str) -> None:
    """질문을 보내고 결과를 대화 기록에 넣는다."""
    ss.messages.append({"role": "user", "content": text})
    try:
        r = api("POST", "/chat", json={"session_id": ss.session_id, "message": text, "user_id": uid()})
    except ApiError as e:
        ss.messages.append({"role": "assistant", "content": f"처리하지 못했습니다. {e}", "trace": []})
        return
    handle(r)


def ask(text: str) -> None:
    """예시·캐릭터 카드 버튼 콜백: 질문을 예약하고 대화 탭으로 넘긴다 (탭 위젯이 그려지기 전에 실행된다)."""
    ss.queued = text
    ss.tab = CHAT_TAB


def handle(r: dict) -> None:
    if r["status"] == "needs_approval":
        ss.pending = r
    else:
        ss.pending = None
        ss.messages.append({"role": "assistant", "content": r["answer"], "trace": r["trace"]})


def resume(approve: bool) -> None:
    try:
        handle(api("POST", "/resume", json={"session_id": ss.session_id, "user_id": uid(), "approve": approve}))
    except ApiError as e:
        ss.pending = None
        ss.messages.append({"role": "assistant", "content": f"처리하지 못했습니다. {e}", "trace": []})


BOLD_END = re.compile(r"([」』\"')\]])\*\*(?=[가-힣A-Za-z0-9])")


def fix_bold(text: str) -> str:
    """'**1화 「청운」**에서' 처럼 닫는 괄호 뒤 ** 에 한글이 바로 붙으면 마크다운이 굵게 처리하지 못한다
    → ** 를 괄호 앞으로 옮긴다 ('**1화 「청운**」에서')."""
    return BOLD_END.sub(r"**\1", text)


def show_trace(trace: list[str]) -> None:
    """에이전트가 거친 단계를 접히는 타임라인으로 보여 준다 (route → rag → agent → answer → reflect)."""
    with st.status(f"처리 과정 · {len(trace)}단계", type="compact", state="complete"):
        for line in trace:
            kind, _, detail = line.partition(": ")
            with st.status(STEPS.get(kind, kind), type="step", state="complete", expanded=True):
                st.caption(detail)


# ── 사이드바 ──────────────────────────────────────────────────────
st.logo(picture("chiikawa"), size="large")
with st.sidebar:
    ok = healthy()
    st.badge("agent-app 연결됨" if ok else "agent-app 연결 안 됨", color="green" if ok else "red",
             icon=":material/check_circle:" if ok else ":material/error:")
    st.text_input("사용자 ID", value="guest", key="user_id", max_chars=40, disabled=bool(ss.pending),
                  help="승인을 기다리는 동안에는 바꿀 수 없습니다.")
    st.caption(f"대화 세션: {ss.session_id}")
    if st.button("새 대화", icon=":material/add_comment:", width="stretch"):
        ss.session_id, ss.messages, ss.pending = uuid.uuid4().hex[:8], [], None
        st.rerun()
    st.space("large")
    art("usagi", 200, "우사기")

# ── 머리말: 제목 + 캐릭터 행렬 ─────────────────────────────────────
st.title("먼작귀 에이전트")
st.caption("기억나는 장면을 말하면 몇 화인지, 캐릭터를 말하면 나온 화와 그 장면 바로가기를 찾아 줘요.")
with st.container(horizontal=True, horizontal_alignment="distribute", gap="small"):
    for name in PARADE:
        art(name, 104)

# key + bind → 탭이 주소(?tab=)에 남고, 버튼 콜백에서 ss.tab 을 바꿔 대화 탭으로 넘길 수 있다
chat_tab, char_tab = st.tabs([CHAT_TAB, CHAR_TAB], key="tab", bind="query-params")

# ── 대화 ─────────────────────────────────────────────────────────
with chat_tab:
    if not ss.messages:
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            st.markdown("안녕! 기억나는 장면을 말해 주면 **몇 화**인지 찾아 줄게. "
                        "캐릭터 이름을 말하면 **나온 화**랑 그 장면 바로가기도 알려 줄게.")
        for col, (img, q) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
            with col.container(border=True, height="stretch"):
                art(img, 120)
                st.markdown(q)
                st.space("stretch")
                st.button("물어보기", key=f"ex-{q}", icon=":material/search:", width="stretch",
                          on_click=ask, args=(q,))
    for m in ss.messages:
        with st.chat_message(m["role"], avatar=AVATAR[m["role"]]):
            st.markdown(fix_bold(m["content"]))
            if m.get("trace"):
                show_trace(m["trace"])
    if ss.pending:
        req = ss.pending["approval"]
        with st.chat_message("assistant", avatar=AVATAR["assistant"]):
            st.warning(req["question"], icon=":material/verified_user:")
            for a in req["actions"]:
                verb = "담기" if a["tool"] == "add_to_watchlist" else "빼기"
                st.markdown(f"- {a['args'].get('ep')}화 시청 목록에서 **{verb}** (사용자 {a['args'].get('user_id')})")
            c1, c2 = st.columns(2)
            if c1.button("승인", type="primary", icon=":material/check:", width="stretch"):
                with st.spinner("실행하는 중"):
                    resume(True)
                st.rerun()
            if c2.button("거절", icon=":material/close:", width="stretch"):
                with st.spinner("취소하는 중"):
                    resume(False)
                st.rerun()

# ── 캐릭터 목록 ───────────────────────────────────────────────────
with char_tab:
    try:
        chars = load_characters()
    except ApiError as e:
        chars = []
        st.error(str(e))
    groups = ["전체"] + list(dict.fromkeys(c["group"] for c in chars))
    pick = st.segmented_control("분류", groups, default="전체", key="group_pick")
    shown = [c for c in chars if pick in (None, "전체") or c["group"] == pick]
    for row in range(0, len(shown), 5):                 # 다섯 개씩 한 줄 → 줄마다 카드 높이를 맞춘다
        for col, c in zip(st.columns(5), shown[row:row + 5]):
            with col.container(border=True, height="stretch"):
                img = picture(c["image"])
                if img is not None:
                    st.image(img, width="stretch", alt=c["name"])
                alias = f" ({', '.join(c['aliases'])})" if c["aliases"] else ""
                st.markdown(f"**{c['name']}**{alias}")
                with st.container(horizontal=True, gap="small"):
                    st.badge(c["group"], color="violet")
                    st.badge(f"총 {c['episode_count']}화 등장", color="blue" if c["episode_count"] else "gray")
                st.space("stretch")
                st.button("나온 화 물어보기", key=f"ask-{c['name']}", icon=":material/movie:", width="stretch",
                          on_click=ask, args=(f"{c['name']} 나온 화 처음 5개만 영상 링크랑 같이 알려줘",))

prompt = st.chat_input("예: 하치와레랑 우사기가 같이 나온 화 알려줘", disabled=bool(ss.pending) or not ok)
if ss.queued and not prompt and not ss.pending and ok:      # 대기 중·연결 끊김이면 보내지 않는다
    prompt, ss.queued = ss.queued, None
if prompt:
    with st.spinner("하치와레가 찾는 중이에요"):
        send(prompt)
    st.rerun()
