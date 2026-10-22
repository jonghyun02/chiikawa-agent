"""
먼작귀 에이전트 — MCP 서버 (Day2 FastMCP + SQLite 도구, Day5 HTTP 전환)
======================================================================
데이터는 scripts/build_data.py 가 만든 SQLite 두 개다.
  - data/chiikawa.db    : 화·소제목·캐릭터·등장 기록 (읽기 전용으로 연다 — Day2 최소 권한)
  - WATCHLIST_DB        : 사용자별 시청 목록 (이 서버에서 유일하게 쓰기가 일어나는 곳)

[도구 분류 — graph.py 의 승인(HITL) 설계 근거]
  읽기 : list_characters, find_character, find_episodes, get_episode, search_titles, get_watchlist
  쓰기 : add_to_watchlist, remove_from_watchlist   → 사람 승인 후에만 실행 (graph.SENSITIVE)

[Day2 보안 원칙 적용]
  - 접근 범위: 데이터 DB 는 mode=ro 로 열어 구조 변경·삭제가 원천적으로 불가능하다
  - 위험한 동작 제한: SQL 은 전부 파라미터 바인딩, 모델이 SQL 을 직접 쓰는 도구는 두지 않는다
  - 오류는 데이터로: 예외 대신 {"error": ...} 를 돌려줘 LLM 이 사용자에게 설명하게 한다 (Day5)

실행 방식은 환경변수 MCP_TRANSPORT 로 고른다.
  - http  (기본): 상시 대기 서버, http://localhost:8001/mcp   (compose 의 mcp-server 컨테이너)
  - stdio       : 클라이언트가 자식 프로세스로 띄움            (오프라인 테스트용)
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from fastmcp import FastMCP

mcp = FastMCP("chiikawa-tools-server")

BASE = Path(__file__).parent
DATA_DB = Path(os.environ.get("DATA_DB", BASE / "data" / "chiikawa.db"))
WATCHLIST_DB = Path(os.environ.get("WATCHLIST_DB", BASE / "data" / "watchlist.sqlite"))
VIDEO_URL = "https://www.youtube.com/watch?v={}"
_LOCK = threading.Lock()        # 시청 목록 "확인 → 쓰기"를 한 덩어리로 (동시 요청 이중 등록 방지)


# ── DB 연결 ───────────────────────────────────────────────────────
# 호출마다 연결을 열고 닫는다. HTTP 모드에서 여러 요청이 동시에 와도 연결을 공유하지 않는다.
@contextmanager
def _data() -> Iterator[sqlite3.Connection]:
    """데이터 DB 를 읽기 전용(mode=ro)으로 연다."""
    con = sqlite3.connect(f"file:{DATA_DB.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


@contextmanager
def _watch() -> Iterator[sqlite3.Connection]:
    """시청 목록 DB. 없으면 만든다 (compose 에서는 볼륨 /data 에 저장되어 재시작해도 남는다).
    블록이 정상으로 끝나면 commit, 예외면 rollback 한 뒤 닫는다."""
    WATCHLIST_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(WATCHLIST_DB)
    con.row_factory = sqlite3.Row
    try:
        con.execute("CREATE TABLE IF NOT EXISTS watchlist (user_id TEXT, ep INTEGER, memo TEXT, added_at TEXT,"
                    " PRIMARY KEY (user_id, ep))")
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── 캐릭터 이름 해석 ──────────────────────────────────────────────
def _norm(s: str) -> str:
    return "".join(s.split()).lower()


def _characters(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute("SELECT * FROM characters ORDER BY sort").fetchall()
    return [{**dict(r), "aliases": json.loads(r["aliases"])} for r in rows]


def _resolve(con: sqlite3.Connection, name: str) -> tuple[dict | None, list[str]]:
    """더빙판 이름·원어 이름·별명 어느 것으로 불러도 같은 캐릭터를 찾는다.
    정확히 일치 → 부분 일치 순서. 못 찾으면 (None, 후보 이름 목록)."""
    key = _norm(name)
    chars = _characters(con)
    for c in chars:
        if key in {_norm(n) for n in [c["name"], *c["aliases"]]}:
            return c, []
    partial = [c for c in chars if any(key in _norm(n) or _norm(n) in key for n in [c["name"], *c["aliases"]])]
    if len(partial) == 1:
        return partial[0], []
    return None, [c["name"] for c in (partial or chars)][:10]


def _first_scene(con: sqlite3.Connection, ep: int, name: str) -> float | None:
    """그 화에서 캐릭터가 처음 보이는 장면의 시각(원본 영상 기준 초). 장면 기록에 없으면 None."""
    for start, chars in con.execute("SELECT start, characters FROM scenes WHERE ep = ? ORDER BY idx", (ep,)):
        if name in json.loads(chars or "[]"):
            return start
    return None


def _episode_links(con: sqlite3.Connection, eps: list[int], name: str | None = None) -> list[dict]:
    """화 번호 목록 → [{ep, title, url, scene_url?}]. name 을 주면 그 캐릭터가 처음 나오는 장면 바로가기도 붙인다."""
    out = []
    for ep in eps:
        row = con.execute("SELECT title, video_id FROM episodes WHERE ep = ?", (ep,)).fetchone()
        if row is None:
            continue
        item = {"ep": ep, "title": row["title"], "url": VIDEO_URL.format(row["video_id"])}
        t = _first_scene(con, ep, name) if name else None
        if t is not None:
            item["scene_url"] = f"{item['url']}&t={int(t)}s"
        out.append(item)
    return out


def _episode_chars(con: sqlite3.Connection, ep: int) -> dict[int, list[str]]:
    """{소제목 번호: [캐릭터 이름, ...]}. 0번은 소제목을 특정할 수 없는 화 단위 등장(장면 기록·캐릭터 인식)."""
    rows = con.execute("SELECT a.seg, c.name FROM appearances a JOIN characters c ON c.id = a.char_id "
                       "WHERE a.ep = ? ORDER BY a.seg, c.sort", (ep,)).fetchall()
    out: dict[int, list[str]] = {}
    for r in rows:
        if r["name"] not in out.setdefault(r["seg"], []):
            out[r["seg"]].append(r["name"])
    return out


# ── 읽기 도구 ─────────────────────────────────────────────────────
@mcp.tool
def list_characters(group: str = "") -> list[dict]:
    """등장인물 목록과 각자 나온 화 수를 돌려준다. group 으로 거를 수 있다
    (주연 / 조연 / 단역 먼작귀 / 갑옷 씨들 / 극장판 / 그 밖의 인물). (읽기 전용)"""
    with _data() as con:
        counts = dict(con.execute("SELECT char_id, COUNT(DISTINCT ep) FROM appearances GROUP BY char_id").fetchall())
        return [{"name": c["name"], "aliases": c["aliases"], "group": c["grp"], "intro": c["intro"],
                 "image": c["img"], "episode_count": counts.get(c["id"], 0)}
                for c in _characters(con) if not group or c["grp"] == group]


@mcp.tool
def find_character(name: str) -> dict:
    """캐릭터 이름(더빙판 '가르마' 또는 원어 '하치와레' 등)으로 프로필과 나온 화 목록을 조회한다. (읽기 전용)
    episodes 에는 화마다 제목·영상 주소(url)와, 그 캐릭터가 처음 나오는 장면 바로가기(scene_url)가 있다 (앞 40화)."""
    with _data() as con:
        c, candidates = _resolve(con, name)
        if c is None:
            return {"error": f"'{name}' 캐릭터를 찾지 못했습니다.", "candidates": candidates}
        eps = [r[0] for r in con.execute("SELECT DISTINCT ep FROM appearances WHERE char_id = ? ORDER BY ep",
                                         (c["id"],)).fetchall()]
        out = {"name": c["name"], "aliases": c["aliases"], "original_name": c["orig"], "group": c["grp"],
               "intro": c["intro"], "intro_source": c["intro_src"], "voice_jp": c["va_jp"],
               "voice_kr": c["va_kr"], "episode_count": len(eps),
               "first_episode": eps[0] if eps else None, "episodes": _episode_links(con, eps[:40], c["name"])}
        if not eps:
            # 기록이 없다는 것과 안 나왔다는 것은 다르다 — 모델이 "안 나온다"고 단정하지 않게 알려 준다
            out["note"] = ("공식 줄거리와 영상 장면 기록에서 이 캐릭터를 찾지 못했습니다. "
                           "극장판에만 나왔거나 기록에서 빠졌을 수 있습니다.")
        return out


@mcp.tool
def find_episodes(characters: list[str], exclude: list[str] | None = None, together: bool = True,
                  limit: int = 15) -> dict:
    """캐릭터가 나온 화를 찾는다. (읽기 전용)
    characters : 찾을 캐릭터 이름 목록 (더빙판·원어 이름 모두 가능)
    exclude    : 나오지 않아야 하는 캐릭터 이름 목록
    together   : True 면 모두 함께 나온 화, False 면 한 명이라도 나온 화
    limit      : 최대 몇 화까지 돌려줄지 (전체 개수는 total 에 따로 담긴다)"""
    with _data() as con:
        want, unknown = [], []
        for n in characters:
            c, cand = _resolve(con, n)
            (want.append(c) if c else unknown.append({"name": n, "candidates": cand}))
        bad = []
        for n in exclude or []:
            c, cand = _resolve(con, n)
            (bad.append(c) if c else unknown.append({"name": n, "candidates": cand}))
        if unknown:
            return {"error": "모르는 캐릭터 이름이 있습니다.", "unknown": unknown}
        if not want:
            return {"error": "찾을 캐릭터를 한 명 이상 주세요."}
        sets = [{r[0] for r in con.execute("SELECT DISTINCT ep FROM appearances WHERE char_id = ?", (c["id"],))}
                for c in want]
        eps = set.intersection(*sets) if together else set.union(*sets)
        for c in bad:
            eps -= {r[0] for r in con.execute("SELECT DISTINCT ep FROM appearances WHERE char_id = ?", (c["id"],))}
        result = []
        for item in _episode_links(con, sorted(eps)[:max(1, min(limit, 50))], want[0]["name"]):
            seg_chars = _episode_chars(con, item["ep"])
            segs = con.execute("SELECT seg, title FROM segments WHERE ep = ? ORDER BY seg", (item["ep"],)).fetchall()
            result.append({**item, "segments": [{"title": s["title"], "characters": seg_chars.get(s["seg"], [])}
                                                for s in segs],
                           "episode_characters": seg_chars.get(0, [])})
        return {"characters": [c["name"] for c in want], "exclude": [c["name"] for c in bad],
                "together": together, "total": len(eps), "episodes": result}


@mcp.tool
def get_episode(ep: int) -> dict:
    """화 번호로 소제목·공식 줄거리·영상 줄거리·장면 목록(시각·바로가기 주소)·등장인물을 조회한다. (읽기 전용)"""
    with _data() as con:
        row = con.execute("SELECT * FROM episodes WHERE ep = ?", (ep,)).fetchone()
        if row is None:
            last = con.execute("SELECT MAX(ep) FROM episodes").fetchone()[0]
            return {"error": f"{ep}화는 없습니다. 1화부터 {last}화까지 있습니다."}
        seg_chars = _episode_chars(con, ep)
        segs = con.execute("SELECT seg, title, synopsis FROM segments WHERE ep = ? ORDER BY seg", (ep,)).fetchall()
        story = con.execute("SELECT summary FROM stories WHERE ep = ?", (ep,)).fetchone()
        scenes = con.execute("SELECT * FROM scenes WHERE ep = ? ORDER BY idx", (ep,)).fetchall()
        url = VIDEO_URL.format(row["video_id"])
        return {"ep": ep, "title": row["title"], "url": url,
                "segments": [{"title": s["title"], "synopsis": s["synopsis"],
                              "characters": seg_chars.get(s["seg"], [])} for s in segs],
                "episode_characters": seg_chars.get(0, []),
                # 영상에서 기록한 줄거리·장면 (장면 기록이 아직 없는 화는 빈 값)
                "story": story["summary"] if story else "",
                "scenes": [{"time": f"{int(s['start']) // 60}:{int(s['start']) % 60:02d}",
                            "url": f"{url}&t={int(s['start'])}s", "place": s["place"],
                            "description": s["description"], "quote": s["quote"],
                            "characters": json.loads(s["characters"])} for s in scenes]}


@mcp.tool
def search_titles(keyword: str, limit: int = 20) -> dict:
    """소제목이나 줄거리에 낱말이 그대로 들어간 화를 찾는다 (정확한 낱말 검색). (읽기 전용)
    장면 묘사처럼 표현이 다를 수 있는 질문은 RAG 검색(search_docs)이 더 낫다."""
    kw = f"%{keyword.strip()}%"
    with _data() as con:
        rows = con.execute("SELECT s.ep, s.title, e.video_id FROM segments s JOIN episodes e ON e.ep = s.ep "
                           "WHERE s.title LIKE ? OR s.synopsis LIKE ? ORDER BY s.ep, s.seg LIMIT ?",
                           (kw, kw, max(1, min(limit, 50)))).fetchall()
    # 빈 리스트는 MCP 전송에서 빈 문자열이 되어 '결과 없음'과 '오류'를 구분하기 어렵다 → 개수를 함께 돌려준다
    return {"keyword": keyword, "count": len(rows),
            "results": [{"ep": r["ep"], "segment_title": r["title"], "url": VIDEO_URL.format(r["video_id"])}
                        for r in rows]}


@mcp.tool
def get_watchlist(user_id: str) -> dict:
    """사용자의 시청 목록을 조회한다. (읽기 전용)"""
    with _watch() as wcon, _data() as con:
        rows = wcon.execute("SELECT ep, memo, added_at FROM watchlist WHERE user_id = ? ORDER BY ep",
                            (user_id,)).fetchall()
        items = []
        for r in rows:
            e = con.execute("SELECT title, video_id FROM episodes WHERE ep = ?", (r["ep"],)).fetchone()
            items.append({"ep": r["ep"], "title": e["title"] if e else "", "memo": r["memo"],
                          "url": VIDEO_URL.format(e["video_id"]) if e else "", "added_at": r["added_at"]})
    return {"user_id": user_id, "count": len(items), "items": items}


# ── 쓰기 도구 (승인 필요) ─────────────────────────────────────────
def _check_user(user_id: str) -> str | None:
    if not user_id or len(user_id) > 40:
        return "사용자 ID 는 1~40자여야 합니다."
    return None


@mcp.tool
def add_to_watchlist(user_id: str, ep: int, memo: str = "") -> dict:
    """화를 사용자의 시청 목록에 담는다. (쓰기 — 사람 승인 후에만 실행되는 도구)"""
    if err := _check_user(user_id):
        return {"error": err}
    with _data() as con:
        if con.execute("SELECT 1 FROM episodes WHERE ep = ?", (ep,)).fetchone() is None:
            return {"error": f"{ep}화는 없습니다."}
    with _LOCK, _watch() as wcon:
        if wcon.execute("SELECT 1 FROM watchlist WHERE user_id = ? AND ep = ?", (user_id, ep)).fetchone():
            return {"status": "already", "ep": ep, "message": f"{ep}화는 이미 시청 목록에 있습니다."}
        wcon.execute("INSERT INTO watchlist VALUES (?, ?, ?, ?)",
                     (user_id, ep, memo[:100], datetime.now(timezone.utc).isoformat(timespec="seconds")))
        total = wcon.execute("SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)).fetchone()[0]
    return {"status": "added", "ep": ep, "watchlist_size": total}


@mcp.tool
def remove_from_watchlist(user_id: str, ep: int) -> dict:
    """화를 사용자의 시청 목록에서 뺀다. (쓰기 — 사람 승인 후에만 실행되는 도구)"""
    if err := _check_user(user_id):
        return {"error": err}
    with _LOCK, _watch() as wcon:
        cur = wcon.execute("DELETE FROM watchlist WHERE user_id = ? AND ep = ?", (user_id, ep))
        total = wcon.execute("SELECT COUNT(*) FROM watchlist WHERE user_id = ?", (user_id,)).fetchone()[0]
    if cur.rowcount == 0:
        return {"status": "not_found", "ep": ep, "message": f"{ep}화는 시청 목록에 없습니다."}
    return {"status": "removed", "ep": ep, "watchlist_size": total}


if __name__ == "__main__":
    transport = os.environ.get("MCP_TRANSPORT", "http")
    if transport == "stdio":
        # stdio 에서는 stdout 이 JSON-RPC 전용이므로 로그는 stderr 로만 보낸다 (Day5 이슈 3)
        print("[mcp-server] stdio transport 로 시작", file=sys.stderr, flush=True)
        mcp.run(transport="stdio", show_banner=False)
    else:
        host = os.environ.get("MCP_SERVER_HOST", "127.0.0.1")
        port = int(os.environ.get("MCP_SERVER_PORT", "8001"))
        print(f"[mcp-server] HTTP transport 로 시작 — http://{host}:{port}/mcp", file=sys.stderr, flush=True)
        mcp.run(transport="http", host=host, port=port)
