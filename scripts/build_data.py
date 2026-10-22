"""
데이터 만들기 — 공식 영상 설명란 + 캐릭터 사전 → SQLite DB + RAG 청크
=====================================================================
입력
  data/raw/videos.jsonl  : scripts/fetch_videos.py 가 받은 AniBox 공식 재생목록 메타데이터 (화마다 한 줄)
  data/characters.json   : 캐릭터 사전 (원어·더빙판 이름, 공식 소개, 줄거리에서 찾을 이름 패턴)
  data/scenes/{화}.json  : scripts/describe_scenes.py 가 영상에서 만든 장면 기록 (있는 화만)
출력
  data/chiikawa.db       : mcp_server.py 가 읽는 DB (episodes · segments · characters · appearances · stories · scenes)
  data/rag_chunks.jsonl  : rag_module.py 가 색인하는 청크 (소제목 줄거리 · 화 · 캐릭터 소개 · 영상 줄거리 · 장면)

등장 기록을 만드는 규칙
  1) 설명란의 "소제목 : 줄거리" 줄에서 캐릭터 이름 패턴을 찾는다 → 소제목 단위 등장 (source=synopsis)
  2) 장면 기록에서 화면에 보인 캐릭터를 넣는다 (source=scene). 장면 기록의 소제목 수가 공식과 같으면
     장면 시각으로 소제목을 정하고, 다르면 화 단위(seg=0)로 넣는다.
  3) 캐릭터 인식 모델(OWLv2, data/vision)이 장면 기록에 없던 캐릭터를 찾았으면 그 장면에 더한다 (source=vision).
  설명란 해시태그는 쓰지 않는다. 128개 영상에 "#치이카와 #하치와레 #토끼 #우사기 #밤만쥬 #쿠리만쥬 #시사" 가
  똑같이 붙어 있어(시사가 처음 나오기 전 화에도 #시사) 그 화에 누가 나왔는지의 근거가 못 된다.

실행:  python scripts/build_data.py
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from import_vision import load_vision, merge

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "videos.jsonl"
CHARS = ROOT / "data" / "characters.json"
DB = ROOT / "data" / "chiikawa.db"
CHUNKS = ROOT / "data" / "rag_chunks.jsonl"
SCENES = ROOT / "data" / "scenes"

EP_RE = re.compile(r"(\d+)화_(.+)$")
SEG_RE = re.compile(r"^\s*(.+?)\s+:\s+(.+)$")          # "단단한 푸딩 : 치이카와는 ..."
VIDEO_URL = "https://www.youtube.com/watch?v={}"


def load_videos() -> dict[int, dict]:
    """화 번호 → 영상. 같은 화가 두 번 올라온 경우(155·156화) 설명이 긴 쪽을 쓴다."""
    eps: dict[int, dict] = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        v = json.loads(line)
        m = EP_RE.search(v.get("title") or "")
        if not m:
            continue                                     # 성우 인터뷰·1시간 노래 영상 등은 뺀다
        ep = int(m.group(1))
        v["ep_title"] = m.group(2).strip()
        if ep not in eps or len(v.get("description") or "") > len(eps[ep].get("description") or ""):
            eps[ep] = v
    return dict(sorted(eps.items()))


def compile_patterns(chars: list[dict]) -> list[tuple[dict, re.Pattern]]:
    """이름 앞에 한글이 붙어 있으면 다른 낱말의 일부로 보고 건너뛴다 (예: '산토끼')."""
    return [(c, re.compile("(?<![가-힣])(?:" + "|".join(c["text_patterns"]) + ")")) for c in chars]


def find_chars(text: str, patterns: list[tuple[dict, re.Pattern]]) -> list[str]:
    found = []
    # '검은 별똥별'은 '별똥별'보다 먼저 찾고, 찾은 자리는 지워서 '별똥별'로 두 번 세지 않는다
    for c, pat in sorted(patterns, key=lambda x: -max(len(p) for p in x[0]["text_patterns"])):
        if pat.search(text):
            found.append(c["id"])
            text = pat.sub(" ", text)
    return found


def parse(video: dict) -> list[dict]:
    """설명란 → 소제목 목록 (해시태그 줄은 건너뛴다)."""
    desc = video.get("description") or ""
    segs = [{"title": m.group(1).strip(), "synopsis": m.group(2).strip()}
            for line in desc.splitlines() if (m := SEG_RE.match(line)) and not line.lstrip().startswith("#")]
    if not segs:                                         # 줄거리가 없는 화: 제목 하나를 소제목으로
        segs = [{"title": video["ep_title"], "synopsis": ""}]
    return segs


def load_scenes() -> dict[int, dict]:
    """화 번호 → describe_scenes.py 결과. 아직 없는 화는 빠진다."""
    return {int(p.stem): json.loads(p.read_text(encoding="utf-8")) for p in sorted(SCENES.glob("[0-9]*.json"))}


def mmss(sec: float) -> str:
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def scene_seg(rec: dict, n_official: int, start: float) -> int:
    """장면 시각(잘린 영상 기준)이 들어가는 소제목 번호. 소제목 수가 공식과 다르면 0(화 단위)."""
    if len(rec["segments"]) != n_official:
        return 0
    for i, s in enumerate(rec["segments"], start=1):
        if start < s["end"]:
            return i
    return n_official


def main() -> None:
    chars = json.loads(CHARS.read_text(encoding="utf-8"))
    patterns = compile_patterns(chars)
    by_id = {c["id"]: c for c in chars}
    name_map = {n: c["id"] for c in chars for n in [c["name"], *c["aliases"]]}
    videos = load_videos()
    scenes = load_scenes()

    DB.unlink(missing_ok=True)
    con = sqlite3.connect(DB)
    con.executescript("""
        CREATE TABLE characters (id TEXT PRIMARY KEY, sort INTEGER, name TEXT, aliases TEXT, orig TEXT, grp TEXT,
                                 intro TEXT, intro_src TEXT, va_jp TEXT, va_kr TEXT, motif TEXT, first_manga TEXT,
                                 img TEXT, doc TEXT);
        CREATE TABLE episodes (ep INTEGER PRIMARY KEY, video_id TEXT, title TEXT, duration INTEGER, upload_date TEXT);
        CREATE TABLE segments (ep INTEGER, seg INTEGER, title TEXT, synopsis TEXT, PRIMARY KEY (ep, seg));
        CREATE TABLE appearances (ep INTEGER, seg INTEGER, char_id TEXT, source TEXT,
                                  PRIMARY KEY (ep, seg, char_id));
        CREATE INDEX idx_app_char ON appearances (char_id);
        CREATE TABLE stories (ep INTEGER PRIMARY KEY, summary TEXT);
        CREATE TABLE scenes (ep INTEGER, idx INTEGER, start REAL, end REAL, place TEXT, description TEXT,
                             quote TEXT, characters TEXT, keywords TEXT, PRIMARY KEY (ep, idx));
    """)
    con.executemany("INSERT INTO characters VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(c["id"], i, c["name"], json.dumps(c["aliases"], ensure_ascii=False), c["orig"], c["grp"],
                      c["intro"], c["intro_src"], c["va_jp"], c["va_kr"], c["motif"], c["first_manga"],
                      c["img"], c["doc"]) for i, c in enumerate(chars)])

    chunks: list[dict] = []
    for ep, v in videos.items():
        con.execute("INSERT INTO episodes VALUES (?,?,?,?,?)",
                    (ep, v["id"], v["ep_title"], v.get("duration"), v.get("upload_date")))
        segs = parse(v)
        for i, s in enumerate(segs, start=1):
            con.execute("INSERT INTO segments VALUES (?,?,?,?)", (ep, i, s["title"], s["synopsis"]))
            ids = find_chars(f"{s['title']} {s['synopsis']}", patterns) if s["synopsis"] else []
            for cid in ids:
                con.execute("INSERT OR IGNORE INTO appearances VALUES (?,?,?,?)", (ep, i, cid, "synopsis"))
            if s["synopsis"]:
                names = ", ".join(by_id[c]["name"] for c in ids) or "확인 안 됨"
                chunks.append({"kind": "segment", "ep": ep, "seg": i, "source": f"{ep}화",
                               "text": f"{ep}화 「{s['title']}」 줄거리: {s['synopsis']} (등장: {names})"})
        # 줄거리도 장면 기록도 없는 화는 제목만이라도 찾을 수 있게 (영상 줄거리가 있는 화는 그쪽이 대신한다.
        #  짧은 '줄거리 없음' 청크가 의미 검색에서 아무 질문에나 걸려 장면 청크를 밀어내기 때문)
        if not any(s["synopsis"] for s in segs) and ep not in scenes:
            chunks.append({"kind": "episode", "ep": ep, "seg": 0, "source": f"{ep}화",
                           "text": f"{ep}화 「{v['ep_title']}」 (공식 줄거리 없음)"})

        rec = scenes.get(ep)
        if rec is None:
            continue
        rec = merge(rec, load_vision(ep))
        # 영상에서 본 등장인물 · 영상 줄거리 · 장면 (시각은 원본 기준 → 유튜브 &t= 로 바로 이동)
        con.execute("INSERT INTO stories VALUES (?,?)", (ep, rec["summary"]))
        story_ids = []
        for i, sc in enumerate(rec["scenes"], start=1):
            ids = [name_map[n] for n in (re.sub(r"\(.*?\)", "", x).strip() for x in sc["characters"]) if n in name_map]
            for cid in ids:
                source = "vision" if by_id[cid]["name"] in sc.get("vision_added", []) else "scene"
                con.execute("INSERT OR IGNORE INTO appearances VALUES (?,?,?,?)",
                            (ep, scene_seg(rec, len(segs), sc["start"]), cid, source))
            story_ids += [c for c in ids if c not in story_ids]
            names = [by_id[c]["name"] for c in ids]
            con.execute("INSERT INTO scenes VALUES (?,?,?,?,?,?,?,?,?)",
                        (ep, i, sc["orig_start"], sc["orig_end"], sc["place"], sc["description"], sc["quote"],
                         json.dumps(names, ensure_ascii=False), json.dumps(sc["keywords"], ensure_ascii=False)))
            quote = f" 대사: \"{sc['quote']}\"" if sc["quote"] else ""
            chunks.append({"kind": "scene", "ep": ep, "seg": i, "source": f"{ep}화 {mmss(sc['orig_start'])}",
                           "text": f"{ep}화 「{v['ep_title']}」 {mmss(sc['orig_start'])}~{mmss(sc['orig_end'])} 장면"
                                   f" ({sc['place']}): {sc['description']}{quote}"
                                   f" (등장: {', '.join(names) or '확인 안 됨'} / 검색어: {', '.join(sc['keywords'])})"
                                   f" 영상: {VIDEO_URL.format(rec['video_id'])}&t={int(sc['orig_start'])}s"})
        for n in rec["characters"]:                     # 화 전체 등장인물 중 어느 장면에도 안 걸린 캐릭터는 화 단위로
            cid = name_map.get(re.sub(r"\(.*?\)", "", n).strip())
            if cid and cid not in story_ids:
                source = "vision" if n in rec["vision_added"] else "scene"
                con.execute("INSERT OR IGNORE INTO appearances VALUES (?,?,?,?)", (ep, 0, cid, source))
                story_ids.append(cid)
        names = ", ".join(by_id[c]["name"] for c in story_ids) or "확인 안 됨"
        chunks.append({"kind": "story", "ep": ep, "seg": 0, "source": f"{ep}화 영상 줄거리",
                       "text": f"{ep}화 「{v['ep_title']}」 영상 줄거리: {rec['summary']} (등장: {names})"})

    for c in chars:
        alias = f" (원어·별명: {', '.join(c['aliases'])})" if c["aliases"] else ""
        parts = [f"캐릭터 소개: {c['name']}{alias} — {c['orig']}. 분류: {c['grp']}.",
                 f"{c['intro_src']}: {c['intro']}"]
        if c["va_jp"] or c["va_kr"]:
            parts.append(f"성우: 일본 {c['va_jp'] or '표기 없음'}, 한국 {c['va_kr'] or '표기 없음'}.")
        if c["motif"]:
            parts.append(f"모티브: {c['motif']}.")
        if c["first_manga"]:
            parts.append(f"원작 만화 첫 등장: {c['first_manga']}.")
        chunks.append({"kind": "character", "ep": 0, "seg": 0, "source": f"등장인물: {c['name']}",
                       "names": [c["name"], *c["aliases"]], "text": " ".join(parts)})
    con.commit()

    with CHUNKS.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    n_app = con.execute("SELECT COUNT(*) FROM appearances").fetchone()[0]
    n_seg = con.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
    n_scene = con.execute("SELECT COUNT(*) FROM scenes").fetchone()[0]
    print(f"화 {len(videos)}개 · 소제목 {n_seg}개 · 등장 기록 {n_app}개 · RAG 청크 {len(chunks)}개"
          f" · 장면 기록 {len(scenes)}화 {n_scene}장면")
    for cid, n in con.execute("SELECT char_id, COUNT(DISTINCT ep) FROM appearances GROUP BY char_id "
                              "ORDER BY 2 DESC").fetchall():
        print(f"  {by_id[cid]['name']}: {n}화")
    con.close()


if __name__ == "__main__":
    main()
