"""
장면 문서 만들기 — data/scenes/{화}.json → 사람이 읽는 마크다운 (옵시디언에서 바로 열림)
=====================================================================
출력
  docs/scenes/{화}화 {제목}.md : 화마다 한 장 (영상 줄거리 · 소제목 · 장면 표 · 받아쓴 대사)
  docs/scenes/00 장면 문서 목차.md : 전체 목차 (화 · 등장인물 · 장면 수)

시각은 원본 영상 기준이고, 시각을 누르면 유튜브 그 장면으로 바로 간다.

실행:  python scripts/build_scene_docs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from import_vision import load_vision, merge

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "data" / "scenes"
DOCS = ROOT / "docs" / "scenes"


def mmss(sec: float) -> str:
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def doc_name(rec: dict) -> str:
    return re.sub(r'[\\/:*?"<>|#^\[\]]', " ", f"{rec['ep']:03d}화 {rec['title']}").strip()


def at(rec: dict, sec: float) -> str:
    """원본 시각 → 유튜브 바로가기 링크."""
    return f"[{mmss(sec)}]({rec['url']}&t={int(sec)}s)"


def cell(s: str) -> str:
    return s.replace("|", "/").replace("\n", " ")


def mark(names: list[str], added: list[str]) -> str:
    """캐릭터 인식 모델이 더한 이름에는 † 를 붙인다."""
    return ", ".join(f"{n}†" if n in added else n for n in names)


def episode_doc(rec: dict) -> str:
    lines = ["---", f"화: {rec['ep']}", f"제목: \"{rec['title']}\"", f"영상: {rec['url']}",
             "등장: [" + ", ".join(rec["characters"]) + "]", "tags: [먼작귀, 장면기록]", "---", "",
             f"# {rec['ep']}화 「{rec['title']}」", "",
             "> [!info] 이 문서는 영상 캡처(1초 간격)와 대사 자동 인식을 바탕으로 만든 기록이다. "
             "공식 자료가 아니라 틀린 곳이 있을 수 있다. 시각은 원본 영상 기준이고, 누르면 그 장면으로 간다.", ""]
    if rec.get("vision_added"):
        lines += [f"> [!note] † 는 캐릭터 인식 모델(OWLv2)이 화면에서 찾아 더한 등장인물: "
                  f"{', '.join(rec['vision_added'])}", ""]
    lines += ["## 영상 줄거리", "", rec["summary"], ""]
    if rec["official_teasers"]:
        lines += ["## 공식 예고 줄거리 (설명란)", ""]
        lines += [f"- **{t['title']}**: {t['teaser']}" for t in rec["official_teasers"]]
        lines.append("")
    lines += ["## 소제목", "", "| 시각 | 소제목 | 줄거리 |", "|---|---|---|"]
    lines += [f"| {at(rec, s['orig_start'])} | {cell(s['title'])} | {cell(s['story'])} |" for s in rec["segments"]]
    lines += ["", "## 장면", "", "| 시각 | 장소 | 등장 | 장면 | 대사 | 검색어 |", "|---|---|---|---|---|---|"]
    for s in rec["scenes"]:
        lines.append(f"| {at(rec, s['orig_start'])}~{mmss(s['orig_end'])} | {cell(s['place'])} | "
                     f"{mark(s['characters'], s.get('vision_added', []))} | {cell(s['description'])} | {cell(s['quote'])} | "
                     f"{', '.join(s['keywords'])} |")
    if rec["unknown_characters"]:
        lines += ["", "## 이름을 확인하지 못한 캐릭터", ""] + [f"- {u}" for u in rec["unknown_characters"]]
    lines += ["", "## 받아쓴 대사 (자동 인식, 틀릴 수 있음)", ""]
    lines += [f"- {at(rec, d['orig'][0])} {d['text']}" for d in rec["dialogue"]] or ["- (인식된 대사 없음)"]
    return "\n".join(lines) + "\n"


def index_doc(recs: list[dict]) -> str:
    lines = ["---", "tags: [먼작귀, 장면기록, 목차]", "---", "", "# 먼작귀 장면 문서 목차", "",
             f"영상에서 장면을 기록한 화: **{len(recs)}화**, 장면 **{sum(len(r['scenes']) for r in recs)}개**.", "",
             "| 화 | 제목 | 등장 | 장면 수 |", "|---|---|---|---|"]
    lines += [f"| {r['ep']} | [[{doc_name(r)}\\|{cell(r['title'])}]] | {mark(r['characters'], r.get('vision_added', []))} | "
              f"{len(r['scenes'])} |" for r in recs]
    return "\n".join(lines) + "\n"


def main() -> None:
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(SCENES.glob("[0-9]*.json"))]
    recs = [merge(r, load_vision(r["ep"])) for r in recs]          # 캐릭터 인식 결과를 더한다 (†)
    DOCS.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (DOCS / f"{doc_name(r)}.md").write_text(episode_doc(r), encoding="utf-8")
    (DOCS / "00 장면 문서 목차.md").write_text(index_doc(recs), encoding="utf-8")
    print(f"{len(recs)}화 문서 → {DOCS}")


if __name__ == "__main__":
    main()
