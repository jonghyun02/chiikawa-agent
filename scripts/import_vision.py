"""
영상 캐릭터 인식 결과 가져오기 — preprocess/charvision/{파일 번호}.json → data/vision/{화}.json
=====================================================================
C:/projects/yt-dlp/preprocess/recognize.py (OWLv2, Intel GPU) 가 잘린 영상 1초마다 남긴 캐릭터별 유사도에서
장면 기록(data/scenes)에 없던 등장을 찾고, 그중 검수(data/vision/review.json)에서 맞다고 확인한 것만
화별로 정리한다. build_data.py · build_scene_docs.py 가 merge() 로 장면 기록에 더한다 (지우지는 않는다).

판정 기준: 유사도 0.85 이상 · 2등 캐릭터와 차이 0.03 이상인 상자가 있는 프레임이 한 화에 2개 이상.
  매 화 끝 '계속/끝' 카드(분홍 차를 탄 가르마)를 잡지 않게 마지막 3초는 뺀다.
정확도 (2026-10-22): 기록과 모델이 같이 본 캐릭터는 8화 기준 96% 일치했지만, 기록에 없던 등장 118건을
  잘라 직접 확인하니 맞은 것은 20건(17%)뿐이었다 → 모델 결과는 후보로만 쓰고, 확인한 것만 반영한다.

실행:  python scripts/import_vision.py      (recognize.py 다음, build_data.py 전에)
       python scripts/import_vision.py --candidates   검수할 후보 목록만 출력
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from describe_scenes import PRE, clean_to_orig, load_cuts

ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "data" / "scenes"
VISION = ROOT / "data" / "vision"
CHARVISION = PRE / "charvision"
REVIEW = VISION / "review.json"
SIM, MARGIN, MIN_FRAMES = 0.85, 0.03, 2
END_CARD = 3.0                     # 마지막 몇 초를 '계속/끝' 카드로 보고 뺄지
MODEL = "OWLv2 (google/owlv2-base-patch16-ensemble) · OpenVINO Intel GPU · 실제 프레임에서 고른 기준 상자"


def load_vision(ep: int) -> dict | None:
    p = VISION / f"{ep:03d}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def merge(rec: dict, vis: dict | None) -> dict:
    """장면 기록에 인식 결과를 더한 사본. 장면마다 그 시간 안에 보인 캐릭터를 더하고 vision_added 에 따로 적는다."""
    rec = copy.deepcopy(rec)
    if not vis:
        return rec
    for s in rec["scenes"]:
        added = [n for n, v in vis["characters"].items()
                 if n not in s["characters"] and any(s["start"] <= t < s["end"] + 0.5 for t in v["clean"])]
        s["characters"] += added
        s["vision_added"] = added
    rec["vision_added"] = [n for n in vis["characters"] if n not in rec["characters"]]
    rec["characters"] += rec["vision_added"]
    return rec


def seen_frames(src: Path) -> dict[str, list[float]]:
    """캐릭터 → 판정 기준을 넘은 프레임 시각(잘린 영상 기준). 끝 카드 구간은 뺀다."""
    frames = json.loads(src.read_text(encoding="utf-8"))["frames"]
    last = frames[-1]["t"] if frames else 0
    seen = defaultdict(list)
    for f in frames:
        if f["t"] > last - END_CARD:
            continue
        for name, (sim, margin, _box) in f["hits"].items():
            if sim >= SIM and margin >= MARGIN:
                seen[name].append(f["t"])
    return {n: ts for n, ts in seen.items() if len(ts) >= MIN_FRAMES}


def main() -> None:
    cuts = load_cuts()
    VISION.mkdir(parents=True, exist_ok=True)
    accepted = json.loads(REVIEW.read_text(encoding="utf-8"))["accepted"] if REVIEW.exists() else {}
    candidates, used, n = [], Counter(), 0
    for p in sorted(SCENES.glob("[0-9]*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        src = CHARVISION / f"{rec['file_key']}.json"
        if not src.exists():
            continue
        new = {name: ts for name, ts in seen_frames(src).items() if name not in rec["characters"]}
        candidates += [{"ep": rec["ep"], "key": rec["file_key"], "name": name} for name in new]
        chars = {name: {"frames": len(ts), "clean": ts,
                        "orig_first": round(clean_to_orig(cuts[rec["file_key"]], ts[0]), 1)}
                 for name, ts in new.items() if rec["ep"] in accepted.get(name, [])}
        (VISION / f"{rec['ep']:03d}.json").write_text(json.dumps(
            {"ep": rec["ep"], "file_key": rec["file_key"], "model": MODEL,
             "rule": {"similarity": SIM, "margin": MARGIN, "min_frames": MIN_FRAMES, "end_card": END_CARD,
                      "review": "data/vision/review.json 에서 확인한 것만"}, "characters": chars},
            ensure_ascii=False, indent=1), encoding="utf-8")
        used.update(chars.keys())
        n += 1
    if "--candidates" in sys.argv:
        print(json.dumps(candidates, ensure_ascii=False))
        return
    print(f"{n}화 · 모델이 찾은 새 등장 후보 {len(candidates)}건 · 검수 통과로 더한 것 {sum(used.values())}건 → {VISION}")
    print("더한 등장(화 수):", dict(used.most_common()))


if __name__ == "__main__":
    main()
