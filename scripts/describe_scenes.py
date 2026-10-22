"""
장면 설명 만들기 — 영상 캡처 모음 + 대사 받아쓰기 → (Claude Code 에이전트가 보고 기록) → 화별 장면 JSON
=====================================================================
API 키를 쓰지 않는다. 모델 호출은 Claude Code 구독 안의 에이전트가 한다.
  1) --prepare : 화마다 작업지(규칙 + 화 정보 + 대사 + 캡처 이미지 경로)를 data/scenes/tasks/{번호}.md 로 만든다
  2) 에이전트  : 작업지를 읽고 캡처 이미지를 직접 본 뒤 data/scenes/raw/{번호}.json 에 기록을 쓴다
  3) --ingest  : raw 기록을 검사하고 원본 시각·메타데이터·대사를 붙여 data/scenes/{화}.json 으로 저장

입력 (영상 전처리 폴더, 기본 C:/projects/yt-dlp/preprocess, 환경변수 PREPROCESS_DIR 로 바꿀 수 있음)
  cuts.json             : 화별 제거 구간(오프닝·전환·엔딩) → 원본 시각 복원
  sheets/{번호}/         : scene_sheets.py 가 만든 1초 간격 4x4 캡처 모음 (칸마다 잘린 영상 기준 시각)
  transcripts/{번호}.json: transcribe.py 가 만든 대사 받아쓰기
  data/raw/videos.jsonl : 공식 설명란 (소제목·예고 줄거리. 해시태그는 쓰지 않는다)
출력
  data/scenes/{화}.json : 전체 줄거리 · 소제목별 줄거리 · 장면 목록(시각·등장인물·장소·행동·대사·검색어)
                          시각은 잘린 영상 기준(start/end)과 원본 기준(orig_start/orig_end, 유튜브 &t=) 둘 다
({번호} 는 파일 앞 세 자리 = 재생목록 순번, {화} 는 화 번호 세 자리)

실행
  python scripts/describe_scenes.py --prepare            # 대사가 준비된 화 중 아직 기록 없는 화의 작업지
  python scripts/describe_scenes.py --prepare 013 024    # 일부 화만
  python scripts/describe_scenes.py --ingest             # raw 기록 → 최종 JSON
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRE = Path(os.environ.get("PREPROCESS_DIR", "C:/projects/yt-dlp/preprocess"))
RAW = ROOT / "data" / "raw" / "videos.jsonl"
OUT = ROOT / "data" / "scenes"
TASKS = OUT / "tasks"
RAW_OUT = OUT / "raw"

EP_RE = re.compile(r"(\d+)화_(.+)$")

RULES = """너는 애니메이션 「먼가 작고 귀여운 녀석」(먼작귀, 원제 ちいかわ 치이카와) 한국어 더빙판의 장면 기록 담당이다.
한 화(오프닝·엔딩을 뺀 본편 약 1분)의 캡처 모음과 받아쓴 대사를 보고, 팬이 "그 장면 몇 화였지?"를 검색할 수 있게 장면을 기록한다.

## 캐릭터 이름 (반드시 더빙판 이름을 쓴다)
- 치이카와: 하얗고 작고 둥근 곰 같은 아이. 둥근 귀, 볼 홍조. 잘 울고 말수가 적다.
- 가르마(원작 하치와레): 흰 고양이 같은 아이. 머리 윗부분이 파란색이고 이마 가운데에서 팔(八)자로 갈라진 무늬. 말을 잘하고 긍정적.
- 토끼(원작 우사기): 크림색(연노랑) 토끼. 긴 귀. "우라!", "야하!", "하아?" 같은 소리만 지른다. 엉뚱하고 힘이 세다.
- 하늘다람쥐(원작 모몽가): 흰 하늘다람쥐. 눈이 크고 반짝인다. 귀여운 척하며 조른다.
- 밤만쥬(원작 쿠리만쥬): 윗부분이 갈색으로 구워진 밤만주 같은 머리. 술과 안주를 좋아하고 "하아~" 하고 감탄한다.
- 해달(원작 랏코): 해달. 토벌 실력이 최고라 모두가 동경한다.
- 시사: 오키나와 시사(사자개) 모습. 라멘 가게 「로우」의 조수, 라멘 갑옷 씨를 '사부님'이라 부른다.
- 헌책방(원작 후루혼야): 헌책을 판다. 게 모양 머리띠를 쓰고 있다.
- 갑옷 씨: 갑옷을 입은 어른들. 포셰트 갑옷 씨(파자마·포셰트를 만들어 팖), 노동 갑옷 씨(노동 접수), 라멘 갑옷 씨(라멘 가게 「로우」 점주), 대장 갑옷 씨(황금 갑옷·왕관 투구·망토), 타코야키 갑옷 씨(붉은 갑옷), 유두부 갑옷 씨(초록 갑옷).
- 그 밖: 파자마 파티스, 인어, 섬 아저씨(시마지로), 별똥별·검은 별똥별, 마녀, 꿀맛맨, 오데(노란 외눈박이 거한), 고블린들, 할멈.
- 적·괴물(키메라, 아노코 등)은 이름을 지어내지 말고 생김새로 적는다(예: "큰 입이 달린 검은 괴물").

## 규칙
1. 캡처에 실제로 보이는 것만 적는다. 생김새로 누구인지 확실하지 않으면 characters 에 넣지 말고 unknown_characters 에 생김새를 적는다.
2. 예고 줄거리는 누가 나오는지와 이야기 흐름을 짐작하는 힌트일 뿐이다. 화면과 다르면 화면을 따른다.
   다만 예고에 위 목록에 없는 이름(예: 풍뎅이)이 나오면 그 화의 새 캐릭터다. 생김새가 맞으면 그 이름을 쓰고, 목록의 다른 캐릭터로 바꿔 부르지 않는다.
3. 대사는 음성 인식 결과라 틀릴 수 있다. 이름 철자는 위 목록으로 고치고, 뜻이 통하지 않는 줄은 쓰지 않는다. quote 에는 그 장면에서 실제로 들린 것으로 보이는 대사만 넣고, 없으면 빈 문자열.
4. 시각은 캡처 칸 왼쪽 위 숫자(초, 잘린 영상 기준)를 쓴다. 장면은 장소·사건이 바뀌는 단위로 나누고, 화 하나에 보통 4~10개.
5. description 은 누가·어디서·무엇을 했고 어떤 표정·감정이었는지 2~3문장. 팬이 기억하는 표현(울었다, 도망쳤다, 맛있게 먹었다, 춤췄다 등)과 등장 음식·물건 이름을 구체적으로 쓴다.
6. keywords 는 검색어 3~8개(음식·물건·장소·행동·감정·유행어). 캐릭터 이름은 넣지 않는다.
7. summary 와 segments.story 는 결말까지 쓴다(예고처럼 "~는데..."로 끝내지 않는다).
8. 모든 글은 한국어.
9. 맨 끝의 "끝"/"계속" 말풍선 마무리 컷(매 화 나오는 짧은 컷)은 장면으로 적지 않고, 그 앞 장면의 end 를 본편 끝으로 둔다."""

FORMAT = """## 출력 형식 (JSON 하나, 다른 글 없이)
{
  "summary": "전체 줄거리 3~6문장, 결말까지",
  "segments": [{"title": "소제목", "start": 0.5, "end": 30.3, "story": "소제목 줄거리 2~4문장"}],
  "scenes": [{"start": 0.5, "end": 12.9, "characters": ["치이카와", "가르마"], "place": "장소",
              "description": "2~3문장", "quote": "대사 또는 빈 문자열", "keywords": ["검색어", "..."]}],
  "characters": ["화 전체 등장인물(더빙판 이름)"],
  "unknown_characters": ["이름을 모르는 캐릭터의 생김새"]
}"""


# ── 시각 변환 (preprocess/common.py 와 같은 규칙) ─────────────────

def load_cuts() -> dict[str, dict]:
    return {c["ep"]: c for c in json.loads((PRE / "cuts.json").read_text(encoding="utf-8"))}


def keep_ranges(c: dict) -> list[tuple[float, float]]:
    """원본에서 남긴 구간."""
    keep, pos = [], 0.0
    for _, a, b in c["remove"]:
        a = max(0.0, a)
        if a > pos + 0.01:
            keep.append((pos, a))
        pos = max(pos, b)
    if pos < c["dur"] - 0.01:
        keep.append((pos, c["dur"]))
    return keep


def clean_to_orig(c: dict, t: float) -> float:
    """잘린 영상 t초 → 원본 초. 본편 길이를 넘는 값(끝 시각 반올림)은 본편 끝으로."""
    acc, keep = 0.0, keep_ranges(c)
    for a, b in keep:
        if t <= acc + (b - a) + 1e-6:
            return a + (t - acc)
        acc += b - a
    return keep[-1][1]


def eye_in_clean(c: dict) -> float | None:
    """전환 카드를 잘라낸 자리(잘린 영상 기준 초). 보통 소제목 경계다."""
    for kind, a, _ in c["remove"]:
        if kind == "EYE":
            return round(sum(min(b, a) - x for x, b in keep_ranges(c) if x < a), 1)
    return None


# ── 화 정보 ──────────────────────────────────────────────────

def load_videos() -> dict[int, dict]:
    """재생목록 순번(파일 앞 세 자리) → 영상 메타데이터."""
    return {int(v["playlist_index"]): v for v in map(json.loads, RAW.read_text(encoding="utf-8").splitlines())}


def episode_info(key: str, videos: dict[int, dict]) -> dict:
    v = videos[int(key)]
    m = EP_RE.search(v["title"])
    desc = v.get("description") or ""
    segs = [ln.split(" : ", 1) for ln in desc.splitlines() if " : " in ln and not ln.startswith("#")]
    return {"key": key, "ep": int(m.group(1)), "title": m.group(2).strip(), "video_id": v["id"],
            "teasers": [{"title": a.strip(), "teaser": b.strip()} for a, b in segs]}


HOTWORDS = "치이카와 가르마 토끼 하늘다람쥐 밤만쥬 해달 시사 헌책방".split()   # transcribe.py 의 철자 힌트


def hotword_echo(text: str) -> bool:
    """철자 힌트를 그대로 읊은 환청 줄 (예: '가르마 토끼'). 낱말 대부분이 힌트 이름이면 버린다."""
    words = re.sub(r"[,.!?]", " ", text).split()
    hits = sum(any(w[:3] == h[:3] for h in HOTWORDS) for w in words)
    return bool(words) and hits >= max(2, len(words) * 0.6)


def transcript_lines(key: str) -> list[dict]:
    p = PRE / "transcripts" / f"{key}.json"
    return [ln for ln in json.loads(p.read_text(encoding="utf-8"))["lines"]
            if (ln["no_speech"] or 0) < 0.6 and not hotword_echo(ln["text"])]   # OpenVINO 받아쓰기는 null


def done_keys() -> set[str]:
    return {json.loads(p.read_text(encoding="utf-8"))["file_key"] for p in OUT.glob("[0-9]*.json")}


# ── 1) 작업지 ────────────────────────────────────────────────

def task_text(key: str, cut: dict, info: dict) -> str:
    meta = json.loads((PRE / "sheets" / key / "meta.json").read_text(encoding="utf-8"))
    clean_len = round(sum(b - a for a, b in keep_ranges(cut)), 1)
    eye = eye_in_clean(cut)
    t = [RULES, "", f"# {info['ep']}화 「{info['title']}」 (본편 길이 {clean_len}초)"]
    if info["teasers"]:
        t.append("## 공식 소제목·예고 줄거리 (힌트)")
        t += [f"- {s['title']}: {s['teaser']}" for s in info["teasers"]]
    else:
        t.append(f"## 공식 소제목: {info['title']} (예고 줄거리 없음)")
    if eye is not None:
        t.append(f"## 전환 카드를 잘라낸 자리: {eye}초 (보통 소제목이 바뀌는 곳, 아닐 수도 있음)")
    t.append("## 받아쓴 대사 (음성 인식, 틀릴 수 있음)")
    t += [f"[{ln['clean'][0]:.1f}~{ln['clean'][1]:.1f}s] {ln['text']}" for ln in transcript_lines(key)] \
        or ["(인식된 대사 없음)"]
    t.append(f"## 캡처 모음 {len(meta['sheets'])}장 (1초 간격, 칸 왼쪽 위 숫자가 시각) — 전부 Read 로 직접 볼 것")
    t += [f"- {(PRE / 'sheets' / key / s['file']).as_posix()}  ({s['times'][0]:.1f}~{s['times'][-1]:.1f}초)"
          for s in meta["sheets"]]
    t += ["", FORMAT, "", f"## 저장 위치\n{(RAW_OUT / f'{key}.json').as_posix()}"]
    return "\n".join(t) + "\n"


def run_prepare(only: list[str]) -> None:
    cuts, videos, done = load_cuts(), load_videos(), done_keys()
    TASKS.mkdir(parents=True, exist_ok=True)
    made = []
    for k in sorted(cuts):
        if (only and k not in only) or (not only and k in done):
            continue
        if not (PRE / "transcripts" / f"{k}.json").exists():
            continue                                     # 대사 받아쓰기가 아직 안 끝난 화
        (TASKS / f"{k}.md").write_text(task_text(k, cuts[k], episode_info(k, videos)), encoding="utf-8")
        made.append(k)
    print(f"작업지 {len(made)}개 → {TASKS}")
    print(" ".join(made))


# ── 3) 검사 · 저장 ───────────────────────────────────────────

def check(data: dict, clean_len: float) -> list[str]:
    """raw 기록의 형식 검사. 문제 목록을 돌려준다 (비면 통과)."""
    errs = [f"키 없음: {k}" for k in ("summary", "segments", "scenes", "characters", "unknown_characters")
            if k not in data]
    if errs:
        return errs
    for i, s in enumerate(data["scenes"]):
        miss = [k for k in ("start", "end", "characters", "place", "description", "quote", "keywords") if k not in s]
        if miss:
            errs.append(f"장면 {i} 키 없음: {miss}")
        elif not (0 <= s["start"] <= s["end"] <= clean_len + 2):
            errs.append(f"장면 {i} 시각 이상: {s['start']}~{s['end']} (본편 {clean_len}초)")
    for i, s in enumerate(data["segments"]):
        if not all(k in s for k in ("title", "start", "end", "story")):
            errs.append(f"소제목 {i} 키 없음")
    if not data["scenes"]:
        errs.append("장면이 비어 있음")
    return errs


SCENE_KEYS = ("start", "end", "characters", "place", "description", "quote", "keywords")
END_CARD = re.compile(r"['\"‘“]?(끝|계속)['\"’”]?\s*(말풍선|이라는|이라고)")


def normalize(data: dict) -> dict:
    """에이전트가 자주 내는 작은 형식 차이를 고친다 (검토 단계 없이 쓰므로 여기서 정리).
    - 장면에 섞여 들어간 엉뚱한 키는 버린다
    - 맨 끝 3초 이하 "끝"/"계속" 말풍선 컷이 장면으로 들어가 있으면 빼고 앞 장면을 늘린다 (규칙 9)"""
    data["scenes"] = [{k: s[k] for k in SCENE_KEYS if k in s} for s in data.get("scenes", [])]
    data["segments"] = [{k: s[k] for k in ("title", "start", "end", "story") if k in s} for s in data.get("segments", [])]
    sc = data["scenes"]
    if len(sc) > 1 and sc[-1]["end"] - sc[-1]["start"] <= 3 and END_CARD.search(sc[-1].get("description", "")):
        sc[-2]["end"] = sc[-1]["end"]
        sc.pop()
    return data


def run_ingest() -> None:
    cuts, videos = load_cuts(), load_videos()
    ok, bad = [], []
    for p in sorted(RAW_OUT.glob("*.json")):
        k = p.stem
        cut, info = cuts[k], episode_info(k, videos)
        try:
            data = normalize(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            bad.append(f"{k}: JSON 오류 {e}")
            continue
        errs = check(data, sum(b - a for a, b in keep_ranges(cut)))
        if errs:
            bad.append(f"{k}: " + "; ".join(errs))
            continue
        for x in data["segments"] + data["scenes"]:
            x["orig_start"] = round(clean_to_orig(cut, x["start"]), 1)
            x["orig_end"] = round(clean_to_orig(cut, x["end"]), 1)
        rec = {"ep": info["ep"], "title": info["title"], "video_id": info["video_id"], "file_key": k,
               "url": f"https://www.youtube.com/watch?v={info['video_id']}",
               "official_teasers": info["teasers"],
               "dialogue": transcript_lines(k), **data, "made_by": "claude-code-agent"}
        (OUT / f"{info['ep']:03d}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        ok.append(k)
    print(f"저장 {len(ok)}화, 문제 {len(bad)}화")
    for b in bad:
        print("  ", b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*", help="파일 앞 세 자리(재생목록 순번). 비우면 아직 기록 없는 화 전부")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--ingest", action="store_true")
    a = ap.parse_args()
    run_prepare(a.keys) if a.prepare else run_ingest()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
