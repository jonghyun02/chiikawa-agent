"""
공식 재생목록 메타데이터 받기 — AniBox(@myanibox) 「[전편공개] 먼작귀」 재생목록
================================================================================
영상은 받지 않고 제목·설명란·길이·올린 날짜만 받는다 (yt-dlp --skip-download).
비공개로 바뀐 영상(119·240화 등)은 건너뛴다.

실행:  python scripts/fetch_videos.py      → data/raw/videos.jsonl  (다음 단계: build_data.py)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PLAYLIST = "https://www.youtube.com/playlist?list=PLsmaUhkCwW3q5q46uc9So2b4D-pZqVZOV"
OUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "videos.jsonl"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "yt_dlp", "--skip-download", "--ignore-errors", "--no-warnings",
           "--print", "%(.{id,title,description,duration,upload_date,playlist_index})j", PLAYLIST]
    with OUT.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, stdout=f, check=False)
    print(f"{OUT} : {sum(1 for _ in OUT.open(encoding='utf-8'))}개 영상")


if __name__ == "__main__":
    main()
