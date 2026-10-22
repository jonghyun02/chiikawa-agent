# data/ — 이 폴더에 있어야 하는 파일

공개 저장소에는 저작권이 있는 자료를 올리지 않아서 `data/` 에는 평가용 질문(`eval_scene_queries.json`)만 있습니다.
실행하려면 아래 파일을 만들어 이 폴더에 두세요 (팀원은 원본 폴더를 따로 받으세요).

| 파일 | 만드는 방법 |
|---|---|
| `raw/videos.jsonl` | `python scripts/fetch_videos.py` — 공식 재생목록 메타데이터 수집 (yt-dlp) |
| `characters.json` | 캐릭터 사전 (이름·별명·원어·등장 낱말 규칙 `text_patterns`) 을 직접 작성 |
| `scenes/{화}.json` | `python scripts/describe_scenes.py --prepare` → 에이전트가 작업지를 보고 기록 → `--ingest` |
| `chiikawa.db`, `rag_chunks.jsonl` | `python scripts/build_data.py` |
| `rag_vectors.npy`, `rag_vectors.json` | `python scripts/build_vectors.py` (Intel GPU/NPU 가 없으면 `--device CPU`, 느림) |

Dockerfile 은 `data/chiikawa.db`, `data/rag_chunks.jsonl`, `data/characters.json`, `data/rag_vectors.*` 를 이미지에 복사하므로
위 파일이 모두 있어야 빌드됩니다.
