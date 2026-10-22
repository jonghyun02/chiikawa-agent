"""
장면 검색 평가 — "이 장면 몇 화야?" 질문에 검색이 정답 화를 가져오는지 잰다 (API 키 안 씀)
=====================================================================
질문 세트: data/eval_scene_queries.json  {q: 팬 말투 질문, eps: 정답 화 목록, alias?: 원어 이름, hard?: 낱말 바꿈}
  - 질문은 화별 영상 줄거리를 읽고 팬 말투로 다시 쓴 것. 줄거리 낱말이 겹쳐 키워드 검색에 유리할 수 있어서,
    일부러 다른 낱말로 쓴 어려운 질문(hard)을 따로 센다 (예: 고추냉이 → 와사비, 개복치 → 커다란 물고기)

지표 (search_docs 와 같은 top_k=6, 화 단위)
  hit@1 : 1등 청크가 정답 화인 비율
  hit@6 : 6개 안에 정답 화가 하나라도 있는 비율
  MRR   : 정답 화가 처음 나온 순위의 역수 평균

실행:  python scripts/eval_search.py           (틀린 질문도 보려면 -v)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_module  # noqa: E402

QUERIES = ROOT / "data" / "eval_scene_queries.json"
TOP_K = 6


def first_rank(hits: list[dict], eps: list[int]) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.get("ep") in eps:
            return rank
    return None


def evaluate(queries: list[dict], top_k: int = TOP_K) -> dict:
    rows = []
    for item in queries:
        hits = rag_module.search(item["q"], top_k=top_k)
        rows.append({**item, "rank": first_rank(hits, item["eps"]), "got": [h.get("source") for h in hits]})
    n = len(rows)
    return {
        "n": n,
        "hit@1": sum(r["rank"] == 1 for r in rows) / n,
        "hit@6": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "rows": rows,
    }


def main() -> None:
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    res = evaluate(queries)
    print(f"모델 {rag_module.EMBED_MODEL_NAME} · 질문 {res['n']}개")
    print(f"hit@1 {res['hit@1']:.0%} · hit@{TOP_K} {res['hit@6']:.0%} · MRR {res['mrr']:.3f}")
    hard = [r for r in res["rows"] if r.get("hard")]
    print(f"  그중 낱말을 바꾼 어려운 질문 {len(hard)}개: hit@1 {sum(r['rank'] == 1 for r in hard) / len(hard):.0%}"
          f" · hit@{TOP_K} {sum(r['rank'] is not None for r in hard) / len(hard):.0%}")
    if "-v" in sys.argv:
        for r in res["rows"]:
            if r["rank"] != 1:
                print(f"  [{r['rank'] or '-'}] {r['q']}  정답 {r['eps']}  → {r['got'][:3]}")


if __name__ == "__main__":
    main()
