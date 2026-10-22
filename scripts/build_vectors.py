"""
RAG 문서 벡터 미리 만들기 — data/rag_chunks.jsonl → data/rag_vectors.npy · rag_vectors.json
=====================================================================
청크 2,776개를 CPU 로 임베딩하면 embeddinggemma 기준 17분 걸린다.
이 PC 의 가속기(Intel Arc GPU · AI Boost NPU)에서 OpenVINO 로 미리 계산해 두면 1분이 안 걸리고,
rag_module.ensure_index 는 이 파일이 지금 청크·모델과 맞을 때 임베딩 없이 바로 Qdrant 에 넣는다 (Docker 안에서도).

장치 고르기 (2026-10-22 측정, embeddinggemma-300m, [8, 288])
  GPU  초당 63개, fastembed(CPU) 결과와 코사인 1.000                          ← 기본값
  NPU  초당 64개, FP16 이라 코사인 평균 0.98 · 최저 0.65 (검색 점수는 같았음)
       질문 벡터는 실행 중 fastembed(CPU)로 만들기 때문에 같은 값이 나오는 GPU 를 쓴다
  CPU  초당 2.5개
NPU 는 정적 shape 만 되므로 어느 장치든 [BATCH, MAX_TOKENS] 로 고정해 컴파일한다.

실행:  python scripts/build_vectors.py [--device GPU|NPU|CPU]   (rag_chunks.jsonl 이 바뀌면 다시)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import rag_module as R  # noqa: E402

BATCH, MAX_TOKENS = 8, 288          # 청크 최대 285토큰 (embeddinggemma 토크나이저, 문서 앞머리 포함)


def load_tokenizer(model_dir: Path) -> Tokenizer:
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    pad = json.loads((model_dir / "special_tokens_map.json").read_text(encoding="utf-8"))["pad_token"]
    pad = pad["content"] if isinstance(pad, dict) else pad
    tok.enable_truncation(max_length=MAX_TOKENS)
    tok.enable_padding(length=MAX_TOKENS, pad_id=tok.token_to_id(pad), pad_token=pad)
    return tok


def compile_model(model_dir: Path, device: str):
    core = ov.Core()
    core.set_property({"CACHE_DIR": str(ROOT / ".ov_cache")})       # NPU 컴파일(수십 초~)을 다음부터 건너뛴다
    model = core.read_model(next(model_dir.rglob("model.onnx")))
    model.reshape({i.get_any_name(): [BATCH, MAX_TOKENS] for i in model.inputs})
    net = core.compile_model(model, device)
    print(f"장치 {device} → 실제 실행 {net.get_property('EXECUTION_DEVICES')}")
    return net


def embed(net, tok: Tokenizer, texts: list[str]) -> np.ndarray:
    out = []
    for s in range(0, len(texts), BATCH):
        batch = texts[s:s + BATCH]
        enc = tok.encode_batch(batch + [""] * (BATCH - len(batch)))   # 마지막 묶음도 같은 shape 로 채운다
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        res = net({"input_ids": ids, "attention_mask": mask})
        out.append(res[net.output("sentence_embedding")][:len(batch)])  # 모델 안에서 풀링·정규화까지 끝난 출력
    v = np.concatenate(out).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="GPU", choices=["GPU", "NPU", "CPU"])
    device = ap.parse_args().device

    chunks = R.load_chunks()
    doc_prefix = R.EMBED_PREFIX.get(R.EMBED_MODEL_NAME, ("", ""))[1]
    texts = [doc_prefix + R.embed_text(c) for c in chunks]
    fe = R.get_embedding_model()                       # fastembed 가 모델을 내려받은 폴더를 그대로 쓴다
    model_dir = Path(fe.model._model_dir)

    tok = load_tokenizer(model_dir)
    net = compile_model(model_dir, device)
    t = time.time()
    vectors = embed(net, tok, texts)
    dt = time.time() - t
    print(f"청크 {len(texts)}개 · {dt:.1f}초 (초당 {len(texts) / dt:.0f}개)")

    # 실행 중 질문은 fastembed(CPU)로 임베딩하므로 같은 공간인지 앞 8개로 확인한다
    ref = np.array(list(fe.embed(texts[:BATCH])))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    cos = (ref * vectors[:BATCH]).sum(1)
    print(f"fastembed(CPU) 와 코사인 최저 {cos.min():.4f} · 평균 {cos.mean():.4f}")

    np.save(R.VECTORS_PATH, vectors)
    R.VECTORS_PATH.with_suffix(".json").write_text(json.dumps({
        "model": R.EMBED_MODEL_NAME, "collection": R.collection_name(chunks), "device": device,
        "count": len(chunks), "dim": int(vectors.shape[1]), "seconds": round(dt, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {R.VECTORS_PATH.relative_to(ROOT)} ({vectors.shape[0]}×{vectors.shape[1]})")


if __name__ == "__main__":
    main()
