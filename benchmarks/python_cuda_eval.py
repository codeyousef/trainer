#!/usr/bin/env python3
"""Benchmark-only Python CUDA competitor for the sinai-agent eval activity."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path


def read_rows(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def unique_texts(rows: list[dict]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in ("query_text", "positive_chunk_text"):
            text = row[key]
            if text not in seen:
                seen.add(text)
                values.append(text)
    return values


def read_thresholds(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "thresholds" in raw:
        return {
            item.get("domain", "default"): float(item.get("threshold", 0.0))
            for item in raw["thresholds"]
        }
    if isinstance(raw, list):
        return {
            item.get("domain", "default"): float(item.get("threshold", 0.0))
            for item in raw
        }
    return {str(key): float(value) for key, value in raw.items()}


def score_rows(rows: list[dict], texts: list[str], embeddings, thresholds: dict[str, float], torch):
    text_index = {text: i for i, text in enumerate(texts)}
    answered = 0
    scores: list[float] = []
    for row in rows:
        query = embeddings[text_index[row["query_text"]]]
        positive = embeddings[text_index[row["positive_chunk_text"]]]
        score = float(torch.dot(query, positive).item())
        threshold = thresholds.get(row.get("domain") or "default", thresholds.get("default", 0.0))
        if score >= threshold:
            answered += 1
        scores.append(score)
    return answered, scores


def run_once(model, texts: list[str], rows: list[dict], thresholds: dict[str, float], batch_size: int, torch):
    torch.cuda.synchronize()
    start = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device="cuda",
    )
    torch.cuda.synchronize()
    encode_s = time.perf_counter() - start
    score_start = time.perf_counter()
    answered, scores = score_rows(rows, texts, embeddings, thresholds, torch)
    score_s = time.perf_counter() - score_start
    return {
        "encode_s": encode_s,
        "score_s": score_s,
        "answer_rate": answered / len(rows) if rows else 0.0,
        "answered": answered,
        "total": len(rows),
        "first_score": scores[0] if scores else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")

    rows = read_rows(Path(args.test_jsonl), args.max_rows)
    texts = unique_texts(rows)
    thresholds = read_thresholds(Path(args.thresholds))

    process_start = time.perf_counter()
    load_start = time.perf_counter()
    model = SentenceTransformer(args.model_dir, device="cuda")
    if not str(model.device).startswith("cuda"):
        raise SystemExit(f"model did not load on CUDA: {model.device}")
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_start

    for _ in range(max(args.warmup, 0)):
        run_once(model, texts, rows, thresholds, args.batch_size, torch)

    torch.cuda.reset_peak_memory_stats()
    runs = []
    for _ in range(max(args.runs, 1)):
        runs.append(run_once(model, texts, rows, thresholds, args.batch_size, torch))
    cold_wall_s = time.perf_counter() - process_start

    output = {
        "implementation": "python-sentence-transformer",
        "device": str(model.device),
        "gpu_required": True,
        "model_dir": args.model_dir,
        "test_jsonl_path": args.test_jsonl,
        "thresholds_path": args.thresholds,
        "rows": len(rows),
        "unique_texts": len(texts),
        "load_s": load_s,
        "cold_wall_s": cold_wall_s,
        "warmup_runs": max(args.warmup, 0),
        "measured_runs": max(args.runs, 1),
        "runs": runs,
        "rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }
    Path(args.result).parent.mkdir(parents=True, exist_ok=True)
    Path(args.result).write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
