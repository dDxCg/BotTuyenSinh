"""Benchmark E5-large local: model load, query embedding và Chroma retrieval."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

from src.rag import embedding
from src.rag.retrieval import DEFAULT_CHROMA_DIR, DEFAULT_ENV_FILE, open_existing_collection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval" / "results"
QUESTIONS = [
    "Chương trình học trong bao lâu?",
    "Lịch học hằng ngày như thế nào?",
    "Địa điểm học ở đâu?",
    "Điều kiện dự tuyển là gì?",
    "Hồ sơ đăng ký gồm những gì?",
    "Bài đánh giá năng lực kiểm tra nội dung gì?",
    "Chương trình có những track nào?",
    "Có thể vừa học vừa đi làm không?",
    "Hạn nộp hồ sơ khóa đang tuyển là khi nào?",
    "Học viên có được hỗ trợ học phí không?",
]


@dataclass
class Measurement:
    question_index: int
    question: str
    backend: str
    embedding_model: str
    embedding_seconds: float | None
    chroma_query_seconds: float | None
    total_retrieval_seconds: float | None
    top_chunk_id: str | None
    top_similarity: float | None
    error: str | None = None
    failure_wait_seconds: float | None = None


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return ordered[index]


def summarize(rows: list[Measurement]) -> dict:
    valid = [row for row in rows if row.embedding_seconds is not None]
    embedding_times = [float(row.embedding_seconds) for row in valid]
    total_times = [
        float(row.total_retrieval_seconds)
        for row in valid
        if row.total_retrieval_seconds is not None
    ]
    attempt_times = [
        float(row.embedding_seconds)
        if row.embedding_seconds is not None
        else float(row.failure_wait_seconds)
        for row in rows
        if row.embedding_seconds is not None or row.failure_wait_seconds is not None
    ]
    return {
        "questions": len(rows),
        "successful": len(valid),
        "failed": len(rows) - len(valid),
        "embedding_mean_seconds": statistics.fmean(embedding_times)
        if embedding_times
        else None,
        "embedding_median_seconds": statistics.median(embedding_times)
        if embedding_times
        else None,
        "embedding_p95_seconds": percentile(embedding_times, 0.95)
        if embedding_times
        else None,
        "embedding_min_seconds": min(embedding_times) if embedding_times else None,
        "embedding_max_seconds": max(embedding_times) if embedding_times else None,
        "total_retrieval_mean_seconds": statistics.fmean(total_times)
        if total_times
        else None,
        "total_retrieval_median_seconds": statistics.median(total_times)
        if total_times
        else None,
        "attempt_mean_seconds": statistics.fmean(attempt_times)
        if attempt_times
        else None,
    }


def query_chroma(
    collection, vector: list[float], top_k: int
) -> tuple[str | None, float | None]:
    raw = collection.query(
        query_embeddings=[vector],
        n_results=min(top_k, collection.count()),
        include=["distances"],
    )
    ids = (raw.get("ids") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]
    if not ids or not distances:
        return None, None
    return str(ids[0]), round(1.0 - float(distances[0]), 6)


def run_backend(
    backend: str,
    model_name: str,
    questions: list[str],
    embed: Callable[[str], list[float]],
    collection,
    top_k: int,
) -> list[Measurement]:
    rows: list[Measurement] = []
    for index, question in enumerate(questions, start=1):
        print(f"[{backend}] {index}/{len(questions)}: {question}", flush=True)
        started = time.perf_counter()
        try:
            vector = embed(question)
            embedded = time.perf_counter()
            top_chunk_id, top_similarity = query_chroma(collection, vector, top_k)
            finished = time.perf_counter()
            rows.append(
                Measurement(
                    question_index=index,
                    question=question,
                    backend=backend,
                    embedding_model=model_name,
                    embedding_seconds=embedded - started,
                    chroma_query_seconds=finished - embedded,
                    total_retrieval_seconds=finished - started,
                    top_chunk_id=top_chunk_id,
                    top_similarity=top_similarity,
                )
            )
        except Exception as exc:
            rows.append(
                Measurement(
                    question_index=index,
                    question=question,
                    backend=backend,
                    embedding_model=model_name,
                    embedding_seconds=None,
                    chroma_query_seconds=None,
                    total_retrieval_seconds=None,
                    top_chunk_id=None,
                    top_similarity=None,
                    error=f"{type(exc).__name__}: {exc}",
                    failure_wait_seconds=time.perf_counter() - started,
                )
            )
    return rows


def seconds(value: float | None) -> str:
    return "ERROR" if value is None else f"{value:.3f}"


def markdown_report(payload: dict) -> str:
    rows = payload["runs"].get("local", [])
    summary = payload["summary"]["local"]
    model = payload.get("embedding_model") or next(
        (row.get("embedding_model") for row in rows if row.get("embedding_model")),
        embedding.MODEL_ID,
    )
    load_seconds = payload["model_load_seconds"].get("local")
    lines = [
        "# Benchmark embedding local — multilingual-e5-large",
        "",
        f"Ngày chạy: {payload['generated_at']} · Top-k: {payload['top_k']}",
        "",
        "Phạm vi: model local, query embedding và Chroma retrieval; **không** đo chat LLM.",
        "",
        "## Tổng hợp",
        "",
        "| Model embedding | Model load (s) | Embedding mean (s) | Median (s) | P95 (s) | Total retrieval mean (s) | Thành công |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| `{model}` | {seconds(load_seconds)} | "
        f"{seconds(summary['embedding_mean_seconds'])} | "
        f"{seconds(summary['embedding_median_seconds'])} | "
        f"{seconds(summary['embedding_p95_seconds'])} | "
        f"{seconds(summary['total_retrieval_mean_seconds'])} | "
        f"{summary['successful']}/{summary['questions']} |",
        "",
        "## Chi tiết",
        "",
        "| # | Câu hỏi | Local embedding (s) | Chroma query (s) | Total (s) | Top chunk |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question_index']} | {row['question'].replace('|', '/')} | "
            f"{seconds(row.get('embedding_seconds'))} | "
            f"{seconds(row.get('chroma_query_seconds'))} | "
            f"{seconds(row.get('total_retrieval_seconds'))} | "
            f"{row.get('top_chunk_id') or 'N/A'} |"
        )
    lines += [
        "",
        "## Ghi chú đo",
        "",
        "- `model_load`: nạp weights local một lần khi process khởi động.",
        "- `embedding`: biến câu hỏi thành vector 1.024 chiều bằng E5-large local.",
        "- `total retrieval`: embedding + Chroma query; chưa gồm chat LLM.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Benchmark E5-large local")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    embedding.load_env_file(DEFAULT_ENV_FILE)
    config = embedding.load_config(DEFAULT_ENV_FILE)
    chroma_dir = embedding.resolve_project_path(
        os.getenv("CHROMA_DIR", ""), DEFAULT_CHROMA_DIR
    )
    collection = open_existing_collection(
        chroma_dir=chroma_dir,
        collection_name=os.getenv("CHROMA_COLLECTION", "").strip()
        or config.collection_name,
        expected_model=config.model,
    )

    started = time.perf_counter()
    embedding.warmup_local_model()
    model_load_seconds = time.perf_counter() - started
    rows = run_backend(
        "local", config.model, QUESTIONS, embedding.embed_query, collection, args.top_k
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "embedding_model": config.model,
        "local_model_path": str(embedding.resolve_model_path()),
        "top_k": args.top_k,
        "questions": QUESTIONS,
        "model_load_seconds": {"local": model_load_seconds},
        "summary": {"local": summarize(rows)},
        "runs": {"local": [asdict(row) for row in rows]},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "embedding-benchmark.json"
    markdown_path = args.output_dir / "embedding-benchmark.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(markdown_report(payload), encoding="utf-8")
    print(f"JSON: {json_path.resolve()}")
    print(f"Markdown: {markdown_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Lỗi benchmark: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
