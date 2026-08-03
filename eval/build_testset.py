"""Sinh eval test set từ data/chunks.json bằng JudgeLLM (model rẻ, không đụng chatbot/llm.py).

uv run python -m eval.build_testset --n 20
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.chatbot.config import Settings
from src.chatbot.llm import JudgeLLM

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FILE = ROOT / "data" / "chunks.json"
OUTPUT_FILE = ROOT / "data" / "eval_testset.json"

CONTACT_SUPPORT_QUESTIONS = [
    "Tôi muốn liên hệ phòng tuyển sinh thì gọi số nào?",
    "Cho tôi email liên hệ để hỏi thêm về chương trình.",
    "Nếu có thắc mắc ngoài những gì đã hỏi thì liên hệ ai?",
    "Ai là người tôi cần liên hệ nếu muốn khiếu nại kết quả tuyển sinh?",
]

QUESTION_PROMPT = (
    "Đọc đoạn tài liệu tuyển sinh sau, đặt 1 câu hỏi tự nhiên (tiếng Việt) mà đoạn này trả lời "
    "được, và viết câu trả lời đúng dựa hoàn toàn vào đoạn văn (không bịa thêm thông tin ngoài "
    "đoạn văn). Trả về đúng 1 object JSON dạng "
    '{"question": "...", "ground_truth": "..."}, không thêm chữ nào khác.'
)


def _stratified_sample(chunks: list[dict], n: int, seed: int = 0) -> list[dict]:
    by_source: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk["metadata"]["source_file"], []).append(chunk)

    rng = random.Random(seed)
    for group in by_source.values():
        rng.shuffle(group)

    sampled: list[dict] = []
    sources = list(by_source.keys())
    index = 0
    while len(sampled) < n and any(by_source.values()):
        source = sources[index % len(sources)]
        if by_source[source]:
            sampled.append(by_source[source].pop())
        index += 1
    return sampled[:n]


def _generate_qa(judge: JudgeLLM, chunk_content: str) -> dict | None:
    response = judge.client.chat.completions.create(
        model=judge.settings.judge_model,
        messages=[
            {"role": "system", "content": QUESTION_PROMPT},
            {"role": "user", "content": chunk_content},
        ],
        temperature=0.3,
        max_tokens=384,
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or "question" not in parsed or "ground_truth" not in parsed:
        return None
    return parsed


def build_testset(n: int, seed: int = 0) -> list[dict]:
    data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    chunks = _stratified_sample(data["chunks"], n, seed=seed)

    settings = Settings.from_env()
    judge = JudgeLLM(settings)

    testset: list[dict] = []
    for chunk in chunks:
        qa = _generate_qa(judge, chunk["content"])
        if qa is None:
            continue
        testset.append(
            {
                "question": qa["question"],
                "ground_truth": qa["ground_truth"],
                "contexts": [chunk["content"]],
                "expected_tool": "attach_source_link",
                "source_chunk_id": chunk["id"],
            }
        )

    for question in CONTACT_SUPPORT_QUESTIONS:
        testset.append(
            {
                "question": question,
                "ground_truth": "",
                "contexts": [],
                "expected_tool": "contact_support",
                "source_chunk_id": None,
            }
        )

    return testset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    testset = build_testset(args.n, seed=args.seed)
    OUTPUT_FILE.write_text(json.dumps(testset, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã sinh {len(testset)} câu hỏi -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
