"""Tách câu hỏi đa chủ đề trước khi retrieve — embedding-based, không phải regex thuần.

Regex chỉ tạo candidate (dấu phẩy/chấm phẩy/liên từ). Candidate ngắn/cụt (vd "email") được
LLM viết lại thành câu hỏi độc lập đầy đủ ngữ cảnh trước khi embed — candidate ngắn thiếu
ngữ cảnh làm cosine distance so sánh sai lệch (đo thực tế: distance giữa 2 clause CÙNG chủ đề
nhưng 1 clause quá ngắn còn cao hơn distance giữa clause KHÁC chủ đề). Sau khi rewrite, so
cosine distance giữa candidate liền kề để xác nhận ranh giới chủ đề thật."""

import re

from src.rag import embedding
from src.rag.semantic_chunking import cosine_similarity

from .llm import AgentLLM, rewrite_query_candidates

_CLAUSE_RE = re.compile(r"\s*(?:,|;|\bvà\b|\bhoặc\b|\bhay\b)\s*", re.IGNORECASE)

# Ngưỡng cosine distance tuyệt đối, không dùng percentile — percentile chỉ ổn định khi
# có nhiều điểm dữ liệu (chunking cả tài liệu); câu hỏi ngắn chỉ 2-4 candidate thì percentile
# gần như luôn ép đúng 1 điểm cắt bất kể nội dung, sai lệch số chủ đề thật.
DEFAULT_DISTANCE_THRESHOLD = 0.055


def split_subquestions(
    question: str,
    llm: AgentLLM,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[str]:
    candidates = [p.strip() for p in _CLAUSE_RE.split(question) if p.strip()]
    if len(candidates) < 2:
        return [question]

    rewritten = rewrite_query_candidates(llm, question, candidates)

    vectors = embedding.embed_documents(rewritten)
    distances = [
        1 - cosine_similarity(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)
    ]

    groups: list[str] = []
    current = [rewritten[0]]
    for i, distance in enumerate(distances):
        if distance > threshold:
            groups.append(" ".join(current))
            current = [rewritten[i + 1]]
        else:
            current.append(rewritten[i + 1])
    groups.append(" ".join(current))

    return groups if len(groups) > 1 else [question]


__all__ = ["split_subquestions"]
