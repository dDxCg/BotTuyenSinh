"""RAGAS lite — ba metric chấm bằng LLM judge, tự cài, không dùng package `ragas`.

`ragas==0.4.3` resolve được trên Python 3.13 nhưng kéo theo 92 package
(`datasets`, `langchain-core`, `langgraph`, `pandas`, `numpy`). Ba metric dưới đây
là toàn bộ thứ cần dùng, nên tự cài rẻ hơn nhiều.

Judge chỉ nhận `JudgeSample` — bản ghi chuỗi thuần. Không import agent, không import
retriever. Nhờ vậy viết và test được trong lúc bot còn đang sửa.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = EVAL_DIR / "results" / "judge_cache.json"

# `judge.py` không import `src.chatbot.config` (để không dính vào code đang được sửa),
# nên phải tự nạp .env — nếu không, chạy `run_ragas.py` sẽ không thấy OPENAI_API.
load_dotenv(EVAL_DIR.parent / ".env")

# Nằm trong khoá cache: sửa prompt judge là mọi điểm cũ tự vô hiệu. Nếu không có,
# sửa prompt xong vẫn đọc điểm cũ mà không hay biết.
PROMPT_VERSION = "1"

DEFAULT_JUDGE_MODEL = "google/gemini-2.5-flash"

Complete = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class JudgeSample:
    """Đơn vị chấm. Chuỗi thuần — không phụ thuộc kiểu dữ liệu của agent."""

    id: str
    question: str
    answer: str
    contexts: list[str]
    reference: str = ""


@dataclass
class MetricResult:
    score: float | None
    detail: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    cached: bool = False


@dataclass(frozen=True)
class JudgeSettings:
    api_key: str
    model: str
    base_url: str
    timeout_seconds: float = 90.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "JudgeSettings":
        # Key riêng cho judge: nhà cung cấp / hạn mức / model của judge không nhất thiết
        # trùng sản phẩm. Chỉ mượn OPENAI_API khi thật sự không có JUDGE_API.
        api_key = os.getenv("JUDGE_API", "") or os.getenv("OPENAI_API", "")
        if not api_key:
            raise RuntimeError("Thiếu JUDGE_API (hoặc OPENAI_API) trong .env")
        model = os.getenv("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
        product_model = os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini")
        if model == product_model:
            raise RuntimeError(
                f"JUDGE_MODEL trùng OPENAI_MODEL ({model}). Model tự chấm bài mình viết "
                "là self-preference bias — điểm sẽ đẹp một cách vô nghĩa. Đặt JUDGE_MODEL khác."
            )
        return cls(
            api_key=api_key,
            model=model,
            base_url=os.getenv("JUDGE_BASE_URL", "")
            or os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
            timeout_seconds=float(os.getenv("JUDGE_TIMEOUT_SECONDS", "90")),
            max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "2")),
        )


def build_complete(settings: JudgeSettings) -> Complete:
    """Client thật. Import `openai` muộn để test offline không cần cài SDK."""
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )

    def complete(messages: list[dict[str, str]]) -> str:
        response = client.chat.completions.create(
            model=settings.model,
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    return complete


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_json(raw: str) -> dict[str, Any]:
    """Judge hay bọc JSON trong ```json fence hoặc kèm lời dẫn — bóc lấy object đầu tiên."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.M).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_BLOCK.search(text)
        if not match:
            raise ValueError(f"không tìm thấy JSON trong output: {raw[:200]!r}") from None
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(f"JSON không phải object: {type(value).__name__}")
    return value


def _clamp(value: Any) -> float:
    """Judge thỉnh thoảng trả 0-100 hoặc chuỗi — kéo về [0,1] thay vì vứt cả lượt chấm."""
    score = float(value)
    if score > 1.0:
        score = score / 100.0 if score <= 100.0 else 1.0
    return max(0.0, min(1.0, score))


CLAIM_PROMPT = """Bạn tách câu trả lời thành các mệnh đề (claim) đơn, độc lập, kiểm chứng được.

Quy tắc:
- Mỗi claim là MỘT dữ kiện, viết lại thành câu đầy đủ (không dùng "nó", "cái này").
- Bỏ qua lời chào, lời mời liên hệ, câu hỏi ngược — chúng không phải dữ kiện.
- Không suy diễn thêm dữ kiện không có trong câu trả lời.

Chỉ trả JSON: {"claims": ["...", "..."]}"""

VERIFY_PROMPT = """Bạn kiểm tra từng mệnh đề có được SUY RA TRỰC TIẾP từ ngữ cảnh không.

Quy tắc:
- supported=true chỉ khi ngữ cảnh nói rõ điều đó. Kiến thức nền của bạn KHÔNG tính.
- Mệnh đề đúng trong thực tế nhưng ngữ cảnh không nhắc ⇒ supported=false.
- Giữ nguyên thứ tự và số lượng mệnh đề đầu vào.

Chỉ trả JSON: {"verdicts": [{"supported": true, "reason": "..."}]}"""

RELEVANCY_PROMPT = """Bạn chấm câu trả lời có đúng TRỌNG TÂM câu hỏi không (không chấm đúng/sai dữ kiện).

Thang điểm:
- 1.0 trả lời thẳng đúng thứ được hỏi
- 0.5 trả lời một phần, hoặc lan man sang chuyện khác
- 0.0 không liên quan, hoặc né không trả lời thứ được hỏi

Chỉ trả JSON: {"score": 0.0, "reason": "..."}"""

CORRECTNESS_PROMPT = """Bạn so câu trả lời với đáp án chuẩn.

Thang điểm:
- 1.0 mọi dữ kiện trong đáp án chuẩn đều có và không có dữ kiện sai
- trừ điểm cho dữ kiện SAI (nặng) và dữ kiện THIẾU (nhẹ)
- 0.0 sai hoàn toàn hoặc không có dữ kiện nào trùng
Khác cách diễn đạt nhưng cùng nội dung thì KHÔNG trừ điểm.

Chỉ trả JSON: {"score": 0.0, "wrong": ["..."], "missing": ["..."], "reason": "..."}"""


def _join_contexts(contexts: list[str]) -> str:
    return "\n\n---\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))


class Judge:
    """Ba metric + cache. `complete` tiêm được để test offline."""

    def __init__(
        self,
        complete: Complete | None = None,
        settings: JudgeSettings | None = None,
        cache_path: Path | None = DEFAULT_CACHE,
        prompt_version: str = PROMPT_VERSION,
    ) -> None:
        self.settings = settings
        self._complete = complete
        self.prompt_version = prompt_version
        self.cache_path = cache_path
        self.cache: dict[str, Any] = {}
        if cache_path and cache_path.exists():
            self.cache = json.loads(cache_path.read_text(encoding="utf-8"))
        self.llm_calls = 0
        self.cache_hits = 0

    # --- hạ tầng ---------------------------------------------------------

    @property
    def model(self) -> str:
        return self.settings.model if self.settings else "injected"

    def complete(self, messages: list[dict[str, str]]) -> str:
        if self._complete is None:
            if self.settings is None:
                self.settings = JudgeSettings.from_env()
            self._complete = build_complete(self.settings)
        self.llm_calls += 1
        return self._complete(messages)

    def _key(self, metric: str, sample: JudgeSample) -> str:
        payload = "|".join(
            [metric, self.model, self.prompt_version, sample.question, sample.answer,
             "\u0000".join(sample.contexts), sample.reference]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _ask_json(self, system: str, user: str) -> dict[str, Any]:
        """Một lượt hỏi judge. Parse hỏng thì thử lại một lần với chỉ dẫn chặt hơn."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        raw = self.complete(messages)
        try:
            return parse_json(raw)
        except (ValueError, json.JSONDecodeError):
            retry = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Output vừa rồi không phải JSON hợp lệ. "
                                            "Trả LẠI đúng một JSON object, không kèm chữ nào khác."},
            ]
            return parse_json(self.complete(retry))

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _run(self, metric: str, sample: JudgeSample, fn: Callable[[], MetricResult]) -> MetricResult:
        key = self._key(metric, sample)
        if key in self.cache:
            self.cache_hits += 1
            hit = self.cache[key]
            return MetricResult(hit["score"], hit.get("detail", {}), hit.get("error", ""), cached=True)
        try:
            result = fn()
        except Exception as exc:  # một sample hỏng không được giết cả run
            result = MetricResult(None, {}, f"{type(exc).__name__}: {exc}")
        # CHỈ cache kết quả thành công. Cache lỗi hạ tầng (hết credit, timeout, 5xx)
        # sẽ đóng băng nó vĩnh viễn: nạp thêm credit rồi chạy lại vẫn ăn cache None.
        if result.error:
            return result
        self.cache[key] = {"score": result.score, "detail": result.detail, "error": result.error}
        return result

    # --- metric ----------------------------------------------------------

    def faithfulness(self, sample: JudgeSample) -> MetricResult:
        """Tỉ lệ claim trong answer được ngữ cảnh hỗ trợ.

        Tách claim rồi mới verify, thay vì hỏi một phát "có trung thực không":
        điểm thô từ câu hỏi mở là ý kiến, còn tỉ lệ claim được hỗ trợ là số đếm
        được và chỉ ra được đúng claim nào không có căn cứ.
        """

        def run() -> MetricResult:
            if not sample.answer.strip():
                return MetricResult(None, {}, "answer rỗng")
            if not sample.contexts:
                return MetricResult(None, {}, "không có context")

            claims_payload = self._ask_json(
                CLAIM_PROMPT, f"Câu hỏi:\n{sample.question}\n\nCâu trả lời:\n{sample.answer}"
            )
            claims = [str(c).strip() for c in claims_payload.get("claims", []) if str(c).strip()]
            if not claims:
                # Answer chỉ có lời mời liên hệ (case chuyển người) — không đo được,
                # khác hẳn với "đo được và bằng 0".
                return MetricResult(None, {"claims": []}, "không tách được claim nào")

            listed = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
            verdict_payload = self._ask_json(
                VERIFY_PROMPT,
                f"Ngữ cảnh:\n{_join_contexts(sample.contexts)}\n\nCác mệnh đề:\n{listed}",
            )
            verdicts = verdict_payload.get("verdicts", [])
            if len(verdicts) != len(claims):
                return MetricResult(
                    None,
                    {"claims": claims, "verdicts": verdicts},
                    f"judge trả {len(verdicts)} verdict cho {len(claims)} claim",
                )

            detail = [
                {
                    "claim": claim,
                    "supported": bool(v.get("supported")),
                    "reason": str(v.get("reason", "")),
                }
                for claim, v in zip(claims, verdicts)
            ]
            supported = sum(1 for d in detail if d["supported"])
            return MetricResult(
                supported / len(detail),
                {"claims": detail, "supported": supported, "total": len(detail)},
            )

        return self._run("faithfulness", sample, run)

    def answer_relevancy(self, sample: JudgeSample) -> MetricResult:
        def run() -> MetricResult:
            if not sample.answer.strip():
                return MetricResult(None, {}, "answer rỗng")
            payload = self._ask_json(
                RELEVANCY_PROMPT, f"Câu hỏi:\n{sample.question}\n\nCâu trả lời:\n{sample.answer}"
            )
            return MetricResult(_clamp(payload["score"]), {"reason": payload.get("reason", "")})

        return self._run("answer_relevancy", sample, run)

    def answer_correctness(self, sample: JudgeSample) -> MetricResult:
        def run() -> MetricResult:
            if not sample.answer.strip():
                return MetricResult(None, {}, "answer rỗng")
            if not sample.reference.strip():
                return MetricResult(None, {}, "thiếu reference")
            payload = self._ask_json(
                CORRECTNESS_PROMPT,
                f"Câu hỏi:\n{sample.question}\n\nĐáp án chuẩn:\n{sample.reference}"
                f"\n\nCâu trả lời cần chấm:\n{sample.answer}",
            )
            return MetricResult(
                _clamp(payload["score"]),
                {
                    "wrong": payload.get("wrong", []),
                    "missing": payload.get("missing", []),
                    "reason": payload.get("reason", ""),
                },
            )

        return self._run("answer_correctness", sample, run)

    def score_all(self, sample: JudgeSample) -> dict[str, MetricResult]:
        return {
            "faithfulness": self.faithfulness(sample),
            "answer_relevancy": self.answer_relevancy(sample),
            "answer_correctness": self.answer_correctness(sample),
        }


METRICS = ("faithfulness", "answer_relevancy", "answer_correctness")


def mean_ignoring_none(values: list[float | None]) -> float | None:
    """Trung bình bỏ qua None. Người gọi phải tự in số sample bị bỏ —
    trung bình trên 20/23 sample mà không nói ra là con số dối."""
    scored = [v for v in values if v is not None]
    return sum(scored) / len(scored) if scored else None
