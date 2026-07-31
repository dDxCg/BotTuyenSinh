"""Chạy bộ eval 20 câu qua stack thật và chấm theo checks trong questions.json.

    uv run python eval/run_eval.py                  # chạy hết
    uv run python eval/run_eval.py --ids S01 M04    # chạy vài case
    uv run python eval/run_eval.py --type multi     # lọc theo loại

Kết quả ghi vào eval/results.json. Cột quan trọng nhất là `status`:
- as_expected  : khớp trạng thái đã biết (pass đúng như kỳ vọng, hoặc fail đúng lỗi đã ghi)
- regression   : case đang xanh nay đỏ  -> hỏng thứ vốn chạy được
- improvement  : case đang đỏ nay xanh  -> lỗi đã được sửa, cập nhật lại `expect`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.chatbot.admission_agent import build_admission_agent  # noqa: E402


_QUOTED = re.compile(r'"([^"]{8,})"')


def strip_question_echo(haystack: str, question: str) -> str:
    """Gỡ phần câu hỏi bị chép lại khỏi vùng quét `answer_none`.

    `contact_support` trả `suggested_question` = câu hỏi người dùng, và model thường
    đặt nó trong ngoặc kép ("Bạn có thể soạn câu hỏi: …"). Với câu hỏi ghép, model chỉ
    chép PHẦN ngoài phạm vi — nên phải gỡ cả đoạn trích dẫn nào là con của câu hỏi,
    không chỉ nguyên văn cả câu.

    Chỉ gỡ đoạn nằm trong ngoặc kép và thật sự là con của câu hỏi ⇒ lời khẳng định
    do agent tự viết vẫn nằm nguyên trong vùng quét.
    """
    q = question.casefold().strip()
    out = haystack.replace(q, " ")
    for quoted in _QUOTED.findall(out):
        if quoted.strip().rstrip("?.! ") in q:
            out = out.replace(quoted, " ")
    return out


def check_case(
    checks: dict,
    answer: str,
    tools: list[str],
    steps: int,
    stopped_early: bool,
    question: str = "",
) -> list[str]:
    """Trả về danh sách lý do trượt; rỗng nghĩa là đạt.

    `question` bị gỡ khỏi vùng quét `answer_none`: `contact_support` trả
    `suggested_question` = nguyên văn câu hỏi người dùng, nên mọi chuỗi cấm lấy từ
    câu hỏi đều dính oan. Đo được ở lần chạy 45 case: S06/M05/A01/A03 bị chấm trượt
    trong khi agent làm hoàn toàn đúng. Bỏ câu hỏi đi thì lời khẳng định thật của
    agent vẫn nằm nguyên trong vùng quét.
    """
    failures: list[str] = []
    haystack = answer.casefold()
    if question:
        haystack = strip_question_echo(haystack, question)

    for tool in checks.get("tools_all", []):
        if tool not in tools:
            failures.append(f"thiếu tool `{tool}` (đã gọi: {tools or 'không tool nào'})")

    any_tools = checks.get("tools_any", [])
    if any_tools and not any(tool in tools for tool in any_tools):
        failures.append(f"không gọi tool nào trong {any_tools}")

    for tool in checks.get("tools_forbidden", []):
        if tool in tools:
            failures.append(f"gọi tool bị cấm `{tool}`")

    for needle in checks.get("answer_all", []):
        if needle.casefold() not in haystack:
            failures.append(f"đáp án thiếu '{needle}'")

    any_answers = checks.get("answer_any", [])
    if any_answers and not any(n.casefold() in haystack for n in any_answers):
        failures.append(f"đáp án không chứa chuỗi nào trong {any_answers}")

    for needle in checks.get("answer_none", []):
        if needle.casefold() in haystack:
            failures.append(f"đáp án chứa chuỗi cấm '{needle}'")

    min_steps = checks.get("min_steps")
    if min_steps is not None and steps < min_steps:
        failures.append(f"chỉ {steps} bước, cần ≥ {min_steps}")

    if checks.get("not_stopped_early") and stopped_early:
        failures.append("chạm trần max_steps")

    return failures


def classify(expect: str, passed: bool) -> str:
    """Đối chiếu kết quả với trạng thái đã biết.

    `expect="unknown"` là case mới chưa từng chạy — không có gì để so, nên gắn
    `baseline` thay vì bịa ra regression/improvement. Sau full run đầu tiên phải
    ghi kết quả thật vào `expect` để lần sau bắt được regression.
    """
    if expect not in ("pass", "fail"):
        return "baseline"
    if passed == (expect == "pass"):
        return "as_expected"
    return "improvement" if passed else "regression"


def run_case(case: dict, max_steps: int, agent=None) -> dict:
    # Agent dựng sẵn và dùng lại: mỗi lần dựng là một lần nạp model E5 (~49 s đo được),
    # dựng lại từng case thì 45 case tốn ~37 phút chỉ để nạp lặp.
    # Lịch sử hội thoại phải reset giữa các case, nếu không case sau thấy case trước.
    agent = agent or build_admission_agent(max_steps=max_steps)
    agent.bot.history.clear()
    started = time.perf_counter()
    try:
        result = agent.run(case["question"])
    except Exception as exc:  # lỗi hạ tầng không được làm chết cả bộ eval
        return {
            **{k: case[k] for k in ("id", "type", "category", "question", "expect")},
            "error": f"{type(exc).__name__}: {exc}",
            "passed": False,
            "status": "error",
            "failures": ["exception khi chạy"],
            "seconds": round(time.perf_counter() - started, 2),
        }
    elapsed = time.perf_counter() - started

    tools = [step.action for step in result.steps if step.action]
    failures = check_case(
        case.get("checks", {}),
        result.answer,
        tools,
        len(result.steps),
        result.stopped_early,
        case["question"],
    )
    passed = not failures
    status = classify(case["expect"], passed)

    return {
        **{k: case[k] for k in ("id", "type", "category", "question", "expect")},
        "why": case.get("why", ""),
        "passed": passed,
        "status": status,
        "failures": failures,
        "tools": tools,
        "steps": len(result.steps),
        "stopped_early": result.stopped_early,
        "best_score": round(result.retrieved[0].score, 4) if result.retrieved else None,
        "retrieved_sources": [c.metadata.get("source_type") for c in result.retrieved],
        # Text chunk, không chỉ source_type: eval/judge.py cần nội dung thật để chấm
        # faithfulness. Không có nó thì chỉ chấm được trên gold context, tức đo nhầm thứ.
        "retrieved_contexts": [c.text for c in result.retrieved],
        # Observation của tool là NGUỒN DỮ KIỆN THỨ HAI của agent, ngang hàng chunk RAG.
        # Thiếu nó thì judge chấm hotline/email do contact_support trả về là "bịa" —
        # đo được ở lần chạy 2 case đầu tiên: N02 tụt còn faithfulness 0.40 vì lý do này.
        "tool_observations": [
            f"[{step.action}] {step.observation}"
            for step in result.steps
            if step.action and step.observation and step.observation != "<final>"
        ],
        "answer": result.answer,
        "seconds": round(elapsed, 2),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="run_eval")
    parser.add_argument("--ids", nargs="*", help="chỉ chạy các case id này")
    parser.add_argument("--type", choices=["single", "multi"], help="lọc theo loại câu hỏi")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results.json")
    args = parser.parse_args()

    suite = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    cases = suite["cases"]
    if args.ids:
        cases = [c for c in cases if c["id"] in set(args.ids)]
    if args.type:
        cases = [c for c in cases if c["type"] == args.type]

    print("Đang dựng agent (nạp model embedding local, ~50 s lần đầu)…", flush=True)
    agent = build_admission_agent(max_steps=args.max_steps)

    results = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['id']} … ", end="", flush=True)
        record = run_case(case, args.max_steps, agent=agent)
        results.append(record)
        mark = "PASS" if record["passed"] else "FAIL"
        print(f"{mark} ({record['status']}, {record['seconds']}s)")
        if record["failures"]:
            for reason in record["failures"]:
                print(f"        - {reason}")
        # Checkpoint sau mỗi case: một timeout giữa chừng không làm mất cả run.
        args.out.write_text(
            json.dumps({"partial": True, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    passed = sum(r["passed"] for r in results)
    by_status: dict[str, int] = {}
    for record in results:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1

    summary = {
        "version": suite["version"],
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "by_status": by_status,
        "total_seconds": round(sum(r["seconds"] for r in results), 1),
        "results": results,
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{passed}/{len(results)} đạt · {by_status} · {summary['total_seconds']}s")
    print(f"Chi tiết: {args.out}")
    regressions = [r["id"] for r in results if r["status"] == "regression"]
    if regressions:
        print(f"REGRESSION: {regressions}")


if __name__ == "__main__":
    main()
