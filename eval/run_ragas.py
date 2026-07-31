"""Chấm RAGAS lite trên `eval/ragas_set.json`.

    uv run python eval/run_ragas.py --dry-run                      # smoke, 0 token
    uv run python eval/run_ragas.py --from-results eval/results.json
    uv run python eval/run_ragas.py --ids S02 N02 --contexts gold

`--contexts` chọn ngữ cảnh đem chấm, và HAI CHẾ ĐỘ ĐO HAI THỨ KHÁC NHAU:
- `retrieved` (mặc định): answer có trung thực với cái model THỰC SỰ nhìn thấy không.
  Đây mới là nghĩa gốc của RAGAS.
- `gold`: answer có trung thực với cái model ĐÁNG LẼ phải thấy không.
Faithfulness cao ở `gold` mà thấp ở `retrieved` ⇒ lỗi nằm ở retrieval, không phải ở model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

from judge import (  # noqa: E402
    DEFAULT_CACHE,
    METRICS,
    Judge,
    JudgeSample,
    JudgeSettings,
    mean_ignoring_none,
)

RAGAS_SET = EVAL_DIR / "ragas_set.json"
RESULTS_DIR = EVAL_DIR / "results"

SHORT = {"faithfulness": "faith", "answer_relevancy": "relev", "answer_correctness": "corr"}


def stub_complete(messages: list[dict[str, str]]) -> str:
    """Judge giả cho `--dry-run`: đủ hình dạng để chạy hết đường ống, 0 token.

    Không phản ánh chất lượng gì cả — chỉ để bắt lỗi cấu trúc (thiếu field, sai
    đường dẫn, ghi báo cáo hỏng) trước khi đốt token thật.
    """
    system = messages[0]["content"]
    if system.startswith("Bạn tách câu trả lời"):
        return json.dumps({"claims": ["claim giả 1", "claim giả 2"]}, ensure_ascii=False)
    if system.startswith("Bạn kiểm tra"):
        return json.dumps(
            {"verdicts": [{"supported": True, "reason": "dry-run"},
                          {"supported": False, "reason": "dry-run"}]},
            ensure_ascii=False,
        )
    return json.dumps({"score": 0.5, "reason": "dry-run", "wrong": [], "missing": []},
                      ensure_ascii=False)


def load_answers(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {r["id"]: r for r in payload.get("results", [])}


def build_samples(
    ragas_set: dict,
    answers: dict[str, dict],
    contexts_mode: str,
    ids: list[str] | None,
) -> tuple[list[JudgeSample], list[str]]:
    samples: list[JudgeSample] = []
    skipped: list[str] = []
    for item in ragas_set["samples"]:
        if ids and item["id"] not in ids:
            continue
        record = answers.get(item["id"])
        if record is None:
            skipped.append(f"{item['id']}: không có answer trong results")
            continue
        answer = record.get("answer", "")
        if contexts_mode == "gold":
            contexts = item["gold_contexts"]
        else:
            contexts = record.get("retrieved_contexts") or []
            if not contexts:
                skipped.append(f"{item['id']}: results thiếu `retrieved_contexts`")
                continue
        # Observation của tool là nguồn dữ kiện thứ hai, ngang hàng chunk RAG. Không đưa
        # vào thì hotline/email do contact_support trả về bị chấm là bịa — đo được ở lần
        # chạy 2 case đầu: N02 tụt còn faithfulness 0.40 vì đúng lý do này.
        contexts = list(contexts) + list(record.get("tool_observations") or [])
        samples.append(
            JudgeSample(
                id=item["id"],
                question=item["question"],
                answer=answer,
                contexts=list(contexts),
                reference=item.get("reference", ""),
            )
        )
    return samples, skipped


def dry_run_answers(ragas_set: dict) -> dict[str, dict]:
    """Answer giả = reference. Đường ống chạy hết mà không cần run agent."""
    return {
        s["id"]: {"id": s["id"], "answer": s["reference"], "retrieved_contexts": s["gold_contexts"]}
        for s in ragas_set["samples"]
    }


def to_markdown(summary: dict) -> str:
    lines = [
        "# RAGAS lite — kết quả",
        "",
        f"- Bộ dữ liệu: `eval/ragas_set.json` version `{summary['dataset_version']}`",
        f"- Judge model: `{summary['judge_model']}`",
        f"- Chế độ contexts: **{summary['contexts_mode']}**"
        f" ({'cái model thực sự thấy' if summary['contexts_mode'] == 'retrieved' else 'cái model đáng lẽ phải thấy'})",
        f"- Sample chấm được: **{summary['scored']}/{summary['dataset_total']}**",
        f"- Gọi LLM thật: {summary['llm_calls']} · cache hit: {summary['cache_hits']}",
        "",
        "## Tổng hợp",
        "",
        "| Metric | Trung bình | Số sample có điểm | Không đo được |",
        "|---|---|---|---|",
    ]
    for metric in METRICS:
        agg = summary["aggregate"][metric]
        mean = "—" if agg["mean"] is None else f"{agg['mean']:.3f}"
        lines.append(f"| `{metric}` | {mean} | {agg['n']} | {agg['none']} |")

    lines += ["", "## Từng sample", "", "| id | faithfulness | relevancy | correctness | ghi chú |", "|---|---|---|---|---|"]
    for row in summary["samples"]:
        cells = []
        for metric in METRICS:
            score = row[metric]["score"]
            cells.append("—" if score is None else f"{score:.2f}")
        notes = "; ".join(row[m]["error"] for m in METRICS if row[m]["error"]) or ""
        lines.append(f"| {row['id']} | {cells[0]} | {cells[1]} | {cells[2]} | {notes} |")

    unsupported = [
        (row["id"], c["claim"])
        for row in summary["samples"]
        for c in row["faithfulness"]["detail"].get("claims", [])
        if isinstance(c, dict) and not c.get("supported")
    ]
    if unsupported:
        lines += ["", "## Claim không có căn cứ trong ngữ cảnh", ""]
        lines += [f"- **{sid}** — {claim}" for sid, claim in unsupported]

    if summary["skipped"]:
        lines += ["", "## Bỏ qua", ""] + [f"- {s}" for s in summary["skipped"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="run_ragas")
    parser.add_argument("--from-results", type=Path, default=EVAL_DIR / "results.json")
    parser.add_argument("--contexts", choices=["retrieved", "gold"], default="retrieved")
    parser.add_argument("--ids", nargs="*", help="chỉ chấm các sample id này")
    parser.add_argument("--dry-run", action="store_true", help="judge giả, 0 token")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "ragas.json")
    args = parser.parse_args()

    ragas_set = json.loads(RAGAS_SET.read_text(encoding="utf-8"))

    if args.dry_run:
        answers = dry_run_answers(ragas_set)
        contexts_mode = "gold"
        judge = Judge(complete=stub_complete, cache_path=None)
        judge_model = "stub (dry-run)"
    else:
        if not args.from_results.exists():
            raise SystemExit(
                f"Không thấy {args.from_results}. Chạy `eval/run_eval.py` trước, "
                "hoặc dùng --dry-run để smoke đường ống."
            )
        answers = load_answers(args.from_results)
        contexts_mode = args.contexts
        settings = JudgeSettings.from_env()
        judge = Judge(settings=settings, cache_path=None if args.no_cache else DEFAULT_CACHE)
        judge_model = settings.model

    samples, skipped = build_samples(ragas_set, answers, contexts_mode, args.ids)
    if not samples:
        raise SystemExit(f"Không có sample nào chấm được. Bỏ qua: {skipped}")

    rows = []
    for index, sample in enumerate(samples, 1):
        print(f"[{index}/{len(samples)}] {sample.id} … ", end="", flush=True)
        scores = judge.score_all(sample)
        rows.append(
            {
                "id": sample.id,
                **{
                    metric: {
                        "score": result.score,
                        "detail": result.detail,
                        "error": result.error,
                        "cached": result.cached,
                    }
                    for metric, result in scores.items()
                },
            }
        )
        shown = " ".join(
            f"{SHORT[m]}={'—' if scores[m].score is None else f'{scores[m].score:.2f}'}"
            for m in METRICS
        )
        print(shown)
        judge.save_cache()

    summary = {
        "dataset_version": ragas_set.get("version", "?"),
        "dataset_total": ragas_set.get("total", len(ragas_set["samples"])),
        "judge_model": judge_model,
        "contexts_mode": contexts_mode,
        "scored": len(rows),
        "llm_calls": judge.llm_calls,
        "cache_hits": judge.cache_hits,
        "skipped": skipped,
        "aggregate": {
            metric: {
                "mean": mean_ignoring_none([r[metric]["score"] for r in rows]),
                "n": sum(1 for r in rows if r[metric]["score"] is not None),
                "none": sum(1 for r in rows if r[metric]["score"] is None),
            }
            for metric in METRICS
        },
        "samples": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    md_path.write_text(to_markdown(summary), encoding="utf-8")

    print()
    for metric in METRICS:
        agg = summary["aggregate"][metric]
        mean = "—" if agg["mean"] is None else f"{agg['mean']:.3f}"
        print(f"{metric:20s} {mean}  ({agg['n']} có điểm, {agg['none']} không đo được)")
    print(f"\nGọi LLM: {judge.llm_calls} · cache hit: {judge.cache_hits}")
    if skipped:
        print(f"Bỏ qua {len(skipped)}: {skipped}")
    print(f"Chi tiết: {args.out} · {md_path}")


if __name__ == "__main__":
    main()
