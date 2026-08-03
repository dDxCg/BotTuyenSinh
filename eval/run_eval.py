"""Chạy eval test set qua graph thật, chấm bằng RAGAS + tool calling accuracy.

uv run python -m eval.run_eval
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness

from src.chatbot.config import Settings
from src.chatbot.graph import build_graph, run_graph
from src.rag.rag_bridge import PgVectorRetriever

ROOT = Path(__file__).resolve().parents[1]
TESTSET_FILE = ROOT / "data" / "eval_testset.json"
RESULTS_FILE = ROOT / "data" / "eval_results.json"


def run_testset(testset: list[dict], settings: Settings) -> list[dict]:
    serde = JsonPlusSerializer(allowed_msgpack_modules=[("src.chatbot.types", "Chunk")])
    checkpointer = InMemorySaver(serde=serde)
    graph = build_graph(settings=settings, retriever=PgVectorRetriever(), checkpointer=checkpointer)

    results: list[dict] = []
    for item in testset:
        thread_id = f"eval-{uuid.uuid4()}"
        state = run_graph(graph, item["question"], thread_id=thread_id)
        results.append(
            {
                "question": item["question"],
                "ground_truth": item["ground_truth"],
                "expected_tool": item["expected_tool"],
                "answer": state["final_answer"],
                "contexts": [chunk.text for chunk in state["retrieved"]] or item["contexts"],
                "actual_tools": state.get("recent_tools", []),
            }
        )
    return results


def tool_calling_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["expected_tool"] in r["actual_tools"])
    return correct / len(results)


def run_ragas(results: list[dict], settings: Settings) -> dict:
    scored = [r for r in results if r["contexts"] and r["ground_truth"]]
    if not scored:
        return {}

    judge_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.judge_model,
            api_key=settings.judge_api_key,
            base_url=settings.judge_base_url,
        )
    )
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"],
                "reference": r["ground_truth"],
            }
            for r in scored
        ]
    )
    scores = evaluate(
        dataset,
        metrics=[Faithfulness(), ContextPrecision(), ContextRecall()],
        llm=judge_llm,
    )
    return scores.to_pandas().mean(numeric_only=True).to_dict()


def main() -> None:
    testset = json.loads(TESTSET_FILE.read_text(encoding="utf-8"))
    settings = Settings.from_env()

    results = run_testset(testset, settings)
    ragas_scores = run_ragas(results, settings)
    accuracy = tool_calling_accuracy(results)

    RESULTS_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print("== RAGAS ==")
    for name, value in ragas_scores.items():
        print(f"{name}: {value:.3f}")
    print("== Tool calling accuracy ==")
    print(f"tool_calling_accuracy: {accuracy:.3f}")
    print(f"Chi tiết từng câu -> {RESULTS_FILE}")


if __name__ == "__main__":
    main()
