# Eval report — RAGAS + tool calling accuracy

Chạy: `uv run python -m eval.build_testset --n 20` → `uv run python -m eval.run_eval`
Test set: 23 câu (19 sinh từ chunks.json bởi JudgeLLM (`google/gemini-3.5-flash-lite`) + 4 câu
`contact_support` hard-code). Chạy qua graph thật (`build_graph`/`run_graph`), judge model chấm
faithfulness/context_precision/context_recall qua RAGAS.

## Kết quả

| Metric | Score |
|---|---|
| faithfulness | 0.856 |
| context_precision | 0.974 |
| context_recall | 1.000 |
| tool_calling_accuracy | 0.565 (13/23) |

`eval/../data/eval_results.json` — chi tiết từng câu (question/answer/contexts/actual_tools).

## Nhận xét

**Retrieval tốt** — context_precision 0.974, context_recall 1.000: chunk đúng luôn nằm trong top-k,
gần như không nhiễu. Không phải vấn đề của retrieval.

**Tool calling accuracy thấp (0.565) — nguyên nhân chính:**

1. **8/23 câu answer rỗng (`""`), `actual_tools=[]`** — agent không gọi tool nào (không
   `attach_source_link`, không `contact_support`) và bị `force_finalize` cắt ngang (do lặp lại
   tool 3 lần liên tiếp — `MAX_CONSECUTIVE_SAME_TOOL` trong `nodes/tools.py:12`) hoặc agent tự
   dừng mà không tool_call. Kết quả: user nhận câu trả lời trống dù context đủ căn cứ
   (`grounded=True` — context_recall confirm nội dung có trong chunk). Đây là **bug thật**, không
   phải giới hạn model — cần xem log chi tiết (đã thấy 1 case: agent lặp gọi tool sai tham số 3
   lần rồi bị force_finalize với answer rỗng thay vì fallback trả lời thẳng bằng context có sẵn).
2. **2/4 câu `contact_support` bị gọi nhầm thành `attach_source_link`** — "Tôi muốn liên hệ phòng
   tuyển sinh thì gọi số nào?" và "Nếu có thắc mắc ngoài những gì đã hỏi thì liên hệ ai?": model
   coi đây là câu hỏi tra cứu thông tin công khai (đúng — số hotline nằm sẵn trong chunk, không
   phải personal data) nên trả lời trực tiếp + `attach_source_link`, không rơi vào
   `out_of_scope`/`no_grounding`. **Đây có thể là lỗi thiết kế test set** (2/4 hard-code
   `contact_support` question thực ra tra cứu được bằng RAG, không thật sự cần
   `contact_support`) chứ không phải lỗi agent — nên review lại nhãn `expected_tool` này.

**Faithfulness 0.856** — vẫn có ~14% câu trả lời chứa chi tiết không bám sát context. Case rõ nhất:
câu "Học viên tham gia chương trình được nhận hỗ trợ tài chính như thế nào?" — model lặp lại
cùng 1 khối gạch đầu dòng ("Có cơ hội..." /"Ứng tuyển...") hàng chục lần liên tiếp cho tới khi cắt
ở max_tokens — **bug lặp vòng sinh text** (không phải hallucination nội dung, nhưng làm giảm
faithfulness score và chất lượng câu trả lời rõ rệt). Nên xem lại `OPENAI_MAX_TOKENS`/stop
condition hoặc lỗi ở tầng agent loop khi nối message.

## Đề xuất tiếp theo

- Sửa `force_finalize` (`nodes/finalize.py`) để khi bị cắt do lặp tool, vẫn cố trả lời từ
  context đã retrieve thay vì trả rỗng.
- Review 2 câu `contact_support` bị gán nhãn sai trong test set (không phải bug code).
- Điều tra case lặp vòng sinh câu trả lời (component nào giữ history/max_tokens gây lặp).
