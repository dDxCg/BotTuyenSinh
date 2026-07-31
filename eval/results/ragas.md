# RAGAS lite — kết quả

- Bộ dữ liệu: `eval/ragas_set.json` version `2026-07-30c`
- Judge model: `google/gemini-3.6-flash`
- Chế độ contexts: **retrieved** (cái model thực sự thấy)
- Sample chấm được: **2/23**
- Gọi LLM thật: 0 · cache hit: 6

## Tổng hợp

| Metric | Trung bình | Số sample có điểm | Không đo được |
|---|---|---|---|
| `faithfulness` | 0.700 | 2 | 0 |
| `answer_relevancy` | 1.000 | 2 | 0 |
| `answer_correctness` | 0.750 | 2 | 0 |

## Từng sample

| id | faithfulness | relevancy | correctness | ghi chú |
|---|---|---|---|---|
| S02 | 1.00 | 1.00 | 1.00 |  |
| N02 | 0.40 | 1.00 | 0.50 |  |

## Claim không có căn cứ trong ngữ cảnh

- **N02** — Số hotline liên hệ của bộ phận tuyển sinh là 0979.489.846.
- **N02** — Số điện thoại liên hệ của Ms. Phương Thảo thuộc bộ phận tuyển sinh là 0388.339.478.
- **N02** — Địa chỉ email liên hệ của bộ phận tuyển sinh là AIthucchien@vinuni.edu.vn.
