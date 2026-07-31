# Báo cáo eval — RAGAS lite

Ngày 2026-07-31 · Judge `google/gemini-3.5-flash-lite` (khác model sản phẩm `openai/gpt-4o-mini`)
Dữ liệu thô: `eval/results/ragas-t080.json` · Bộ sample: `eval/ragas_set.json`

## Bộ test có gì

**23 sample**, lọc từ `questions.json` bằng quy tắc cơ học `gold_chunk_ids != []`.

22 case negative bị loại có lý do: khi trả lời **đúng**, chúng trả về template `contact_support` — cùng một chuỗi hotline. Chấm `faithfulness` lên đó ra ~0 với case đúng, metric quay ngược đầu. Nhóm đó đo bằng abstention correctness ở báo cáo agent.

Mỗi sample có: `question` · `reference` (đáp án chuẩn viết tay) · `gold_chunk_ids` + `gold_contexts` · `gold_source_types`.

Ba metric tự cài (không dùng package `ragas` — nó kéo 92 dependency):

| Metric | Cách tính |
|---|---|
| `faithfulness` | tách answer thành claim, kiểm từng claim có suy ra được từ context không. Điểm = claim được hỗ trợ / tổng |
| `answer_relevancy` | 0–1, answer có đúng trọng tâm câu hỏi không |
| `answer_correctness` | 0–1, answer so với `reference`, tính cả dữ kiện sai lẫn thiếu |

Context đem chấm gồm **chunk RAG + Observation của tool** — thiếu vế sau thì hotline/email do `contact_support` trả về bị chấm là bịa.

## Kết quả — chấm đủ 23/23, không lỗi

| Metric | Trung bình |
|---|---|
| `faithfulness` | **0.914** |
| `answer_relevancy` | **0.761** |
| `answer_correctness` | **0.646** |

| id | check | faith | relev | corr |
|---|---|---|---|---|
| S01 | ❌ | 1.00 | 1.00 | 0.80 |
| S02 | ✅ | 1.00 | 1.00 | 1.00 |
| S03 | ✅ | 0.89 | 1.00 | 1.00 |
| S04 | ❌ | 1.00 | **0.00** | **0.00** |
| S05 | ✅ | 1.00 | 1.00 | 0.80 |
| M01 | ❌ | 1.00 | 1.00 | 0.80 |
| M02 | ❌ | 0.83 | 1.00 | 0.50 |
| M03 | ✅ | 0.86 | 1.00 | 0.80 |
| M05 | ✅ | 1.00 | 1.00 | 1.00 |
| M06 | ❌ | 0.60 | **0.00** | **0.00** |
| M07 | ❌ | 0.67 | 0.50 | 0.30 |
| N01 | ✅ | 1.00 | 1.00 | 0.85 |
| N02 | ❌ | 1.00 | **0.00** | **0.00** |
| N03 | ❌ | 1.00 | 1.00 | 0.80 |
| N04 | ❌ | 1.00 | **0.00** | **0.00** |
| N05 | ❌ | 1.00 | **0.00** | **0.00** |
| N06 | ✅ | 1.00 | 1.00 | 1.00 |
| N07 | ✅ | 1.00 | 1.00 | 0.60 |
| N08 | ✅ | 1.00 | 1.00 | 1.00 |
| N09 | ❌ | 0.67 | 1.00 | 0.80 |
| N10 | ✅ | 1.00 | 1.00 | 1.00 |
| N11 | ✅ | 0.50 | 1.00 | 1.00 |
| N12 | ✅ | 1.00 | 1.00 | 0.80 |

---

## Nguyên nhân điểm thấp

### 5 case `relevancy` = 0.00 — agent từ chối trả lời

S04, M06, N02, N04, N05. Tất cả **có gold chunk**, dữ kiện nằm sẵn trong corpus. Judge ghi lý do:

> *"Câu trả lời né tránh việc trả lời câu hỏi, chỉ cung cấp kênh liên hệ và thông báo không tìm thấy dữ kiện."*

Đây là dạng lỗi tốn kém nhất: agent có đủ thông tin mà không dùng.

### `answer_correctness` 0.646 — thấp nhất, và đúng chỗ

Judge hầu như **không tìm thấy dữ kiện sai**. Điểm trừ đến từ **thiếu**:

- **M07** (0.30) — nói "track AI Engineer không có trong ngữ cảnh". Thực tế `AI Engineer` **chính là** `AI Applications`, cùng một track, ghi rõ trong `chunk_116a812d982820d9`. Agent không nhận ra hai tên gọi của cùng một thứ.
- **M02** (0.50) — có nội dung, thiếu URL nguồn dù đã gọi `attach_source_link`
- **N07** (0.60) — thiếu bước xếp lớp theo trình độ sau 2 vòng tuyển

Đây là loại lỗi bài kiểm tra chuỗi không bao giờ đo được: **đúng nhưng không đủ**.

### 2 case `faithfulness` thấp — có lý do thật

- **M06** (0.60) — câu trả lời hứa *"đã chuyển câu hỏi đến bộ phận tuyển sinh"*, việc này không có trong context
- **N11** (0.50) — quy đổi "12 tuần = 3 tháng". Phép quy đổi đúng nhưng "3 tháng" không có trong tài liệu

N11 là giới hạn cố hữu của metric: suy luận số học đúng vẫn bị tính là không có căn cứ. Không cần sửa, cần biết khi đọc số.

---

## Giá trị lớn nhất: bắt được thứ check chuỗi bỏ lọt

Trước khi sửa bộ case, **3 case qua bài kiểm tra chuỗi nhưng relevancy 0.00**: S04, M06, N02. Câu từ chối vẫn thoả `answer_none` (không chứa chuỗi cấm) và `answer_any` (khớp từ chung chung).

Đã sửa `questions.json` theo phát hiện này — thêm chốt chặn câu từ chối cho 22 case phải trả lời được. Ba case đó giờ đã thành FAIL đúng.

Đối chiếu hai tầng đo sau khi sửa:

| | Trước | Sau |
|---|---|---|
| Hai tầng khớp nhau | 16/23 | **18/23** |
| check PASS nhưng RAGAS ~0 | 3 | **0** |

5 case còn lệch (S01, M01, M02, N03, N09) đều là "check FAIL, RAGAS tốt" — **đúng phân công**: check bắt lỗi tool/format/thiếu dữ kiện cụ thể mà judge không quan tâm; judge bắt nội dung rỗng mà chuỗi không thấy. Cần cả hai.

---

## Chưa làm

**Mutation check cho chính metric.** Chưa bơm answer bịa để xác nhận `faithfulness` tụt. Phân tán hiện tại (0.50 → 1.00) là tín hiệu judge phân biệt được, chưa phải bằng chứng.
