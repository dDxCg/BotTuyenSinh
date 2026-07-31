# RAGAS lite — kết quả

- Bộ dữ liệu: `eval/ragas_set.json` version `2026-07-30c`
- Judge model: `google/gemini-3.5-flash-lite`
- Chế độ contexts: **retrieved** (cái model thực sự thấy)
- Sample chấm được: **23/23**
- Gọi LLM thật: 44 · cache hit: 36

## Tổng hợp

| Metric | Trung bình | Số sample có điểm | Không đo được |
|---|---|---|---|
| `faithfulness` | 0.914 | 23 | 0 |
| `answer_relevancy` | 0.761 | 23 | 0 |
| `answer_correctness` | 0.646 | 23 | 0 |

## Từng sample

| id | faithfulness | relevancy | correctness | ghi chú |
|---|---|---|---|---|
| S01 | 1.00 | 1.00 | 0.80 |  |
| S02 | 1.00 | 1.00 | 1.00 |  |
| S03 | 0.89 | 1.00 | 1.00 |  |
| S04 | 1.00 | 0.00 | 0.00 |  |
| S05 | 1.00 | 1.00 | 0.80 |  |
| M01 | 1.00 | 1.00 | 0.80 |  |
| M02 | 0.83 | 1.00 | 0.50 |  |
| M03 | 0.86 | 1.00 | 0.80 |  |
| M05 | 1.00 | 1.00 | 1.00 |  |
| M06 | 0.60 | 0.00 | 0.00 |  |
| M07 | 0.67 | 0.50 | 0.30 |  |
| N01 | 1.00 | 1.00 | 0.85 |  |
| N02 | 1.00 | 0.00 | 0.00 |  |
| N03 | 1.00 | 1.00 | 0.80 |  |
| N04 | 1.00 | 0.00 | 0.00 |  |
| N05 | 1.00 | 0.00 | 0.00 |  |
| N06 | 1.00 | 1.00 | 1.00 |  |
| N07 | 1.00 | 1.00 | 0.60 |  |
| N08 | 1.00 | 1.00 | 1.00 |  |
| N09 | 0.67 | 1.00 | 0.80 |  |
| N10 | 1.00 | 1.00 | 1.00 |  |
| N11 | 0.50 | 1.00 | 1.00 |  |
| N12 | 1.00 | 1.00 | 0.80 |  |

## Claim không có căn cứ trong ngữ cảnh

- **S03** — Chương trình AI Thực Chiến của VinUni có ba track chính.
- **M02** — Thông tin về lộ trình được lấy từ nguồn không chính thức.
- **M03** — Thông tin về lịch học hằng ngày được lấy từ phản hồi của cộng đồng học viên trên Facebook.
- **M06** — Người trả lời đã chuyển câu hỏi về thời gian học đến bộ phận tuyển sinh để xác nhận sự khác biệt giữa trang chính thức và nhóm Facebook.
- **M06** — Bộ phận tuyển sinh sẽ cho biết nên tin bên nào nếu có sự khác biệt về thông tin thời gian học.
- **M07** — Thông tin về track AI Engineer không có trong ngữ cảnh hiện tại.
- **N09** — Không có thông tin cụ thể về tất cả các công ty tham gia và địa điểm thực chiến.
- **N11** — Chương trình học kéo dài 3 tháng.
