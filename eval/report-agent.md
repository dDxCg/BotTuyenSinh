# Báo cáo eval — agent

Ngày 2026-07-31 · `openai/gpt-4o-mini` + RAG ChromaDB + ReAct
Dữ liệu thô: `eval/results/full-run-t080.json` · Bộ case: `eval/questions.json` v`2026-07-31`

## Bộ test có gì

**45 case**, mỗi case chấm bằng 8 loại check tất định (chuỗi bắt buộc/cấm, tool bắt buộc/cấm, số bước, không chạm trần).

| Chiều | Phân bố |
|---|---|
| Category | in_scope 19 · out_of_scope 12 · no_grounding 5 · safety 5 · conflicting 2 · source_labeling 1 · mixed_scope 1 |
| Difficulty | easy 14 · medium 17 · hard 14 |
| Nguồn | tự viết 35 · **chatlog người dùng thật 10** |
| Gold chunk | 23 case có gold (15 multi-hop) · 22 case negative |

## Kết quả: 29/45 đạt (64%)

| Status | Số |
|---|---|
| `as_expected` | 39 |
| `regression` | 4 — S04, M02, M06, M07 |
| `improvement` | 2 — S10, M03 |

| Category | Đạt | | Difficulty | Đạt |
|---|---|---|---|---|
| no_grounding | 5/5 | | easy | 11/14 |
| conflicting | 1/2 | | medium | 9/17 |
| source_labeling | 1/1 | | hard | 9/14 |
| mixed_scope | 1/1 | | | |
| safety | 5/5 | | | |
| in_scope | 11/19 | | | |
| out_of_scope | 5/12 | | | |

**Điểm mạnh:** an toàn (safety 5/5) và chống bịa khi hoàn toàn không có căn cứ (no_grounding 5/5).
**Điểm yếu:** out_of_scope 5/12 và in_scope 11/19.

---

## 16 case trượt — nguyên nhân

### 1. Từ chối câu trả lời được — 3 case: S04, M06, N02

Cả ba **có gold chunk**, dữ kiện nằm sẵn trong corpus, score đều **trên** ngưỡng. Agent vẫn trả lời *"Mình chưa tìm thấy dữ kiện đủ căn cứ…"*.

Không phải grounding gate chặn — model tự chọn từ chối. N02 còn dao động: lượt chạy trước nó trả lời đúng, lượt này từ chối (`temperature=0.3`).

### 2. Thiếu tool bắt buộc — 4 case: S01, M01, M04, M08

S01/M01 trả lời đúng nội dung nhưng không gọi `attach_source_link` ⇒ mất trích nguồn.
M04/M08 là câu ngoài phạm vi mà không gọi `contact_support` ⇒ tự trả lời thay vì chuyển người.

### 3. Không chuyển người với câu chatlog thật — 3 case: C02, C06, C09

"bây h là mấy giờ", "phóng to slide thế nào để full màn", "tóm tắt nội dung chính trong slide này". Đây là câu người dùng thật gõ, agent tự trả lời.

**Abstention correctness: 14/22 (64%)** — chỉ số yếu nhất của sản phẩm.

### 4. Thiếu dữ kiện trong câu trả lời — 3 case: M02, N03, N05

M02 gọi `attach_source_link` 2 lần nhưng câu trả lời **không có URL**, dù người dùng hỏi thẳng link.
N03 thiếu "miễn học phí", N05 thiếu cấu hình laptop tối thiểu.

### 5. Vòng lặp tool, chạm trần `max_steps` — 2 case: M07, N09

Gọi `attach_source_link` **6 lần liên tiếp** rồi hết bước. M04 (nhóm 2) cũng 6 lần.

Nguyên nhân: repeat-guard `react.py:127` chỉ chặn khi **tham số trùng**, model đổi `chunk_ids` mỗi lượt nên lách được.

### 6. Gọi tool bị cấm — 1 case: N04

Câu có đủ căn cứ nhưng vẫn gọi `contact_support`.

---

## Ngưỡng grounding không dùng được

Quét toàn dải trên 45 case: **không tồn tại ngưỡng nào tách được** in_scope khỏi negative.

- Điểm tách tốt nhất (0.83): FPR vẫn **41%**
- **26/45 case nằm chen chúc trong dải 0.82–0.86**
- M03 (in_scope) và A01 (negative) **cùng score 0.8589**

E5 nén cosine quá chặt. Phải đổi cơ chế — score gap `top1−top5`, chuẩn hoá phân vị, hoặc rerank — chứ không phải chỉnh số.

---

## Metric hành vi

| | |
|---|---|
| Latency p50 / p90 / max | 4.9 / 13.2 / 30.6 s |
| Tổng 45 case | 292 s |
| Lượt gọi tool | `attach_source_link` 66 · `contact_support` 20 |
| Case không gọi tool | 14/45 |
| Chạm trần `max_steps` | 2 |

---

## Hai lỗi của chính bộ test đã sửa

1. **`answer_none` dính oan** — `contact_support` chép nguyên câu hỏi vào `suggested_question`, nên chuỗi cấm lấy từ câu hỏi luôn khớp. Thêm `strip_question_echo()`. Trả lại đúng cho S06, M05, A01, A03.
2. **Câu từ chối được tính là đạt** — thêm chốt chặn `"chưa tìm thấy dữ kiện"` / `"đã chuyển câu hỏi"` cho 22 case phải trả lời được. Lộ ra 3 case đạt giả: S04, M06, N02.

Sau khi sửa, **không còn case nào trượt do lỗi bài test**. 25 case `expect: unknown` đã chốt thành pass/fail ⇒ lần chạy sau bắt được regression thật.

## Ưu tiên sửa

1. Bỏ ngưỡng tuyệt đối, thử score gap
2. Chặn lặp `attach_source_link` theo tên tool — gỡ 3 case
3. Giảm từ chối oan (S04, M06, N02)
4. Nâng abstention nhóm chatlog (14/22)
