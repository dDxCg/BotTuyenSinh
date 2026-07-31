# Báo cáo đánh giá hệ thống

Ngày chạy gần nhất: 2026-07-31
Môi trường: Windows, Python 3.10  
Kết luận: **PASS — toàn bộ test offline và RAG local hoạt động ổn định.**

## 1. Kết quả test

Lệnh:

```powershell
python -m pytest tests -q --durations=10
```

Kết quả:

| Chỉ số | Giá trị |
| --- | ---: |
| Passed | 113 |
| Failed | 0 |
| Deselected | 11 |
| Thời gian | 4,65 giây |

`11 deselected` thuộc marker `live` và `e2e`, bị loại theo cấu hình mặc định vì gọi chat API thật. Đây không phải test lỗi.

Test chậm nhất:

| Test | Thời gian |
| --- | ---: |
| `test_http_health_page_chat_and_reset` | 0,68 giây |
| `test_http_rejects_empty_message` | 0,52 giây |
| `test_ui_path_and_static_route_are_confined_to_ui_folder` | 0,51 giây |

Các nhóm đã kiểm tra gồm chatbot, ReAct loop, prompt, RAG bridge, tool registry,
source attachment, contact support, demo service, HTTP API và static asset của UI
(banner WebP, ảnh chương trình, footer, mascot SVG và MIME type tương ứng).

UI có thêm kiểm thử hồi quy cho luồng streaming: phần nội dung đang gõ được render
dưới dạng text an toàn; dòng bảng Markdown chưa hoàn chỉnh luôn được tiêu thụ để
không thể khóa main thread. Luồng browser với câu hỏi tuyển sinh kết thúc bình
thường, nút `Dừng tạo` được ẩn và không còn cursor sau khi hoàn tất.

## 2. Kiểm tra RAG local

### Schema và ChromaDB

```powershell
python src\rag\embedding.py --validate-only
```

| Kiểm tra | Kết quả |
| --- | --- |
| `chunks.json` | Hợp lệ |
| Số chunks | 82 |
| Số records Chroma | 82 |
| Model | `intfloat/multilingual-e5-large` |
| Vector dimension | 1.024 |
| Distance metric | cosine |
| Embedding backend | local-only |

### Smoke retrieval thật

Câu hỏi:

```text
Chương trình học trong bao lâu?
```

Kết quả top 1:

| Field | Giá trị |
| --- | --- |
| Chunk ID | `chunk_9286a7121125a5ea` |
| Nội dung | Lộ trình 12 tuần, mô hình 3+3+6 |
| Cosine similarity | `0.848624` |
| Model | `intfloat/multilingual-e5-large` local |
| Cold-start retrieval | 19,4 giây |

Kết quả đúng ngữ nghĩa câu hỏi. Metadata giữ đủ `source_link`, `loai_nguon`, `source_file` và `embedding_model` cho tầng chatbot/tool.

## 3. Kiểm tra kỹ thuật

```powershell
python -m compileall -q src eval
```

- Compile thành công, không có syntax error.
- Không còn reference tới `local_embedding.py` đã xóa.
- Không còn embedding API, API key embedding, HTTP embedding hoặc lexical fallback.
- `retrieval.py` gọi trực tiếp `embedding.embed_query()`.
- `rag_bridge.py` chuyển payload RAG sang `Chunk` chatbot và giữ registry nguồn cho `attach_source_link`.

## 4. Đánh giá

| Thành phần | Trạng thái |
| --- | --- |
| Chunking/JSON contract | PASS |
| E5-large local embedding | PASS |
| Chroma indexing/retrieval | PASS |
| RAG ↔ chatbot bridge | PASS |
| Tool `attach_source_link` | PASS |
| Tool `contact_support` | PASS |
| ReAct orchestration | PASS |
| Demo HTTP API | PASS |
| Chat provider thật | PASS — browser smoke 1 câu hỏi lõi |

## 5. Browser smoke end-to-end

Luồng chạy thật ngày 2026-07-31:

```text
prototype.html (root) → POST /api/chat → E5-large local → ChromaDB
→ model chat đã cấu hình → source attachment → UI
```

| Kiểm tra | Kết quả |
| --- | --- |
| Câu hỏi | `Chương trình học trong bao lâu?` |
| Đáp án | `Chương trình học kéo dài 12 tuần.` |
| Nguồn hiển thị | `20K AI Handbook — VinUni` |
| URL nguồn | PDF chính thức VinUni |
| Reset hội thoại | PASS — còn 0 user message, 1 greeting, 0 source cũ |
| Mobile | PASS — panel đúng viewport 390 × 844, composer nằm trong viewport |

Lần chạy đầu phát hiện nguồn Facebook được hiện như URL trần dù có nguồn
chính thức gần tương đương. Hệ thống đã được sửa để:

1. Ưu tiên nguồn chính thức khi similarity cách top không quá `0.03`.
2. Giữ `source_type`, nhãn và cảnh báo ở backend để trace/test.
3. Theo quyết định UI mới, chỉ hiện `Nguồn tham khảo` + URL ở cuối bubble,
   sau khi câu trả lời stream xong; không hiện metadata cho người dùng.

Hai unit test hồi quy mới bảo vệ việc ưu tiên nguồn chính thức và giữ cảnh báo
cho nguồn cộng đồng.

Hệ thống đủ điều kiện chạy demo offline ở tầng RAG. Muốn xác nhận model chat/provider thật, chạy riêng `pytest -m live` hoặc `pytest -m e2e`; các lệnh này cần `OPENAI_API` và có thể phát sinh chi phí/network.
