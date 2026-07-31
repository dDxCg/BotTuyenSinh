# Benchmark embedding local — multilingual-e5-large

Ngày chạy: 2026-07-30T19:40:58+07:00 · Top-k: 5

Phạm vi: model local, query embedding và Chroma retrieval; **không** đo chat LLM.

## Tổng hợp

| Model embedding | Model load (s) | Embedding mean (s) | Median (s) | P95 (s) | Total retrieval mean (s) | Thành công |
|---|---:|---:|---:|---:|---:|---:|
| `intfloat/multilingual-e5-large` | 29.007 | 0.442 | 0.350 | 1.080 | 0.446 | 10/10 |

## Chi tiết

| # | Câu hỏi | Local embedding (s) | Chroma query (s) | Total (s) | Top chunk |
|---:|---|---:|---:|---:|---|
| 1 | Chương trình học trong bao lâu? | 1.080 | 0.005 | 1.085 | chunk_9286a7121125a5ea |
| 2 | Lịch học hằng ngày như thế nào? | 0.361 | 0.005 | 0.367 | chunk_a5053f371cbe59c1 |
| 3 | Địa điểm học ở đâu? | 0.294 | 0.005 | 0.298 | chunk_6a231d6360e44323 |
| 4 | Điều kiện dự tuyển là gì? | 0.328 | 0.003 | 0.331 | chunk_8859b11be20d1d08 |
| 5 | Hồ sơ đăng ký gồm những gì? | 0.338 | 0.002 | 0.341 | chunk_6052aa2c8b2a9338 |
| 6 | Bài đánh giá năng lực kiểm tra nội dung gì? | 0.495 | 0.004 | 0.498 | chunk_1857a1bbe80d8eab |
| 7 | Chương trình có những track nào? | 0.310 | 0.005 | 0.315 | chunk_a787c7ad6db8eec3 |
| 8 | Có thể vừa học vừa đi làm không? | 0.322 | 0.006 | 0.328 | chunk_1d2406442beb2481 |
| 9 | Hạn nộp hồ sơ khóa đang tuyển là khi nào? | 0.511 | 0.003 | 0.514 | chunk_868464c016258eec |
| 10 | Học viên có được hỗ trợ học phí không? | 0.379 | 0.007 | 0.387 | chunk_a92bd5324bc0c968 |

## Ghi chú đo

- `model_load`: nạp weights local một lần khi process khởi động.
- `embedding`: biến câu hỏi thành vector 1.024 chiều bằng E5-large local.
- `total retrieval`: embedding + Chroma query; chưa gồm chat LLM.
