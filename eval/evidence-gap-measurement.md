# Đo khoảng trống bằng chứng bắt buộc

Ngày audit: 2026-07-31 (Asia/Bangkok)

## 1. Độ trễ từ câu hỏi đến phản hồi đầu

### Định nghĩa

Độ trễ cần đo là khoảng thời gian từ timestamp đăng câu hỏi công khai đến
timestamp comment trả lời đầu tiên. Đây không phải độ trễ API hoặc thời gian
chatbot sinh câu trả lời.

### Nguồn đã kiểm tra

- `docs/brief-de-tai.md`: có 6 câu hỏi nhưng không có timestamp bài/comment.
- `AI-spec.md`: có Q01-Q06 nhưng thiếu URL bài, ngày đăng và timestamp phản hồi.
- `data/Data_FaceBook_ckean/ai_thuc_chien_facebook_feedback_clean.md`: dữ liệu đã
  gộp, viết lại và loại metadata theo từng bài/comment.
- Group nguồn `https://www.facebook.com/groups/2125430681651241`: tìm kiếm web
  công khai theo 4 câu nguyên văn không trả kết quả; phiên browser không khả dụng.

### Kết quả

| Chỉ số | Kết quả |
|---|---:|
| Mẫu có đủ timestamp hỏi + phản hồi đầu | **0** |
| Median | **Không tính được (n=0)** |
| P90 | **Không tính được (n=0)** |

Kết luận: đã thực hiện audit nhưng chưa thể đo latency vì artifact nguồn đã mất
timestamp. Không thay thế bằng số ước lượng.

## 2. Evidence chuẩn A

### Điều kiện rubric

- Ít nhất 20 người ngoài nhóm.
- Ít nhất 50% xác nhận pain.
- Có log đầy đủ câu hỏi và từng câu trả lời nguyên văn.

### Kết quả audit

| Chỉ số | Kết quả |
|---|---:|
| Người ngoài nhóm có log hợp lệ | **0/20** |
| Người xác nhận pain | **0** |
| Tỷ lệ xác nhận | **Không tính được (n=0)** |
| Evidence A | **CHƯA ĐẠT** |

Không dùng Q01-Q06 để tính Evidence A vì đó là mining công khai, không phải log
khảo sát. Không dùng câu mô phỏng M01-M05 vì không có người trả lời thật.

## Dữ liệu cần bổ sung để chốt hai phép đo

1. Với latency: URL/ID bài, timestamp câu hỏi và timestamp phản hồi đầu của từng
   mẫu; khi có dữ liệu sẽ tính median, p90 và n.
2. Với Evidence A: tối thiểu 20 dòng gồm tên hoặc mã người thử, vai trò, câu hỏi
   khảo sát, câu trả lời nguyên văn và cờ xác nhận pain.
