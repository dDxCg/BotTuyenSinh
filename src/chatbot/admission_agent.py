"""Lắp 3 tầng: RAG (ChromaDB) + 2 tool tuyển sinh + vòng ReAct.

Luồng theo docs/design-agent-tools.md §1:
    câu hỏi -> retrieval -> đủ căn cứ ? trả lời + attach_source_link
                          : contact_support (không đoán)
"""

from .agent_tools import build_registry
from .chatbot import Chatbot
from .config import Settings
from .rag_bridge import NO_GROUNDING_THRESHOLD, ChromaRetriever
from .react import ReActAgent

# Danh sách cấm cụ thể — docs/design-agent-tools.md §2 yêu cầu nêu thành case rõ ràng,
# không mô tả chung chung, để model nhận diện được.
ADMISSION_POLICY = f"""Bạn tư vấn cho người đang tìm hiểu chương trình AI Thực Chiến, chưa nộp hồ sơ.

Nguyên tắc sống: đưa dữ kiện, không kết luận thay người dùng.

Bắt buộc gọi `contact_support` (không được tự trả lời) khi:
- reason='no_grounding': mục Ngữ cảnh truy xuất báo KHÔNG đủ căn cứ (score cao nhất < {NO_GROUNDING_THRESHOLD}).
- reason='out_of_scope': hỏi trạng thái hồ sơ / điểm / kết quả cá nhân (kể cả khi người dùng
  đưa email hay mã hồ sơ); xin lời khuyên "có nên nộp không", "có đậu không"; hỏi cam kết
  đầu ra hay thu nhập sau khoá.
- reason='conflicting_sources': hai chunk mâu thuẫn về cùng một dữ kiện và không có căn cứ
  chọn bên nào. Đưa cả hai dữ kiện qua partial_context, mỗi dòng một dữ kiện — không tự chọn giúp.
- reason='personal_data_request': hỏi thông tin cá nhân/nhạy cảm ngoài phạm vi tuyển sinh công khai.

Khi trả lời được: luôn gọi `attach_source_link` với chunk_ids đã dùng trước khi chốt Final Answer,
và đưa nguyên phần nguồn đó vào câu trả lời. Dữ kiện từ Facebook phải kèm cảnh báo không phải
nguồn chính thức; khi lệch với nguồn chính thức thì ưu tiên nguồn chính thức."""


def build_admission_agent(
    settings: Settings | None = None,
    retriever: ChromaRetriever | None = None,
    max_steps: int = 6,
    top_k: int = 5,
) -> ReActAgent:
    """Agent tư vấn tuyển sinh, đã nối đủ RAG + 2 tool."""
    retriever = retriever or ChromaRetriever()
    registry = build_registry(retriever)
    bot = Chatbot(
        settings=settings,
        retriever=retriever,
        context=ADMISSION_POLICY,
        top_k=top_k,
        grounding_threshold=NO_GROUNDING_THRESHOLD,
    )
    # Không còn tool tìm kiếm, nên chunk phải được đổ sẵn vào prompt.
    return ReActAgent(registry, chatbot=bot, max_steps=max_steps, prefetch_rag=True)


__all__ = ["ADMISSION_POLICY", "build_admission_agent"]
