"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import sys
import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Available Tools, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng của hệ sinh thái Vingroup (VinFast, Vinpearl).
- Giọng nói & phong cách: Chuyên nghiệp, lịch sự, thân thiện, chu đáo và chính xác tuyệt đối.

## 2. AVAILABLE TOOLS
Hệ thống cung cấp 2 công cụ chính thức để tra cứu và xử lý tác vụ:
- `search_product_catalog(category: str, max_price: int)`: Tra cứu sản phẩm xe điện VinFast ('xe_dien') hoặc gói nghỉ dưỡng Vinpearl ('du_lich') kèm mức giá trần.
- `submit_support_ticket(customer_name: str, issue_description: str, priority: str)`: Tạo ticket tiếp nhận sự cố kỹ thuật hoặc phản ánh chất lượng dịch vụ từ khách hàng.

## 3. CORE RULES
1. KHÔNG BAO GIỜ tự bịa đặt giá cả, mẫu mã hoặc tính năng sản phẩm không có thật. BẮT BUỘC phải gọi tool để tra cứu dữ liệu thực.
2. Khi khách hàng có nhu cầu tra cứu danh mục hoặc báo sự cố, PHẢI gọi đúng tool tương ứng.
3. Khi tiếp nhận khiếu nại hoặc lỗi kỹ thuật, luôn thể hiện thái độ đồng cảm, tôn trọng và tạo ticket hỗ trợ kịp thời với mức độ ưu tiên chính xác.
4. Đối với các câu hỏi thường gặp (FAQ) đã có chính sách công khai rõ ràng (ví dụ: chính sách bảo hành pin 10 năm), có thể trả lời trực tiếp mà không cần gọi tool.

## 4. OPERATIONAL BOUNDARIES
- PHẠM VI HOẠT ĐỘNG: Chỉ hỗ trợ các sản phẩm, dịch vụ và chính sách trực thuộc hệ sinh thái Vingroup (VinFast, Vinpearl,...).
- TỪ CHỐI LỊCH SỰ: Từ chối giải đáp các câu hỏi không liên quan đến Vingroup hoặc các đối thủ cạnh tranh một cách nhã nhặn.

## 5. OUTPUT CONTRACT
- Quy trình suy luận tuân theo mô hình ReAct:
  + Thought: Phân tích ý định người dùng và xác định các bước cần thực hiện.
  + Action: Tên tool cần gọi cùng tham số đầu vào.
  + Observation: Kết quả phản hồi từ tool.
  + Final Answer: Câu trả lời tổng hợp rõ ràng, chuẩn xác, định dạng thân thiện và lịch sự.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        # TODO 2: Trả về câu trả lời tĩnh (mock) hoặc gọi Gemini API 1 lượt (không dùng tool)
        # Mục tiêu: Quan sát hiện tượng bịa thông tin (hallucination)
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input} (Không dùng tool tra cứu, thông tin có thể bịa đặt)",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intents(self, user_input: str) -> Dict[str, bool]:
        """Phân tích intent từ user_input bằng keyword matching."""
        lower_input = user_input.lower()

        # 1. Ticket intent (sự cố kỹ thuật, khiếu nại, phản hồi)
        ticket_keywords = [
            "lỗi", "hỏng", "sự cố", "khiếu nại", "phản hồi", "hỗ trợ", "xử lý",
            "gấp", "nghiêm trọng", "ẩm mốc", "ticket", "tôi tên", "tên tôi"
        ]
        needs_ticket = any(kw in lower_input for kw in ticket_keywords)

        # 2. Catalog intent (tra cứu sản phẩm, tìm kiếm, xem giá)
        catalog_keywords = [
            "xem", "tìm", "mua", "giá", "dưới", "triệu", "tỷ", "có xe nào",
            "có phòng nào", "resort", "gợi ý", "danh mục", "chi phí", "sản phẩm"
        ]
        is_warranty_faq = "bảo hành" in lower_input and not any(k in lower_input for k in ["giá", "dưới", "triệu", "xem", "mua", "resort"])
        needs_catalog = any(kw in lower_input for kw in catalog_keywords) and not is_warranty_faq

        # 3. FAQ intent (chính sách chung, không cần tool)
        faq_keywords = ["chính sách", "bảo hành", "kéo dài bao lâu", "ở đâu", "giờ mở cửa"]
        is_faq = any(kw in lower_input for kw in faq_keywords) and not needs_ticket and not needs_catalog

        return {
            "needs_catalog": needs_catalog,
            "needs_ticket": needs_ticket,
            "is_faq": is_faq
        }

    def _extract_catalog_args(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho tool search_product_catalog."""
        lower = user_input.lower()

        if any(w in lower for w in ["resort", "du lịch", "khách sạn", "nghỉ dưỡng", "phòng", "vinpearl"]):
            category = "du_lich"
        else:
            category = "xe_dien"

        max_price = 999999999999
        price_match = re.search(r'(?:dưới|<)?\s*(\d+(?:[.,]\d+)?)\s*(triệu|tr|trieu|tỷ|ty)', lower)
        if price_match:
            val = float(price_match.group(1).replace(',', '.'))
            unit = price_match.group(2)
            if unit in ["tỷ", "ty"]:
                max_price = int(val * 1_000_000_000)
            else:
                max_price = int(val * 1_000_000)

        return {"category": category, "max_price": max_price}

    def _extract_ticket_args(self, user_input: str) -> Dict[str, Any]:
        """Trích xuất tham số cho tool submit_support_ticket."""
        # Trích xuất tên khách hàng
        name_match = re.search(
            r'(?:tôi tên(?:\s+là)?|tên tôi(?:\s+là)?|khách hàng(?:\s+là)?)\s*[:]?\s*([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|\bxe\b|\bphòng\b|\bvấn đề\b|\bvà\b|$)',
            user_input,
            re.IGNORECASE
        )
        customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

        # Trích xuất priority
        lower = user_input.lower()
        if any(w in lower for w in ["gấp", "nghiêm trọng", "khẩn cấp", "high", "nguy hiểm"]):
            priority = "high"
        elif any(w in lower for w in ["thấp", "low", "không gấp"]):
            priority = "low"
        else:
            priority = "medium"

        # Trích xuất issue description
        issue_match = re.search(r'(?:xe|phòng|vấn đề|bị|lỗi|phản hồi:?)\s*(.*?)(?:\.|$)', user_input, re.IGNORECASE)
        if issue_match:
            issue_description = issue_match.group(0).strip("., ")
        else:
            issue_description = user_input.strip()

        return {
            "customer_name": customer_name,
            "issue_description": issue_description,
            "priority": priority
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []

        # TODO 3: Phân tích intent từ user_input
        intents = self._detect_intents(user_input)

        # TODO 4: Xây dựng Agent Loop
        iteration = 0
        catalog_results = None
        ticket_result = None

        while iteration < self.max_iterations:
            iteration += 1

            # 1. Xử lý FAQ (không cần gọi tool)
            if intents["is_faq"]:
                answer = (
                    "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm hoặc 200.000 km "
                    "(tùy điều kiện nào đến trước). VinFast cam kết thay thế hoặc sửa chữa miễn phí "
                    "nếu dung lượng pin xuống dưới mức 70%."
                )
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Khách hàng hỏi FAQ về chính sách bảo hành. Trả lời trực tiếp không cần tool.",
                    "action": None,
                    "observation": None,
                    "final_answer": answer
                })
                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # 2. Xử lý Single Tool: search_product_catalog
            if intents["needs_catalog"] and not intents["needs_ticket"]:
                args = self._extract_catalog_args(user_input)
                results = search_product_catalog(**args)
                catalog_results = results

                self.trace.append({
                    "iteration": iteration,
                    "thought": f"Khách hàng muốn xem sản phẩm {args['category']}. Gọi search_product_catalog.",
                    "action": "search_product_catalog",
                    "action_input": args,
                    "observation": results
                })

                if not results:
                    answer = (
                        f"Rất tiếc, hiện tại chúng tôi không tìm thấy sản phẩm nào thuộc danh mục '{args['category']}' "
                        f"có mức giá dưới {args['max_price']:,} VNĐ trong hệ thống."
                    )
                else:
                    lines = ["Dưới đây là các sản phẩm phù hợp với nhu cầu của bạn:"]
                    for p in results:
                        lines.append(f"- {p['name']} ({p.get('brand', 'Vingroup')}): {p['price_vnd']:,} VNĐ — {p.get('description', '')}")
                    answer = "\n".join(lines)

                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # 3. Xử lý Single Tool: submit_support_ticket
            if intents["needs_ticket"] and not intents["needs_catalog"]:
                args = self._extract_ticket_args(user_input)
                res = submit_support_ticket(**args)
                ticket_result = res

                self.trace.append({
                    "iteration": iteration,
                    "thought": "Khách hàng thông báo sự cố hoặc phản ánh. Gọi submit_support_ticket.",
                    "action": "submit_support_ticket",
                    "action_input": args,
                    "observation": res
                })

                answer = (
                    f"Kính chào {res['customer_name']},\n"
                    f"Yêu cầu hỗ trợ của bạn đã được ghi nhận thành công với mã ticket: {res['ticket_id']}.\n"
                    f"- Mức độ ưu tiên: {res['priority']}\n"
                    f"- Trạng thái: {res['status']}\n"
                    f"Đội ngũ kỹ thuật/CSKH của chúng tôi sẽ liên hệ xử lý trong thời gian sớm nhất."
                )

                return {
                    "answer": answer,
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # 4. Xử lý kết hợp: Cả Catalog và Ticket
            if intents["needs_catalog"] and intents["needs_ticket"]:
                # Step 1: Catalog
                cat_args = self._extract_catalog_args(user_input)
                cat_res = search_product_catalog(**cat_args)
                catalog_results = cat_res
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Yêu cầu đa nhiệm. Bước 1: Gọi search_product_catalog để tra cứu sản phẩm.",
                    "action": "search_product_catalog",
                    "action_input": cat_args,
                    "observation": cat_res
                })

                # Step 2: Ticket
                iteration += 1
                tick_args = self._extract_ticket_args(user_input)
                tick_res = submit_support_ticket(**tick_args)
                ticket_result = tick_res
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Bước 2: Gọi submit_support_ticket để ghi nhận sự cố.",
                    "action": "submit_support_ticket",
                    "action_input": tick_args,
                    "observation": tick_res
                })

                # Tổng hợp câu trả lời cuối cùng
                answer_parts = []
                if catalog_results:
                    answer_parts.append("Dưới đây là thông tin dịch vụ/sản phẩm bạn quan tâm:")
                    for p in catalog_results:
                        answer_parts.append(f"- {p['name']}: {p['price_vnd']:,} VNĐ — {p.get('description', '')}")
                else:
                    answer_parts.append("Rất tiếc, không tìm thấy sản phẩm phù hợp yêu cầu ngân sách.")

                answer_parts.append(
                    f"\nĐồng thời, yêu cầu phản hồi của bạn đã được tiếp nhận:\n"
                    f"- Mã ticket: {tick_res['ticket_id']}\n"
                    f"- Khách hàng: {tick_res['customer_name']}\n"
                    f"- Mức độ ưu tiên: {tick_res['priority']}"
                )

                return {
                    "answer": "\n".join(answer_parts),
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "completed"
                }

            # Fallback nếu không khớp intent nào
            answer = "VinAssistant xin chào bạn! Tôi có thể giúp bạn tra cứu thông tin sản phẩm xe VinFast, kỳ nghỉ Vinpearl hoặc tiếp nhận yêu cầu hỗ trợ kỹ thuật."
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": iteration,
                "status": "completed"
            }

        # Nếu vượt quá max_iterations
        return {
            "answer": "Lỗi: Vượt quá số bước tối đa cho phép.",
            "trace": self.trace,
            "iterations": iteration,
            "status": "max_iterations_reached"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:\n", result["answer"])
    print("\nTrace Log:\n", json.dumps(agent.trace, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
