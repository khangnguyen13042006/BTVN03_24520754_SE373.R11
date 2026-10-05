# -*- coding: utf-8 -*-
"""BTVN03 · Model giả lập (không phải LLM), tương thích LangChain — như demo/lib/model_gia.py.

Mục đích: chạy được không cần API key, và mỗi lần chạy ra đúng một trace để
so sánh ba mẫu agent. Hành vi được cố ý mô phỏng theo các lỗi trên slide và
DemoFlightAgent/flight_agent_failure_mode.py:

    ReAct    goal drift      nhớ ngân sách nhưng "quên" giờ bay → chọn chuyến rẻ nhất (S62)
             loop            gặp timeout thì gọi lại y hệt (S39)
             state           search trả {} → kết luận "không có chuyến" (S65)
             hallucination   bia=True: chưa đặt gì đã báo "Đã đặt VN999" (S58)
    Planner  đọc danh sách chuyến, viết TRỌN kế hoạch với chuyến cụ thể; chỗ chưa
             biết (ghế, mã vé) để biến $GHE · $MA cho bộ thực thi điền.
             Không có chuyến thoả → kế hoạch rỗng.

Đổi model thật: đặt biến môi trường SE373_MODEL (vd "openai:gpt-4.1-mini").
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

NGAN_SACH = 2_000_000


class ModelGia(BaseChatModel):
    """Model ReAct giả lập: đọc các ToolMessage trong lịch sử rồi chọn bước kế."""

    bia: bool = False
    _luot: int = 0

    @property
    def _llm_type(self) -> str:
        return "btvn03-model-gia"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ModelGia":
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._luot += 1
        return ChatResult(generations=[ChatGeneration(message=self._quyet_dinh(messages))])

    def _quyet_dinh(self, messages: list[BaseMessage]) -> AIMessage:
        obs = [(m.name, _doc(m.content)) for m in messages if m.type == "tool"]
        if not obs:
            return self._goi("search_flights", {"origin": "SGN", "dest": "DAD", "date": "2026-10-07"})
        if self.bia:
            return AIMessage(content="Đã đặt xong! Chuyến VN999, ghế 5C, giá 1,200,000đ.")

        ten, o = obs[-1]
        if ten == "get_booking" and o.get("booking_status") == "confirmed":
            return AIMessage(content=f"Đã đặt chuyến {o['flight']} khởi hành {o['depart']} ngày {o['date']}, "
                                     f"ghế {o['seat']}, giá {o['price']:,}đ, mã vé {o['booking_code']}.")
        if ten == "pay" and o.get("paid"):
            return self._goi("get_booking", {"booking_code": o["booking_code"]})
        if ten == "book_seat" and o.get("status") == "ok":
            return self._goi("pay", {"booking_code": o["booking_code"], "method": "corp_card"})
        if ten == "check_seat" and o.get("available"):
            return self._goi("book_seat", {"flight": o["flight"], "seat": o["seats"][0]})
        if ten == "check_seat" and o.get("error") == "timeout":
            return self._goi("check_seat", {"flight": o["flight"]})       # thử lại y hệt, bỏ qua hint

        chuyen = next((o["flights"] for t, o in obs if t == "search_flights" and "flights" in o), None)
        if not chuyen:       # đọc {} thành "không có chuyến" — state corruption
            return AIMessage(content="Không có chuyến bay nào từ SGN đi DAD ngày 2026-10-07.")
        # Chọn chuyến kế tiếp: rẻ nhất trong ngân sách, bỏ chuyến đã hết chỗ / bị từ chối.
        bo = {o.get("flight") for _, o in obs if o.get("status") == "denied" or o.get("available") == 0}
        ung_vien = sorted((c for c in chuyen if c["price"] <= NGAN_SACH and c["flight"] not in bo),
                          key=lambda c: c["price"])
        if ung_vien:
            return self._goi("check_seat", {"flight": ung_vien[0]["flight"]})
        return AIMessage(content="Không tìm được chuyến phù hợp với yêu cầu.")

    def _goi(self, ten: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": ten, "args": args, "id": f"call_{self._luot}",
                                                  "type": "tool_call"}])


def ke_hoach_gia(rb, chuyen: list, loai_tru: list) -> list[dict]:
    """Planner giả lập: chọn chuyến rẻ nhất thoả ràng buộc (trừ chuyến bị loại), viết trọn kế hoạch."""
    hop_le = [c for c in chuyen if not rb.vi_pham(c) and c["flight"] not in loai_tru]
    if not hop_le:
        return []
    f = min(hop_le, key=lambda c: c["price"])["flight"]
    return [
        {"tool": "check_seat", "args": {"flight": f}},
        {"tool": "book_seat", "args": {"flight": f, "seat": "$GHE"}},
        {"tool": "pay", "args": {"booking_code": "$MA", "method": "corp_card"}},
        {"tool": "get_booking", "args": {"booking_code": "$MA"}},
    ]


def _doc(content: Any) -> Any:
    try:
        return json.loads(content)
    except Exception:
        return {}
