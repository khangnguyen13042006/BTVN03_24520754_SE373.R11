# -*- coding: utf-8 -*-
"""BTVN03 · Bộ tool mockup đặt vé máy bay.

Dữ liệu tĩnh, không gọi mạng, mỗi kịch bản cho ra đúng một kết quả để so sánh
ba mẫu agent công bằng.

    search_flights(origin, dest, date)   -> danh sách chuyến
    check_seat(flight)                   -> ghế trống + giá thực
    book_seat(flight, seat)              -> giữ chỗ, trả mã booking (status held)
    pay(booking_code, method)            -> thanh toán, booking chuyển confirmed
    get_booking(booking_code)            -> đọc lại booking (kiểm chứng chéo)

Mọi kết quả đều có "status" rõ ràng: ok · invalid_param · error (kèm hint),
theo slide 56 và 66 — trừ kịch bản "rong", cố ý trả {} để tái hiện state corruption.
"""
import copy
import re

NGAY = "2026-10-07"


def _cb(flight, depart, price, refundable=True, seats=("12A", "12B")):
    return dict(flight=flight, depart=depart, price=price, refundable=refundable, seats=list(seats))


# Mỗi kịch bản là một "thế giới" khác nhau cho cùng câu yêu cầu.
KICH_BAN = {
    "chuan": dict(
        mo_ta="Bình thường: có chuyến sáng dưới 2 triệu còn ghế.",
        chuyen=[_cb("QH118", "15:40", 1_640_000), _cb("VN122", "08:10", 1_850_000),
                _cb("VJ604", "10:20", 2_150_000)],
        loi={}),
    "het_cho": dict(
        mo_ta="Chuyến sáng rẻ nhất (VN120) đã hết ghế, phải đổi sang chuyến khác.",
        chuyen=[_cb("QH118", "15:40", 1_640_000), _cb("VN120", "07:00", 1_700_000, seats=()),
                _cb("VJ610", "09:30", 1_790_000, seats=("7D",))],
        loi={}),
    "timeout": dict(
        mo_ta="Dịch vụ check_seat của VN122 luôn timeout; VJ610 vẫn bình thường.",
        chuyen=[_cb("QH118", "15:40", 1_640_000), _cb("VN122", "08:10", 1_850_000),
                _cb("VJ610", "09:30", 1_890_000)],
        loi={"check_seat:VN122": "timeout"}),
    "can_duyet": dict(
        mo_ta="Chuyến sáng duy nhất là vé KHÔNG hoàn, giá vượt hạn mức tự duyệt.",
        chuyen=[_cb("QH118", "15:40", 1_640_000), _cb("VJ612", "07:00", 1_950_000, refundable=False)],
        loi={}),
    "vuot_ngan_sach": dict(
        mo_ta="Mọi chuyến sáng đều trên 2 triệu; chuyến rẻ thì bay chiều.",
        chuyen=[_cb("QH118", "15:40", 1_640_000), _cb("VN122", "08:10", 2_310_000),
                _cb("VJ604", "10:20", 2_080_000)],
        loi={}),
    # Hai kịch bản failure mode (DemoFlightAgent/flight_agent_failure_mode.py)
    "rong": dict(
        mo_ta="State corruption: dịch vụ tìm chuyến timeout và trả {} — không phải 'không có chuyến'.",
        chuyen=[_cb("QH118", "15:40", 1_640_000), _cb("VN122", "08:10", 1_850_000)],
        loi={"search_flights": "empty"}),
    "bia": dict(
        mo_ta="Hallucination: model ReAct chưa đặt gì đã báo 'Đã đặt VN999'.",
        chuyen=[_cb("QH118", "15:40", 1_640_000), _cb("VN122", "08:10", 1_850_000)],
        loi={}, model_bia=True),
}

# ponytail: trạng thái toàn cục, một phiên chạy tại một thời điểm; cần chạy song song thì đưa vào object.
_KHO = {}


def nap_kich_ban(ten: str) -> None:
    """Nạp lại dữ liệu sạch cho một kịch bản (gọi trước mỗi lần chạy agent)."""
    kb = KICH_BAN[ten]
    _KHO.clear()
    _KHO.update(chuyen={c["flight"]: c for c in copy.deepcopy(kb["chuyen"])},
                loi=dict(kb["loi"]), booking={})


def _sai_ngay(date: str):
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(date)):
        return {"status": "invalid_param", "param": "date",
                "hint": "Dùng YYYY-MM-DD, ví dụ 2026-10-07"}
    return None


# ---------------------------------------------------------------- tools
def search_flights(origin: str, dest: str, date: str) -> dict:
    """Tìm chuyến bay theo điểm đi, điểm đến và ngày (YYYY-MM-DD). Trả danh sách chuyến kèm giờ bay và giá."""
    if (loi := _sai_ngay(date)):
        return loi
    if _KHO["loi"].get("search_flights") == "empty":
        return {}                                        # tool thiết kế tệ: lỗi mà im lặng
    if (origin.upper(), dest.upper(), date) != ("SGN", "DAD", NGAY):
        return {"status": "ok", "flights": []}          # thật sự không có chuyến
    return {"status": "ok", "flights": [
        {k: c[k] for k in ("flight", "depart", "price", "refundable")} for c in _KHO["chuyen"].values()]}


def check_seat(flight: str) -> dict:
    """Kiểm tra ghế trống và giá thực của một chuyến bay (mã chuyến lấy từ search_flights)."""
    if _KHO["loi"].get(f"check_seat:{flight}") == "timeout":
        return {"status": "error", "error": "timeout", "flight": flight,
                "hint": "Dịch vụ kiểm ghế lỗi, thử lại sau hoặc chọn chuyến khác"}
    c = _KHO["chuyen"].get(flight)
    if not c:
        return {"status": "error", "error": "flight_not_found", "flight": flight,
                "hint": "Gọi search_flights để lấy mã chuyến hợp lệ"}
    return {"status": "ok", "flight": flight, "price": c["price"],
            "available": len(c["seats"]), "seats": c["seats"][:3]}


def book_seat(flight: str, seat: str) -> dict:
    """Giữ một ghế trên chuyến bay. Trả mã booking ở trạng thái held (chưa thanh toán)."""
    c = _KHO["chuyen"].get(flight)
    if not c or seat not in c["seats"]:
        return {"status": "error", "error": "seat_unavailable", "flight": flight, "seat": seat,
                "hint": "Gọi check_seat để lấy ghế còn trống"}
    c["seats"].remove(seat)
    ma = f"PNR{len(_KHO['booking']) + 101}"
    _KHO["booking"][ma] = dict(booking_code=ma, flight=flight, seat=seat, date=NGAY,
                               depart=c["depart"], price=c["price"], refundable=c["refundable"],
                               booking_status="held", paid=False)
    return {"status": "ok", "booking_code": ma, "flight": flight, "seat": seat, "booking_status": "held"}


def pay(booking_code: str, method: str = "corp_card") -> dict:
    """Thanh toán một booking đang held. Sau khi trả tiền, booking chuyển sang confirmed."""
    b = _KHO["booking"].get(booking_code)
    if not b:
        return {"status": "error", "error": "booking_not_found", "booking_code": booking_code}
    b.update(paid=True, booking_status="confirmed")
    return {"status": "ok", "booking_code": booking_code, "paid": True, "method": method}


def get_booking(booking_code: str) -> dict:
    """Đọc lại thông tin booking: chuyến, ghế, ngày, giờ, giá, trạng thái và đã trả tiền chưa."""
    b = _KHO["booking"].get(booking_code)
    if not b:
        return {"status": "error", "error": "booking_not_found", "booking_code": booking_code}
    return {"status": "ok", **b}


TOOLS = {f.__name__: f for f in (search_flights, check_seat, book_seat, pay, get_booking)}


def tools_langchain():
    """Bọc các hàm Python thành tool LangChain cho create_agent."""
    from langchain_core.tools import tool
    return [tool(f) for f in TOOLS.values()]
