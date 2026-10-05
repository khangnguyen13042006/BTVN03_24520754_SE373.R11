# -*- coding: utf-8 -*-
"""BTVN03 · Lớp harness viết tay cho agent đặt vé — dùng chung cho cả 3 mẫu.

Bốn lớp đề bài yêu cầu (slide 69), cộng hai lớp dừng từ Demo 2:

    RangBuoc            ràng buộc là DỮ LIỆU, không nằm rải rác trong prompt     (S63)
    Phien.hoan_thanh    tiêu chí hoàn thành kiểm bằng CODE qua get_booking       (S43, S44)
    Phien.kiem_truoc    kiểm quyền TRƯỚC khi thực thi tool                       (S35, S41, S60)
    Phien.ban_giao      dừng bất thường thì bàn giao, không im lặng              (S48)
    LoopDetector        phát hiện lặp (tool, args)                               (S39, S46)
    kiem_can_cu         đối chiếu câu trả lời với kết quả tool, chống bịa        (S59)
    kết quả rỗng {}     coi là tool lỗi, không phải "không có gì"                (S64–S66)

Không phụ thuộc LangChain: mẫu nào gọi tool cũng phải đi qua Phien.thuc_thi().
Phien(bat_harness=False) tắt mọi lớp trên (chỉ còn whitelist + ngân sách cứng)
để so sánh ON/OFF như DemoFlightAgent/flight_agent_failure_mode.py.
"""
from collections import deque
from dataclasses import asdict, dataclass
import json
import re

from lib.tools_ve import TOOLS, get_booking

HAN_MUC_TU_DUYET = 1_900_000      # trên mức này, hoặc vé không hoàn → phải có người duyệt


# ================================================== 1 · RÀNG BUỘC LÀ DỮ LIỆU
@dataclass(frozen=True)
class RangBuoc:
    origin: str = "SGN"
    dest: str = "DAD"
    date: str = "2026-10-07"
    depart_before: str = "12:00"
    max_price: int = 2_000_000

    def vi_pham(self, chuyen: dict) -> list:
        """Danh sách ràng buộc mà một chuyến/booking vi phạm; rỗng = thoả hết."""
        loi = []
        if chuyen.get("depart", "99:99") >= self.depart_before:
            loi.append(f"giờ bay {chuyen.get('depart')} không trước {self.depart_before}")
        if chuyen.get("price", float("inf")) > self.max_price:
            loi.append(f"giá {chuyen.get('price'):,} vượt {self.max_price:,}")
        if "date" in chuyen and chuyen["date"] != self.date:
            loi.append(f"ngày {chuyen['date']} khác {self.date}")
        return loi

    def mo_ta(self) -> str:
        return (f"Đặt vé {self.origin} → {self.dest} ngày {self.date}, khởi hành trước "
                f"{self.depart_before}, giá không quá {self.max_price:,}đ.")


# ======================================================== 2 · PHÁT HIỆN LẶP
class LoopDetector:
    """Giữ nguyên từ demo/lib/harness.py: so (tool, args) trong cửa sổ gần nhất."""

    def __init__(self, window=6, repeat_k=3):
        self.recent = deque(maxlen=window)
        self.k = repeat_k

    def check(self, tool: str, args: dict):
        fp = (tool, repr(sorted(args.items())))
        n = self.recent.count(fp) + 1
        self.recent.append(fp)
        if n >= self.k:
            return f"LOOP · '{tool}' gọi {n} lần với cùng tham số trong {self.recent.maxlen} vòng gần nhất"
        return None


def ban_giao(ly_do: str, da_thu: list, trang_thai: dict, cau_hoi: str) -> dict:
    return {"stop_reason": ly_do, "da_thu": da_thu, "trang_thai": trang_thai, "cau_hoi_cho_nguoi": cau_hoi}


# ===================================================== 3 · KIỂM TRA CĂN CỨ
_DU_KIEN = r"[A-Z]{2,3}\d{3}|\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}|\d[\d.,]*\d"


def kiem_can_cu(cau_tra_loi: str, ket_qua_tool: list) -> list:
    """Trả danh sách dữ kiện (mã chuyến, mã vé, ngày, giờ, số tiền) KHÔNG có trong kết quả tool."""
    nguon = " ".join(ket_qua_tool)
    nguon_so = re.sub(r"(?<=\d)[.,](?=\d)", "", nguon)       # 1,850,000 ≡ 1850000
    return [d for d in dict.fromkeys(re.findall(_DU_KIEN, cau_tra_loi))
            if d not in nguon and re.sub(r"[.,]", "", d) not in nguon_so]


# ============================================================ PHIÊN HARNESS
class Phien:
    """Một lần chạy agent. Ghi trace, đếm chi phí, chặn tool, xét dừng, bàn giao."""

    def __init__(self, rang_buoc: RangBuoc = RangBuoc(), da_duyet: bool = False, max_tool: int = 15,
                 bat_harness: bool = True):
        """bat_harness=False: chỉ còn whitelist + ngân sách cứng (như framework), để so sánh ON/OFF."""
        self.rb, self.da_duyet, self.max_tool, self.bat = rang_buoc, da_duyet, max_tool, bat_harness
        self.det = LoopDetector()
        self.trace, self.da_thu, self.quan_sat = [], [], []
        self.so_model = self.so_tool = self.token = self.vong = 0
        self.chuyen = {}          # flight -> thông tin thấy từ search_flights
        self.booking = None       # mã booking đã giữ (tác dụng phụ!)
        self.dung = None          # {"kieu", "ly_do"} khi harness chủ động dừng

    # ------------------------------------------------------------ ghi chép
    def ghi(self, ai: str, noi_dung) -> None:
        if not isinstance(noi_dung, str):
            noi_dung = json.dumps(noi_dung, ensure_ascii=False)
        self.trace.append(f"[V{self.vong}] {ai:<7} {noi_dung}")

    def dem_model(self, ngu_canh: str) -> None:
        """Mỗi lần gọi model: +1 lượt, token ước lượng = số ký tự ngữ cảnh / 4."""
        self.so_model += 1
        self.token += len(ngu_canh) // 4

    # --------------------------------------------- 0 · KIỂM QUYỀN TRƯỚC KHI CHẠY
    def kiem_truoc(self, ten: str, args: dict):
        """None = cho chạy. Ngược lại trả observation từ chối (model đọc được, có hint)."""
        if ten not in TOOLS:
            return {"status": "denied", "error": "unknown_tool", "hint": f"Chỉ được gọi: {', '.join(TOOLS)}"}
        if self.so_tool >= self.max_tool:
            self.dung = {"kieu": "Hết ngân sách", "ly_do": f"đã gọi {self.so_tool}/{self.max_tool} tool"}
            return {"status": "stopped", "error": "budget"}
        if not self.bat:
            return None
        if (canh_bao := self.det.check(ten, args)):
            self.dung = {"kieu": "Phát hiện lặp", "ly_do": canh_bao}
            return {"status": "stopped", "error": "loop"}
        if ten == "book_seat":
            c = self.chuyen.get(args.get("flight"))
            if not c:
                return {"status": "denied", "error": "unknown_flight", "flight": args.get("flight"),
                        "hint": "Chỉ đặt chuyến đã thấy trong kết quả search_flights"}
            if (vp := self.rb.vi_pham(c)):
                return {"status": "denied", "error": "constraint_violation", "flight": c["flight"],
                        "vi_pham": vp, "hint": "Chọn chuyến khác thoả mọi ràng buộc"}
        if ten == "pay" and not self.da_duyet:
            b = get_booking(args.get("booking_code", ""))
            if b["status"] == "ok" and (not b["refundable"] or b["price"] > HAN_MUC_TU_DUYET):
                ly_do = (f"{b['flight']} · {b['price']:,}đ · "
                         f"{'không hoàn' if not b['refundable'] else 'hoàn được'} · hạn mức tự duyệt {HAN_MUC_TU_DUYET:,}đ")
                self.dung = {"kieu": "Cần con người", "ly_do": "thanh toán vượt thẩm quyền: " + ly_do}
                return {"status": "needs_approval", "booking_code": b["booking_code"], "ly_do": ly_do}
        return None

    # ------------------------------------------------- cửa duy nhất gọi tool
    def thuc_thi(self, ten: str, args: dict, ai: str = "AI") -> dict:
        """ai = ai đề xuất bước này: "AI" (model ReAct) hay "EXEC" (bộ thực thi kế hoạch)."""
        self.vong += 1
        self.ghi(ai, f"{ten}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
        self.da_thu.append(f"{ten}({', '.join(map(str, args.values()))})")
        obs = self.kiem_truoc(ten, args)
        if obs is None:
            try:
                obs = TOOLS[ten](**args)
            except TypeError as e:                     # sai tên/thiếu tham số
                obs = {"status": "invalid_param", "hint": str(e)}
            self.so_tool += 1
            self.ghi_ket_qua(ten, obs)
            if self.bat and not obs.get("status"):          # kết quả rỗng = tool lỗi, KHÔNG phải "không có gì"
                self.ghi("Tool", obs)
                self.dung = {"kieu": "Tool lỗi", "ly_do": f"{ten} trả {json.dumps(obs)} — lỗi dịch vụ, "
                                                          "không phải 'không có chuyến'"}
                obs = {"status": "stopped", "error": "empty_result"}
        self.ghi("Tool" if obs.get("status") not in ("denied", "stopped", "needs_approval") else "HARNESS", obs)
        return obs

    def ghi_ket_qua(self, ten: str, obs: dict) -> None:
        self.quan_sat.append(json.dumps(obs, ensure_ascii=False))
        if ten == "search_flights" and obs.get("status") == "ok":
            self.chuyen.update({c["flight"]: c for c in obs["flights"]})
        if ten == "book_seat" and obs.get("status") == "ok":
            self.booking = obs["booking_code"]

    # ------------------------------------------- 1 · TIÊU CHÍ HOÀN THÀNH (code)
    def hoan_thanh(self):
        """Không tin model nói 'đã đặt'. Đọc lại booking và kiểm từng ràng buộc."""
        if not self.booking:
            return False, "chưa có booking nào được giữ"
        b = get_booking(self.booking)
        if b.get("booking_status") != "confirmed" or not b.get("paid"):
            return False, f"booking {self.booking} đang {b.get('booking_status')}, paid={b.get('paid')}"
        if (vp := self.rb.vi_pham(b)):
            return False, "booking vi phạm ràng buộc: " + "; ".join(vp)
        return True, f"{b['flight']} {b['depart']} · ghế {b['seat']} · {b['price']:,}đ · confirmed & paid"

    # --------------------------------------------------------- 4 · BÀN GIAO
    def ban_giao(self) -> dict:
        kieu = self.dung["kieu"]
        cau_hoi = {
            "Cần con người": "Duyệt thanh toán vé này (Có/Không)? Booking đang được giữ chỗ.",
            "Phát hiện lặp": "Dịch vụ không phản hồi. Chờ thử lại sau, hay cho phép chọn chuyến khác?",
            "Hết ngân sách": "Agent đã dùng hết lượt. Nới ngân sách lượt gọi hay xử lý thủ công?",
            "Kế hoạch lỗi thời": "Một bước của kế hoạch thất bại. Duyệt kế hoạch mới bỏ chuyến này, hay xử lý thủ công?",
            "Tool lỗi": "Dịch vụ tìm chuyến không trả dữ liệu. Thử lại sau hay đặt thủ công?",
            "Bịa đặt thông tin": "Câu trả lời nhắc tới vé mà không tool nào tạo ra. Tìm và đặt lại từ đầu?",
            "Người duyệt từ chối": "Mọi kế hoạch đều bị từ chối. Người duyệt muốn chọn chuyến nào?",
        }.get(kieu, "Không có chuyến thoả mọi ràng buộc. Nới giá, đổi giờ hay đổi ngày?")
        return ban_giao(
            ly_do=f"{kieu.upper()} · {self.dung['ly_do']}",
            da_thu=self.da_thu,
            trang_thai={"so_lan_goi_tool": self.so_tool, "booking_dang_giu": self.booking,
                        "chuyen_da_thay": len(self.chuyen)},
            cau_hoi=cau_hoi)

    # ------------------------------------------------------------ kết luận
    def ket_luan(self, cau_tra_loi):
        """Chạy sau khi agent dừng: kiểm hoàn thành + căn cứ, hoặc bàn giao."""
        dat, chi_tiet = self.hoan_thanh()
        if cau_tra_loi:
            self.ghi("AI", cau_tra_loi)
        bia = kiem_can_cu(cau_tra_loi or "", self.quan_sat)
        if not self.bat:      # không harness: tin lời model; `dat` chỉ để người chấm thấy sự thật
            self.ghi("THỰC TẾ", ("đạt · " if dat else "KHÔNG đạt · ") + chi_tiet)
            return dict(dat=dat, kieu_dung="Tin model (tắt harness)", tra_loi=cau_tra_loi, ban_giao=None,
                        **self.so_lieu())
        if not self.dung and not dat:
            self.dung = ({"kieu": "Bịa đặt thông tin", "ly_do": f"{bia} không có trong kết quả tool nào; {chi_tiet}"}
                         if bia else {"kieu": "Không đạt tiêu chí", "ly_do": chi_tiet})
        if self.dung:
            bg = self.ban_giao()
            self.ghi("HARNESS", "DỪNG BẤT THƯỜNG · " + bg["stop_reason"])
            return dict(dat=False, kieu_dung=self.dung["kieu"], tra_loi=None, ban_giao=bg, **self.so_lieu())
        if bia:          # vé đã đặt thật, nhưng câu trả lời có số không nguồn → không cho trả ra
            self.ghi("HARNESS", f"câu trả lời có dữ kiện không nguồn: {bia} → thay bằng dữ liệu get_booking")
            cau_tra_loi = "Đã đặt vé: " + chi_tiet
        self.ghi("HARNESS", "ĐẠT TIÊU CHÍ · " + chi_tiet)
        return dict(dat=True, kieu_dung="Đạt mục tiêu", tra_loi=cau_tra_loi, ban_giao=None, **self.so_lieu())

    def so_lieu(self) -> dict:
        b = get_booking(self.booking) if self.booking else {}
        ve_sai = bool(b.get("paid")) and not self.hoan_thanh()[0]      # đã trả tiền cho vé sai yêu cầu
        vuot_quyen = (bool(b.get("paid")) and not self.da_duyet        # đã trả tiền mà chưa ai duyệt
                      and (not b["refundable"] or b["price"] > HAN_MUC_TU_DUYET))
        return dict(so_model=self.so_model, so_tool=self.so_tool, token=self.token,
                    ve_sai=ve_sai, vuot_quyen=vuot_quyen,
                    trace=self.trace, rang_buoc=asdict(self.rb))
