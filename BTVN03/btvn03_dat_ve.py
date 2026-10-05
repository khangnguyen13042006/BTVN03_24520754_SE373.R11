# -*- coding: utf-8 -*-
"""BTVN#3 · Agent đặt vé máy bay bằng LangChain / LangGraph · 3 mẫu thiết kế.

Cùng một yêu cầu, cùng bộ tool mockup, cùng một lớp harness (lib/harness.py).
Chỉ đổi CÁCH TỔ CHỨC SUY LUẬN:

    react   create_agent (LangChain) + HarnessMiddleware
            model chọn từng bước sau mỗi observation                       (S18–S21)
    pte     Plan-then-Execute · StateGraph (LangGraph)
            tìm chuyến → model viết TRỌN kế hoạch (1 lần) → người duyệt
            → code thực thi tuần tự; bước hỏng thì dừng, không sửa kế hoạch (S22–S23)
    lai     Lai ReAct + Plan · StateGraph
            như pte, nhưng observation đổi đáng kể thì lập lại kế hoạch    (S24)

Chạy (không cần API key, dùng model giả lập):
    py btvn03_dat_ve.py --mau react --kich-ban chuan
    py btvn03_dat_ve.py --mau react --kich-ban chuan --tat-harness   # thấy lỗi khi không có harness
    py btvn03_dat_ve.py --mau pte   --kich-ban chuan --hoi-duyet     # bạn là người duyệt kế hoạch
    py btvn03_dat_ve.py --mau pte   --kich-ban can_duyet --duyet
    py btvn03_dat_ve.py --danh-gia                                   # 3 mẫu × 7 kịch bản, harness ON/OFF
Model thật: đặt SE373_MODEL="openai:gpt-4.1-mini" (+ OPENAI_API_KEY) trong .env.
"""
import argparse
import json
import os
import sys
import time
from typing import Callable, TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, hook_config
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from lib.harness import Phien, RangBuoc
from lib.model_gia import ModelGia, ke_hoach_gia
from lib.tools_ve import KICH_BAN, TOOLS, get_booking, nap_kich_ban, tools_langchain

YEU_CAU = "Đặt vé SGN → DAD sáng 07/10, dưới 2 triệu."
RANG_BUOC = RangBuoc()          # ràng buộc là dữ liệu: prompt, harness, tiêu chí hoàn thành đều đọc từ đây
RUN_LIMIT = 10                  # ngân sách lượt gọi model của ReAct
MAX_LAP_KE_HOACH = 2            # lập lại kế hoạch tối đa 2 lần (sau khi bị từ chối / bước hỏng)
MAU = {"react": "ReAct", "pte": "Plan-then-Execute", "lai": "Lai (ReAct + Plan)"}


def tao_model(kich_ban: str):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if os.environ.get("SE373_MODEL"):
        from langchain.chat_models import init_chat_model
        return init_chat_model(os.environ["SE373_MODEL"])
    return ModelGia(bia=KICH_BAN[kich_ban].get("model_bia", False))


# ==================================================================== 1 · ReAct
class HarnessMiddleware(AgentMiddleware):
    """Cắm Phien vào vòng lặp của create_agent.

    before_model    đếm chi phí; harness đã ra lệnh dừng thì kết thúc graph
    wrap_tool_call  mọi tool call đi qua Phien.thuc_thi (kiểm quyền → chạy → ghi)
    """

    def __init__(self, phien: Phien, system_prompt: str):
        super().__init__()
        self.p, self.sp = phien, system_prompt

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if self.p.dung:
            return {"jump_to": "end"}
        self.p.dem_model(self.sp + "".join(f"{m.content}{getattr(m, 'tool_calls', '')}" for m in state["messages"]))
        return None

    def wrap_tool_call(self, request, handler):
        # Không gọi handler: Phien.thuc_thi là cửa duy nhất chạm vào tool thật,
        # để ba mẫu dùng chung đúng một lớp kiểm quyền và cùng một cách đếm.
        c = request.tool_call
        obs = self.p.thuc_thi(c["name"], c["args"])
        return ToolMessage(content=json.dumps(obs, ensure_ascii=False), tool_call_id=c["id"], name=c["name"])


def chay_react(model, phien: Phien) -> str | None:
    sp = (f"Bạn là agent đặt vé máy bay. Yêu cầu: {phien.rb.mo_ta()} "
          "Chỉ dùng dữ liệu từ tool. Quy trình: search_flights → check_seat → book_seat → pay → get_booking.")
    agent = create_agent(model=model, tools=tools_langchain(), system_prompt=sp,
                         middleware=[HarnessMiddleware(phien, sp),
                                     ModelCallLimitMiddleware(run_limit=RUN_LIMIT, exit_behavior="end")])
    kq = agent.invoke({"messages": [{"role": "user", "content": YEU_CAU}]}, {"recursion_limit": 50})
    cuoi = kq["messages"][-1]
    if not phien.dung and phien.so_model >= RUN_LIMIT and not phien.hoan_thanh()[0]:
        phien.dung = {"kieu": "Hết ngân sách", "ly_do": f"chạm run_limit={RUN_LIMIT} lượt gọi model"}
    return None if phien.dung or cuoi.type != "ai" or cuoi.tool_calls else str(cuoi.content)


# ======================================================= 2 & 3 · Plan / Lai
class Buoc(BaseModel):
    tool: str
    args: dict[str, str]


class KeHoach(BaseModel):
    buoc: list[Buoc]       # danh sách rỗng = không có chuyến nào thoả ràng buộc


PROMPT_PLANNER = """Lập TRỌN kế hoạch đặt vé máy bay, trả danh sách bước {{tool, args}}.
Yêu cầu: {yeu_cau}
Tool: check_seat(flight) · book_seat(flight,seat) · pay(booking_code,method) · get_booking(booking_code)
Ghế và mã booking chưa biết: ghi "$GHE" và "$MA", bộ thực thi sẽ điền.
Chuyến tìm được: {chuyen}
Chuyến bị loại (đã hỏng hoặc bị người duyệt từ chối): {loai_tru} · Lý do lập lại: {ly_do}
Nếu không chuyến nào thoả MỌI ràng buộc, trả kế hoạch rỗng."""


def lap_ke_hoach(model, phien: Phien, loai_tru: list, ly_do: str) -> list[dict]:
    chuyen = list(phien.chuyen.values())
    prompt = PROMPT_PLANNER.format(yeu_cau=phien.rb.mo_ta(), chuyen=json.dumps(chuyen, ensure_ascii=False),
                                   loai_tru=loai_tru or "không", ly_do=ly_do)
    phien.dem_model(prompt)
    if isinstance(model, ModelGia):
        return ke_hoach_gia(phien.rb, chuyen, loai_tru)
    return [b.model_dump() for b in model.with_structured_output(KeHoach).invoke(prompt).buoc]


def chuyen_cua(ke_hoach: list):
    return next((b["args"]["flight"] for b in ke_hoach if "flight" in b["args"]), None)


def mo_ta_ke_hoach(ke_hoach: list) -> str:
    return " → ".join(f"{b['tool']}({', '.join(b['args'].values())})" for b in ke_hoach) or \
        "(rỗng: không chuyến nào thoả ràng buộc)"


class TrangThai(TypedDict, total=False):
    ke_hoach: list
    buoc: int
    ghe: str | None
    loai_tru: list
    so_lan_lap: int
    loi: dict | None


def tao_graph(model, phien: Phien, lai: bool, nguoi_duyet: Callable[[list], bool] | None):
    def n_tim_chuyen(s: TrangThai):
        # Ngữ cảnh cho planner: một lần search chỉ-đọc, tham số lấy thẳng từ RangBuoc.
        rb = phien.rb
        phien.thuc_thi("search_flights", {"origin": rb.origin, "dest": rb.dest, "date": rb.date}, ai="EXEC")
        return {}

    def n_lap_ke_hoach(s: TrangThai):
        lan = s.get("so_lan_lap", -1) + 1
        ly_do = "lần đầu" if lan == 0 else f"{s['loi']}"
        kh = lap_ke_hoach(model, phien, s.get("loai_tru", []), ly_do)
        phien.ghi("PLAN" if lan == 0 else "REPLAN", mo_ta_ke_hoach(kh))
        if (la := [b["tool"] for b in kh if b["tool"] not in TOOLS]):        # duyệt bằng code: chỉ tool hợp lệ
            phien.dung = {"kieu": "Kế hoạch lỗi thời", "ly_do": f"kế hoạch có tool không hợp lệ {la}"}
        cap_nhat = {"ke_hoach": kh, "buoc": 0, "ghe": None, "loi": None, "so_lan_lap": lan}
        if kh and nguoi_duyet and not phien.dung:                            # người duyệt TRƯỚC khi chạy
            dong_y = nguoi_duyet(kh)
            phien.ghi("NGƯỜI", "duyệt kế hoạch" if dong_y else "TỪ CHỐI kế hoạch → yêu cầu lập lại")
            if not dong_y:
                cap_nhat["loi"] = {"error": "rejected", "flight": chuyen_cua(kh)}
                cap_nhat["loai_tru"] = s.get("loai_tru", []) + [chuyen_cua(kh)]
        return cap_nhat

    def sau_lap_ke_hoach(s: TrangThai):
        if phien.dung or not s["ke_hoach"]:
            return END
        if s.get("loi"):                                   # bị từ chối → viết kế hoạch mới (cả pte lẫn lai)
            return "lap_ke_hoach" if s["so_lan_lap"] < MAX_LAP_KE_HOACH else END
        return "thuc_thi"

    def n_thuc_thi(s: TrangThai):
        b = s["ke_hoach"][s["buoc"]]
        bien = {"$GHE": s.get("ghe"), "$MA": phien.booking}
        obs = phien.thuc_thi(b["tool"], {k: bien.get(v, v) for k, v in b["args"].items()}, ai="EXEC")
        cap_nhat = {"buoc": s["buoc"] + 1, "loi": None}
        if b["tool"] == "check_seat" and obs.get("seats"):
            cap_nhat["ghe"] = obs["seats"][0]
        if obs.get("status") != "ok" or obs.get("available") == 0:     # observation đổi đáng kể
            cap_nhat["loi"] = obs
            cap_nhat["loai_tru"] = s.get("loai_tru", []) + [chuyen_cua(s["ke_hoach"])]
        return cap_nhat

    def sau_thuc_thi(s: TrangThai):
        if phien.dung:
            return END
        if s.get("loi"):
            return "lap_ke_hoach" if lai and s["so_lan_lap"] < MAX_LAP_KE_HOACH else END
        return END if s["buoc"] >= len(s["ke_hoach"]) else "thuc_thi"

    g = StateGraph(TrangThai)
    g.add_node("tim_chuyen", n_tim_chuyen)
    g.add_node("lap_ke_hoach", n_lap_ke_hoach)
    g.add_node("thuc_thi", n_thuc_thi)
    g.add_edge(START, "tim_chuyen")
    g.add_conditional_edges("tim_chuyen", lambda s: END if phien.dung else "lap_ke_hoach")
    g.add_conditional_edges("lap_ke_hoach", sau_lap_ke_hoach)
    g.add_conditional_edges("thuc_thi", sau_thuc_thi)
    return g.compile()


def chay_ke_hoach(model, phien: Phien, lai: bool, nguoi_duyet=None) -> str | None:
    s = tao_graph(model, phien, lai, nguoi_duyet).invoke({"loai_tru": []}, {"recursion_limit": 50})
    if not phien.dung:
        loi = s.get("loi") or {}
        if loi.get("error") == "rejected":
            phien.dung = {"kieu": "Người duyệt từ chối", "ly_do": f"từ chối {s['so_lan_lap'] + 1} kế hoạch"}
        elif loi:
            phien.dung = {"kieu": "Bế tắc" if lai else "Kế hoạch lỗi thời",
                          "ly_do": f"bước {s['buoc']} thất bại: {json.dumps(loi, ensure_ascii=False)}"}
        elif "ke_hoach" in s and not s["ke_hoach"]:
            phien.dung = {"kieu": "Không đạt tiêu chí", "ly_do": "planner: không chuyến nào thoả ràng buộc"}
    if not phien.bat and not phien.booking:        # tắt harness: chỉ còn lời "model" planner để tin
        return None if phien.chuyen else "Không có chuyến bay nào từ SGN đi DAD ngày 2026-10-07."
    if phien.dung or not phien.booking:
        return None
    # Câu trả lời dựng từ get_booking bằng code → mẫu Plan không có kênh để bịa số liệu.
    b = get_booking(phien.booking)
    return (f"Đã đặt chuyến {b['flight']} khởi hành {b['depart']} ngày {b['date']}, ghế {b['seat']}, "
            f"giá {b['price']:,}đ, mã vé {b['booking_code']}.")


# ================================================================= điều phối
def chay(mau: str, kich_ban: str, da_duyet: bool = False, bat_harness: bool = True,
         nguoi_duyet: Callable[[list], bool] | None = None, model=None) -> dict:
    """nguoi_duyet(ke_hoach) -> True/False: người duyệt kế hoạch (chỉ pte/lai). None = tự đồng ý."""
    nap_kich_ban(kich_ban)
    phien = Phien(RANG_BUOC, da_duyet=da_duyet, bat_harness=bat_harness)
    phien.trace.append(f'[V0] Human   "{YEU_CAU}"')
    model = model or tao_model(kich_ban)
    t0 = time.perf_counter()
    if mau == "react":
        tra_loi = chay_react(model, phien)
    else:
        tra_loi = chay_ke_hoach(model, phien, lai=(mau == "lai"), nguoi_duyet=nguoi_duyet)
    kq = phien.ket_luan(tra_loi)
    return dict(mau=mau, kich_ban=kich_ban, ms=round((time.perf_counter() - t0) * 1000, 1), **kq)


def danh_gia(da_duyet: bool = False, bat_harness: bool = True) -> list[dict]:
    return [chay(m, kb, da_duyet, bat_harness) for kb in KICH_BAN for m in MAU]


def su_co(r: dict) -> str:
    return " · ".join(x for x, co in (("💸 vé sai", r["ve_sai"]), ("⚠ trả tiền chưa duyệt", r["vuot_quyen"])) if co) or "—"


def in_bang(rows: list[dict]) -> str:
    dong = ["| Kịch bản | Mẫu | Đặt đúng vé | Kiểu dừng | Sự cố tiền | Lượt model | Lượt tool | Token ~ |",
            "|---|---|---|---|---|---:|---:|---:|"]
    dong += [f"| {r['kich_ban']} | {MAU[r['mau']]} | {'✅' if r['dat'] else '⛔'} | {r['kieu_dung']} "
             f"| {su_co(r)} | {r['so_model']} | {r['so_tool']} | {r['token']:,} |" for r in rows]
    dong += ["", "| Mẫu | Đặt đúng vé | Trả tiền vé sai | Trả tiền chưa duyệt | TB lượt model | TB lượt tool | TB token ~ |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m, ten in MAU.items():
        rs = [r for r in rows if r["mau"] == m]
        tb = lambda k: sum(r[k] for r in rs) / len(rs)          # noqa: E731
        dong.append(f"| {ten} | {sum(r['dat'] for r in rs)}/{len(rs)} | {sum(r['ve_sai'] for r in rs)} "
                    f"| {sum(r['vuot_quyen'] for r in rs)} | {tb('so_model'):.1f} | {tb('so_tool'):.1f} "
                    f"| {tb('token'):,.0f} |")
    return "\n".join(dong)


def _hoi_nguoi(ke_hoach: list) -> bool:
    print("\n--- kế hoạch (model viết trong MỘT lần gọi) ---")
    for i, b in enumerate(ke_hoach, 1):
        print(f"  Bước {i}: {b['tool']}({b['args']})")
    tl = ""
    while tl not in ("y", "n"):
        tl = input("Người duyệt - đồng ý kế hoạch này? (y/n): ").strip().lower()
    return tl == "y"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="BTVN#3 · agent đặt vé máy bay")
    p.add_argument("--mau", default="react", choices=list(MAU))
    p.add_argument("--kich-ban", default="chuan", choices=list(KICH_BAN))
    p.add_argument("--duyet", action="store_true", help="người duyệt đã đồng ý thanh toán vượt hạn mức")
    p.add_argument("--hoi-duyet", action="store_true", help="hỏi bạn duyệt từng kế hoạch (pte/lai)")
    p.add_argument("--tat-harness", action="store_true", help="tắt harness để thấy failure mode")
    p.add_argument("--danh-gia", action="store_true", help="chạy 3 mẫu × mọi kịch bản, harness ON rồi OFF")
    a = p.parse_args()

    if a.danh_gia:
        print("## Harness BẬT\n\n" + in_bang(danh_gia(a.duyet, True)))
        print("\n## Harness TẮT\n\n" + in_bang(danh_gia(a.duyet, False)))
        return
    kq = chay(a.mau, a.kich_ban, a.duyet, not a.tat_harness, _hoi_nguoi if a.hoi_duyet else None)
    print("=" * 78)
    print(f"BTVN03 · {MAU[a.mau]} · harness {'TẮT' if a.tat_harness else 'BẬT'} · "
          f"kịch bản {a.kich_ban}: {KICH_BAN[a.kich_ban]['mo_ta']}")
    print("=" * 78)
    print("\n".join(kq["trace"]))
    print("-" * 78)
    print(f"Kiểu dừng : {kq['kieu_dung']} · {kq['so_model']} lượt model · {kq['so_tool']} lượt tool · ~{kq['token']:,} token")
    if kq["ve_sai"] or kq["vuot_quyen"]:
        print("CẢNH BÁO  :", su_co(kq))
    if kq["ban_giao"]:
        print("Bàn giao  :", json.dumps(kq["ban_giao"], ensure_ascii=False, indent=2))
    else:
        print("Trả lời   :", kq["tra_loi"])


if __name__ == "__main__":
    main()
