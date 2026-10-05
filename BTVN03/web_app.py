# -*- coding: utf-8 -*-
"""Web demo BTVN#3 · chạy đúng code trong btvn03_dat_ve.py.

Chạy:  py -m streamlit run web_app.py
"""
import html
import os

import pandas as pd
import streamlit as st

from btvn03_dat_ve import MAU, RANG_BUOC, YEU_CAU, chay, danh_gia, in_bang, su_co
from lib.harness import HAN_MUC_TU_DUYET
from lib.tools_ve import KICH_BAN

NGUOI_DUYET = {
    "auto": "Tự đồng ý (chỉ duyệt bằng code)",
    "tu_choi_1": "Từ chối kế hoạch đầu tiên",
    "tu_choi_het": "Từ chối mọi kế hoạch",
}

st.set_page_config(page_title="BTVN03 · Agent đặt vé", page_icon="✈️", layout="wide")

with st.sidebar:
    st.header("Thiết lập")
    kich_ban = st.selectbox("Kịch bản", list(KICH_BAN), format_func=lambda k: f"{k} — {KICH_BAN[k]['mo_ta']}")
    # Ô chọn là <input> nên cắt chữ; dải bên dưới hiện trọn câu, kéo ngang để đọc.
    st.markdown(
        """<style>.kb-cuon{white-space:nowrap;overflow-x:auto;margin:-.5rem 0 .75rem;padding:.4rem .6rem .7rem;
        border:1px solid rgba(128,128,128,.3);border-radius:.5rem;font-size:.85rem;
        scrollbar-width:thin;scrollbar-color:rgba(128,128,128,.6) rgba(128,128,128,.12) !important}</style>"""
        f'<div class="kb-cuon" tabindex="0"><b>{kich_ban}</b> — {html.escape(KICH_BAN[kich_ban]["mo_ta"])}</div>',
        unsafe_allow_html=True)
    mau = st.radio("Mẫu thiết kế", list(MAU), format_func=MAU.get)
    bat_harness = st.toggle("Bật harness", value=True,
                            help="Tắt để thấy failure mode: goal drift, hallucination, state corruption…")
    da_duyet = st.checkbox("Người duyệt đã đồng ý thanh toán vượt hạn mức")
    kieu_duyet = st.radio("Người duyệt kế hoạch (Plan / Lai)", list(NGUOI_DUYET), format_func=NGUOI_DUYET.get,
                          disabled=(mau == "react"))
    st.divider()
    st.caption("Model")
    st.code(os.environ.get("SE373_MODEL") or "ModelGia (giả lập, không cần API key)", language=None)


def tao_nguoi_duyet(kieu: str):
    if kieu == "auto":
        return None
    lan = []
    return lambda kh: bool(lan.append(1)) or (kieu == "tu_choi_1" and len(lan) > 1)


st.title("✈️ Agent đặt vé máy bay · 3 mẫu thiết kế")
st.markdown(f"> **Yêu cầu:** {YEU_CAU}  \n> **Ràng buộc (dữ liệu):** {RANG_BUOC.mo_ta()}")

tab_chay, tab_so_sanh, tab_du_lieu = st.tabs(["▶ Chạy agent", "📊 So sánh 3 mẫu", "🧰 Dữ liệu & harness"])

with tab_chay:
    nhan = f"Chạy {MAU[mau]} · '{kich_ban}' · harness {'BẬT' if bat_harness else 'TẮT'}"
    if st.button(nhan, type="primary"):
        kq = chay(mau, kich_ban, da_duyet, bat_harness, tao_nguoi_duyet(kieu_duyet))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kiểu dừng", kq["kieu_dung"])
        c2.metric("Lượt gọi model", kq["so_model"])
        c3.metric("Lượt gọi tool", kq["so_tool"])
        c4.metric("Token ước lượng", f"{kq['token']:,}")
        if kq["ve_sai"] or kq["vuot_quyen"]:
            st.error(f"**Sự cố tiền:** {su_co(kq)}")
        if not bat_harness:
            st.info(f"**Model nói:** {kq['tra_loi'] or '(không trả lời)'}")
            (st.success if kq["dat"] else st.error)(
                f"**Thực tế (kiểm bằng code, chỉ để bạn so sánh):** {'đạt' if kq['dat'] else 'KHÔNG đạt'} yêu cầu")
        elif kq["dat"]:
            st.success(f"**Đạt tiêu chí hoàn thành (kiểm bằng code).** {kq['tra_loi']}")
        else:
            b = kq["ban_giao"]
            st.warning(f"**Dừng bất thường → bàn giao cho người.** {b['stop_reason']}")
            st.markdown(f"**Câu hỏi cho người:** {b['cau_hoi_cho_nguoi']}")
            st.markdown("**Đã thử:** " + " → ".join(f"`{x}`" for x in b["da_thu"]))
            st.json(b["trang_thai"])
        st.subheader("Trace")
        st.code("\n".join(kq["trace"]), language=None)
        st.caption("AI = model đề xuất · EXEC = bộ thực thi kế hoạch (code) · Tool = observation · "
                   "HARNESS = lớp kiểm chặn/kết luận · PLAN/REPLAN = model lập kế hoạch · NGƯỜI = người duyệt")

with tab_so_sanh:
    c1, c2 = st.columns(2)
    harness_ss = c1.toggle("Bật harness khi so sánh", value=True, key="harness_ss")
    duyet_ss = c2.checkbox("Bật duyệt thanh toán khi so sánh", key="duyet_ss")
    if st.button(f"Chạy 3 mẫu × {len(KICH_BAN)} kịch bản", type="primary"):
        rows = danh_gia(duyet_ss, harness_ss)
        df = pd.DataFrame([{"Kịch bản": r["kich_ban"], "Mẫu": MAU[r["mau"]], "Đặt đúng vé": "✅" if r["dat"] else "⛔",
                            "Kiểu dừng": r["kieu_dung"], "Sự cố tiền": su_co(r), "Lượt model": r["so_model"],
                            "Lượt tool": r["so_tool"], "Token ~": r["token"], "ms": r["ms"]} for r in rows])
        st.dataframe(df, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.caption("Lượt gọi model theo kịch bản")
        c1.bar_chart(df.pivot(index="Kịch bản", columns="Mẫu", values="Lượt model"), stack=False)
        c2.caption("Token ước lượng theo kịch bản")
        c2.bar_chart(df.pivot(index="Kịch bản", columns="Mẫu", values="Token ~"), stack=False)
        st.markdown(in_bang(rows).split("\n\n")[1])

with tab_du_lieu:
    st.subheader(f"Chuyến bay trong kịch bản '{kich_ban}'")
    st.dataframe(pd.DataFrame([{**c, "seats": len(c["seats"])} for c in KICH_BAN[kich_ban]["chuyen"]]),
                 hide_index=True)
    if KICH_BAN[kich_ban]["loi"] or KICH_BAN[kich_ban].get("model_bia"):
        st.caption(f"Lỗi cài sẵn: {KICH_BAN[kich_ban]['loi'] or 'model ReAct bịa câu trả lời'}")
    st.subheader("Các lớp harness")
    st.markdown(f"""
| Lớp | Chạy khi nào | Làm gì | Failure mode chặn được |
|---|---|---|---|
| Ràng buộc là dữ liệu | mọi nơi | `RangBuoc` — prompt, kiểm quyền, tiêu chí hoàn thành cùng đọc một nguồn | — |
| Kiểm quyền | **trước** khi chạy tool | whitelist tool · `book_seat` phải thoả ràng buộc · `pay` vé không hoàn hoặc > {HAN_MUC_TU_DUYET:,}đ cần người duyệt | Goal drift |
| Phát hiện lặp | trước khi chạy tool | cùng `(tool, args)` 3 lần trong 6 vòng → dừng | Infinite loop |
| Kết quả rỗng | sau khi chạy tool | `{{}}` là tool lỗi, không phải "không có chuyến" | State corruption |
| Ngân sách | mỗi vòng | ReAct `run_limit=10` · tối đa 15 tool · `recursion_limit=50` (luôn bật) | Chi phí không trần |
| Tiêu chí hoàn thành | sau khi agent dừng | `get_booking` → confirmed & paid & thoả ràng buộc | Tin lời model |
| Kiểm căn cứ | trước khi trả lời | mã chuyến, giờ, ngày, số tiền phải có trong kết quả tool | Hallucination |
| Duyệt kế hoạch | trước khi thực thi (Plan/Lai) | người đồng ý/từ chối; từ chối → model lập kế hoạch mới | — |
| Bàn giao | khi dừng bất thường | lý do · đã thử · trạng thái · câu hỏi cho người | — |
""")
