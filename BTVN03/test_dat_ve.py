# -*- coding: utf-8 -*-
"""Kiểm tra nhanh: py test_dat_ve.py  (hoặc pytest). Sai một ô là harness/mẫu đã đổi hành vi."""
from btvn03_dat_ve import chay
from lib.harness import RangBuoc, kiem_can_cu

DAT = "Đạt mục tiêu"
# kịch bản -> kiểu dừng mong đợi của (react, pte, lai), harness BẬT
MONG_DOI = {
    "chuan": (DAT, DAT, DAT),
    "het_cho": (DAT, "Kế hoạch lỗi thời", DAT),
    "timeout": ("Phát hiện lặp", "Kế hoạch lỗi thời", DAT),
    "can_duyet": ("Cần con người",) * 3,
    "vuot_ngan_sach": ("Không đạt tiêu chí",) * 3,
    "rong": ("Tool lỗi",) * 3,
    "bia": ("Bịa đặt thông tin", DAT, DAT),
}


def test_harness_bat():
    for kb, kieu3 in MONG_DOI.items():
        for mau, kieu in zip(("react", "pte", "lai"), kieu3):
            kq = chay(mau, kb)
            assert kq["kieu_dung"] == kieu, (kb, mau, kq["kieu_dung"])
            assert kq["dat"] == (kieu == DAT)
            assert (kq["ban_giao"] is None) == kq["dat"]          # dừng bất thường luôn có bàn giao
            assert not kq["ve_sai"] and not kq["vuot_quyen"]       # harness bật: không mất tiền oan


def test_harness_tat_lo_failure_mode():
    assert chay("react", "chuan", bat_harness=False)["ve_sai"]                 # goal drift: trả tiền QH118 chiều
    assert chay("pte", "can_duyet", bat_harness=False)["vuot_quyen"]           # trả vé không hoàn chưa duyệt
    assert "Không có chuyến" in chay("react", "rong", bat_harness=False)["tra_loi"]   # state corruption
    assert "VN999" in chay("react", "bia", bat_harness=False)["tra_loi"]       # hallucination lọt ra ngoài


def test_nguoi_duyet():
    assert all(chay(m, "can_duyet", da_duyet=True)["dat"] for m in ("react", "pte", "lai"))
    lan = []
    tu_choi_lan_dau = lambda kh: bool(lan.append(1)) or len(lan) > 1        # noqa: E731
    kq = chay("pte", "het_cho", nguoi_duyet=tu_choi_lan_dau)               # từ chối VN120 → kế hoạch VJ610
    assert kq["dat"] and any("TỪ CHỐI" in d for d in kq["trace"])


def test_rang_buoc_va_can_cu():
    rb = RangBuoc()
    assert rb.vi_pham({"depart": "08:10", "price": 1_850_000}) == []
    assert len(rb.vi_pham({"depart": "15:40", "price": 2_100_000})) == 2
    nguon = ['{"flight": "VN122", "price": 1850000, "depart": "08:10"}']
    assert kiem_can_cu("Đã đặt VN122 lúc 08:10 giá 1,850,000đ", nguon) == []
    assert kiem_can_cu("Done! Booked VN999, seat 5C, for 1,200,000 VND.", nguon) == ["VN999", "1,200,000"]


if __name__ == "__main__":
    test_harness_bat()
    test_harness_tat_lo_failure_mode()
    test_nguoi_duyet()
    test_rang_buoc_va_can_cu()
    print("OK · 4 test đạt")
