# BTVN03 · Agent đặt vé máy bay (SE373.R11)

Nguyễn Đỗ Hoàng Khang · MSSV 24520754

Agent đặt vé máy bay viết bằng LangChain / LangGraph, gồm tool mockup, lớp harness và 3 mẫu thiết kế: **ReAct**, **Plan-then-Execute** và **Lai**.

Yêu cầu agent nhận: *"Đặt vé SGN → DAD sáng 07/10, dưới 2 triệu."*

Bài dùng model giả lập (giống demo trên lớp) nên **không cần API key**.

## Cấu trúc

```
BTVN03/
├── btvn03_dat_ve.py   # 3 mẫu agent + chạy dòng lệnh + đánh giá
├── web_app.py         # web demo (Streamlit)
├── test_dat_ve.py     # test
└── lib/
    ├── tools_ve.py    # 5 tool mockup + 7 kịch bản
    ├── harness.py     # lớp harness dùng chung
    └── model_gia.py   # model giả lập
24520754_btvn03.pdf    # báo cáo
```

## Cách chạy

```bash
cd BTVN03
pip install -r requirements.txt

py btvn03_dat_ve.py --mau react --kich-ban chuan    # chạy 1 lần, in trace
py btvn03_dat_ve.py --danh-gia                      # so sánh 3 mẫu × 7 kịch bản
py test_dat_ve.py                                   # chạy test
py -m streamlit run web_app.py                      # web demo: http://localhost:8501
```

Tuỳ chọn: `--mau react|pte|lai`, `--kich-ban chuan|het_cho|timeout|can_duyet|vuot_ngan_sach|rong|bia`, `--tat-harness`, `--hoi-duyet`, `--duyet`.

## Kết quả chính (bật harness)

| Mẫu | Đặt được vé | TB lượt gọi model | TB token |
|---|---:|---:|---:|
| ReAct | 2/7 | 5.1 | 1.266 |
| Plan-then-Execute | 2/7 | 0.9 | 151 |
| Lai | 4/7 | 1.1 | 210 |

Chi tiết xem trong file báo cáo PDF.
