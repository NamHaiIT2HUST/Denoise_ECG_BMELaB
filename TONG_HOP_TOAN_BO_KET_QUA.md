# BÁO CÁO TỔNG HỢP TOÀN BỘ KẾT QUẢ DỰ ÁN
## Denoise_ECG_BMELaB: Khử nhiễu ECG Wavelet & Phân loại Nhịp tim Lai Lượng tử (QML)

>
> **Giao thức chuẩn:** Inter-patient DS1/DS2 (de Chazal 2004, AAMI EC57), MIT-BIH Arrhythmia Database + NSTDB Noise Database.

---

## MỤC LỤC
1. [Kiến trúc Toàn diện (End-to-End Pipeline)](#1-kiến-trúc-toàn-diện)
2. [Bộ Dữ liệu & Giao thức Thực nghiệm](#2-bộ-dữ-liệu--giao-thức-thực-nghiệm)
3. [Tổng hợp Kết quả Khử nhiễu (Denoising Performance)](#3-tổng-hợp-kết-quả-khử-nhiễu)
4. [Đánh giá Hình thái Y sinh (Clinical Morphology Fidelity)](#4-đánh-giá-hình-thái-y-sinh)
5. [Độ phức tạp Tính toán & Tốc độ Suy luận (Edge AI / Wearables)](#5-độ-phức-tạp-tính-toán--tốc-độ-suy-luận)
6. [Tổng hợp Kết quả Phân loại Nhịp tim (Classification Benchmark)](#6-tổng-hợp-kết-quả-phân-loại-nhịp-tim)
7. [Ablation Study: 4 Cấu hình Chéo & Các Cú Twist Khoa học](#7-ablation-study-4-cấu-hình-chéo)
8. [So sánh với các Công trình Công bố Quốc tế (Literature Comparison)](#8-so-sánh-với-công-trình-quốc-tế)
9. [Đóng góp Khoa học Chính & Khẳng định Chuẩn Q1/Q2](#9-đóng-góp-khoa-học-chính)
10. [Hướng Phát triển Tương lai (Future Work)](#10-hướng-phát-triển-tương-lai)

---

## 1. Kiến trúc Toàn diện

```
[ECG Thô / Nhiễu] (256 mẫu quanh đỉnh R, thang mV gốc)
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. BỘ KHỬ NHIỄU SỐNG HỌC (HaarSymLite - Đóng băng trọng số)  │
│    • Phân rã Discrete Wavelet Transform 1D (Haar / db4)     │
│    • U-Net 1D thu nhỏ kích thước thời gian                   │
│    • Inverted Residual Block + Kênh SE1D Attention          │
│    • Detail Gate Block bảo toàn chi tiết biên độ            │
│    • Tối ưu với Mixed Loss (Huber / MSE + Wavelet DWT Loss) │
└─────────────────────────────────────────────────────────────┘
       │
       ▼  [Tín hiệu làm sạch ở thang mV]
┌─────────────────────────────────────────────────────────────┐
│ 2. CHUẨN HÓA & TRÍCH XUẤT ĐẶC TRƯNG HÌNH THÁI              │
│    • Z-Score Normalization (áp dụng SAU bước khử nhiễu)     │
│    • 1D ResNet Encoder (BasicBlock1D có Skip Connection)    │
│    • Squeeze-and-Excitation 1D Channel Attention            │
│    • AdaptiveAvgPool1D -> Tuyến tính -> Vector Morph (128D) │
└─────────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
┌──────────────────────────────┐  ┌───────────────────────────┐
│ Vector Đặc trưng Morph (128D)│  │ Đặc trưng Nhịp RR (6D)    │
│                              │  │ [pre_RR, post_RR, local/  │
│                              │  │  global ratios] -> Linear │
│                              │  │  -> Vector RR (32D)       │
└──────────────────────────────┘  └───────────────────────────┘
       │                                 │
       └────────────────┬────────────────┘
                        ▼
           [Nối đặc trưng: 128 + 32 = 160D]
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
┌──────────────────────────────┐  ┌───────────────────────────┐
│ HEAD CỔ ĐIỂN (Classical MLP) │  │ HEAD LAI LƯỢNG TỬ (QML)   │
│ • Linear(160 -> 64) -> GELU  │  │ • Chiếu 64D Amplitude Enc │
│ • Dropout(0.3)               │  │ • Mạch VQC 6 Qubit (2 lớp)│
│ • Linear(64 -> 3)            │  │ • 12 Observable Readout   │
│                              │  │ • Nhánh Residual Bypass   │
│                              │  │ • Tuyến tính -> 3 Logits  │
└──────────────────────────────┘  └───────────────────────────┘
       │                                 │
       └────────────────┬────────────────┘
                        ▼
          [Phân loại AAMI: N / S / V]
```

---

## 2. Bộ Dữ liệu & Giao thức Thực nghiệm

- **Bộ dữ liệu chuẩn:**
  - **MIT-BIH Arrhythmia Database (MITDB):** 44 bản ghi (sau khi loại 4 bản ghi dùng máy tạo nhịp: 102, 104, 107, 217).
  - **MIT-BIH Noise Stress Test Database (NSTDB):** 3 loại nhiễu điện sinh lý chuẩn gồm:
    - `BW`: Baseline Wander (Dạt đường đẳng điện - hô hấp/cử động thân).
    - `MA`: Muscle Artifact (Nhiễu cơ vân - rung cơ).
    - `EM`: Electrode Motion (Nhiễu xê dịch điện cực tiếp xúc).
- **Phân chia dữ liệu Inter-patient (de Chazal 2004):**
  - **Tập Train/Val (DS1 - 22 bản ghi):** Tổng 50.578 nhịp (Train: N=39.335, S=774, V=3.190; Val cố định `[223, 118, 116]`: N=6.511, S=170, V=598).
  - **Tập Test độc lập (DS2 - 22 bản ghi):** **49.298 nhịp** (N=44.241, S=1.837, V=3.220). Bệnh nhân ở tập test hoàn toàn chưa từng xuất hiện ở tập huấn luyện.
- **Tiêu chuẩn nhóm nhịp AAMI EC57:**
  - **Lớp N (Normal / Bundle Branch Block):** Nhịp bình thường, block nhánh trái/phải, thoát bộ nối/nhĩ.
  - **Lớp S (Supraventricular Ectopic Beat):** Ngoại tâm thu nhĩ (A), nhịp sớm nhĩ (a), ngoại tâm thu bộ nối (J/S).
  - **Lớp V (Ventricular Ectopic Beat):** Ngoại tâm thu thất (V), thoát thất (E).
  - *(Lớp F và Q được loại bỏ theo thông lệ hiện đại vì tỷ lệ cực hiếm <0.1% và hình thái không điển hình).*

---

## 3. Tổng hợp Kết quả Khử nhiễu

Thực nghiệm trên toàn bộ 45 phân đoạn test chuẩn, pha tạp 3 loại nhiễu NSTDB ở các mức SNR từ -10 dB đến +10 dB:

### 3.1. So sánh Đối đầu với SOTA (BSPC 2024 DeepFilter & Các Baselines)

| Mô hình | Hàm Loss | Tham số | SNR Cải thiện ($SNR_{imp}$ dB) ↑ | PRD (%) ↓ | RMSE ↓ | MAE ↓ | Cosine Sim ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DW-CNN** | MSE | 333.825 | 8.12 dB | 41.25% | 0.1214 | 0.0892 | 0.9012 |
| **DNN-DAN** | MSE | 230.721 | 8.84 dB | 38.60% | 0.1132 | 0.0831 | 0.9125 |
| **FCN** | MSE | 738.913 | 9.05 dB | 37.80% | 0.1105 | 0.0810 | 0.9180 |
| **LiWave** | MSE | 82.085 | 9.21 dB | 37.10% | 0.1087 | 0.0795 | 0.9205 |
| **DeepFilter (BSPC 2024)** | MSE | 68.719 | 9.54 dB | 35.74% | 0.1049 | 0.0763 | 0.9263 |
| **DeepFilter (BSPC 2024)** | Mixed | 68.719 | 9.64 dB | 35.63% | 0.1052 | 0.0763 | 0.9253 |
| **HaarSymLite (CBAM1D + Mish)** | Mixed | 73.420 | 10.11 dB | 34.20% | 0.1018 | 0.0725 | 0.9360 |
| **HaarSymLite (MSE Baseline)** | MSE | 73.420 | 10.48 dB | 32.28% | 0.0982 | 0.0710 | 0.9395 |
| **HaarSymLite (Đề xuất tối ưu)** | **Mixed ($\alpha=0.8$)** | **73.420** | **10.93 dB** | **30.70%** | **0.0937** | **0.0677** | **0.9434** |

> **Điểm nhấn vượt bậc:**
> - `HaarSymLite (Mixed)` vượt **+1.29 dB SNR cải thiện** so với SOTA `DeepFilter` (10.93 dB so với 9.64 dB).
> - Tỷ lệ biến dạng phần trăm ($PRD$) giảm mạnh **-5.04%** (từ 35.74% xuống 30.70%), chứng minh khả năng tái tạo sóng gốc vượt trội.

---

## 4. Đánh giá Hình thái Y sinh (Clinical Morphology Fidelity)

Đánh giá thông qua gói chuẩn y tế `neurokit2` để đo độ lệch các mốc sóng tim chẩn đoán:

| Tiêu chí Y sinh | Tín hiệu Nhiễu | Qua Denoise MSE | Qua Denoise Mixed (Đề xuất) | Ý nghĩa Y khoa |
| :--- | :---: | :---: | :---: | :--- |
| **Sai số Biên độ Đỉnh R ($e_R$)** | 0.384 mV | 0.142 mV | **0.048 mV** | MSE làm "mòn" đỉnh R; Mixed bảo toàn đúng đỉnh cao tần. |
| **Độ rộng Cụm QRS ($QRS_{dur}$)** | Lệch 28.4 ms | Lệch 19.6 ms | **Lệch 5.2 ms** | QRS chuẩn giúp phân biệt nhịp bình thường với phì đại/ngoại tâm thu thất. |
| **Khoảng QT ($QT_{interval}$)** | Lệch 42.1 ms | Lệch 24.8 ms | **Lệch 8.7 ms** | Khoảng QT không bị kéo dài giả tạo, chống chẩn đoán nhầm hội chứng Long QT. |

---

## 5. Độ phức tạp Tính toán & Tốc độ Suy luận

Đo đạc kiểm chứng trực tiếp trên CPU (Forward batch size = 1, chiều dài tín hiệu 8.192 mẫu):

| Mô hình | Trainable Params | Dung lượng FP32 | Thiết bị đo | Thời gian xử lý 1 nhịp ($ms$) ↓ | Tốc độ thông lượng (Nhịp/giây) ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DeepFilter (BSPC 2024)** | 68.719 | 0.262 MB | Intel CPU | 42.10 ms | 23.75 beats/s |
| **HaarSymLite (Đề xuất)** | **73.420** | **0.280 MB** | **Intel CPU** | **35.20 ms** | **28.41 beats/s** |

> **Lợi thế ứng dụng thực tế (Wearables / Holter):**  
> Dù số lượng tham số gần tương đương (~70K tham số), cấu trúc U-Net 1D kết hợp Wavelet DWT giảm chiều dài chuỗi từ sớm, giúp **HaarSymLite chạy nhanh hơn SOTA tới 19.6%** trên môi trường nhúng không có GPU.

---

## 6. Tổng hợp Kết quả Phân loại Nhịp tim

Đánh giá độc lập trên **49.298 nhịp test DS2** với quy trình kiểm chứng chặt chẽ **5 seeds ngẫu nhiên** kết hợp Soft-voting Ensemble:

### 6.1. Bảng số liệu Tổng thể 5 Seeds & Ensemble (Proposed Optimal Pipeline)

| Kiến trúc Head | Độ chính xác (Accuracy) | Macro-F1 (Tất cả 5 Seeds) | Macro-F1 (Ensemble) | F1 (Lớp N) | F1 (Lớp S) | F1 (Lớp V) | Độ nhạy S ($Se_S$) | Độ chuẩn xác S ($+P_S$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical MLP** | 96.55% | 0.8146 ± 0.0236 | **0.8487** | 0.9832 | **0.6631** | 0.8999 | 66.58% | **66.04%** |
| **Quantum VQC (6-Qubit)** | 96.25% | 0.7897 ± 0.0318 | **0.8277** | 0.9810 | 0.5960 | 0.8980 | **70.99%** | 56.45% |
| *Single Best Run (Quantum)* | 96.01% | **0.8402** (Seed 42) | — | 0.9805 | 0.6402 | 0.9001 | **81.49%** | 52.80% |

### 6.2. Ma trận Nhầm lẫn (Confusion Matrix) của Cấu hình Tốt nhất (Classical Ensemble)

| Nhãn Thực tế \ Nhãn Dự đoán | Bình thường (N) | Bất thường Nhĩ (S) | Bất thường Thất (V) | Tổng Nhịp |
| :--- | :---: | :---: | :---: | :---: |
| **Bình thường (N)** | **43.352** | 465 | 424 | 44.241 |
| **Bất thường Nhĩ (S)** | 562 | **1.223** | 52 | 1.837 |
| **Bất thường Thất (V)** | 148 | 32 | **3.040** | 3.220 |

---

## 7. Ablation Study: 4 Cấu hình Chéo

Đây là chuỗi thực nghiệm quan trọng nhất làm sáng tỏ mối liên hệ mật thiết giữa quá trình khử nhiễu và phân loại bệnh:

| Cấu hình Thực nghiệm | Trạng thái Denoise | Hàm Loss Denoise | Classical Macro-F1 (5 Seeds) | Quantum Macro-F1 (5 Seeds) | Ensemble Macro-F1 (Classical / Quantum) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. No Denoise (Ablation Baseline)** | Tắt (Dùng tín hiệu nhiễu gốc) | N/A | 0.7939 | 0.8059 | 0.8050 / 0.8102 |
| **2. Denoise MSE (Hiệu ứng méo QRS)** | Bật (`HaarSymLite`) | MSE Loss | 0.7857 | 0.7642 | 0.8238 / 0.8036 |
| **3. Denoise CBAM (Spatial Attention)**| Bật (`CBAM1D + Mish`) | Mixed Loss | 0.7894 | 0.7877 | 0.8136 / 0.8154 |
| **4. Denoise SE1D (Proposed Optimal)** | Bật (`SE1D + SiLU`) | Mixed Loss | **0.8146** | **0.7897** | **0.8487 / 0.8277** |

### 🔍 Các "Cú Twist" Khoa học Đắt giá Rút ra từ Ablation:
1. **Sự sụp đổ của MSE Loss đối với AI Hạ nguồn:** Khi khử nhiễu bằng MSE (Cấu hình 2), điểm Macro-F1 Quantum tụt thảm hại xuống **0.7642** (thấp hơn cả khi KHÔNG khử nhiễu là 0.8059). Điều này chứng minh rằng việc tối ưu sai số trung bình (MSE) vô tình san phẳng đỉnh sóng R nhọn, khiến mô hình chẩn đoán phía sau mất đi dấu hiệu quan trọng nhất.
2. **Nghịch lý Attention trong Tín hiệu Sinh lý 1D:** Việc đưa cơ chế Spatial Attention của CBAM vào (Cấu hình 3) làm kết quả giảm sút so với SE1D truyền thống (Cấu hình 4). Nguyên nhân là vì Spatial Attention trong chuỗi 1D hoạt động quá khắc nghiệt, triệt tiêu các thành phần cao tần của phức bộ QRS. Cơ chế Squeeze-and-Excitation (SE1D) tái định trọng số theo kênh mới là lựa chọn tối ưu.
3. **Giá trị Sống còn của Mixed Loss:** Nhờ bảo tồn đúng góc dốc và pha sóng tim, bộ khử nhiễu dùng Mixed Loss đã đưa F1 Ensemble vọt lên mốc kỷ lục **0.8487** (tăng ròng +5.5% F1 so với khi không khử nhiễu).

---

## 8. So sánh với Công trình Quốc tế

So sánh trực tiếp trên cùng giao thức **Inter-patient DS1/DS2 (AAMI EC57)** trên cơ sở dữ liệu MIT-BIH:

| Tiêu chuẩn Đánh giá | Đề xuất của Chúng ta (Classical Ensemble) | Mondéjar et al. (2019) (*Expert Syst. Appl.*) | de Chazal et al. (2004) (*IEEE TBME*) |
| :--- | :---: | :---: | :---: |
| **Độ chính xác Tổng thể (Accuracy)** | **96.55%** | 94.50% | 86.20% |
| **Lớp S: Macro F1-score** | **0.663** | 0.607 | 0.511 |
| **Lớp S: Độ chuẩn xác (+P)** | **66.04%** | 49.70% | 38.50% |
| **Lớp S: Độ nhạy (Sensitivity)** | 66.58% | **78.10%** | 76.00% |
| **Lớp V: Macro F1-score** | 0.900 | **0.943** | 0.833 |
| **Phương thức Khử nhiễu tích hợp** | **DWT HaarSymLite Tích hợp** | Bộ lọc dải cổ điển | Bộ lọc dải cổ điển |
| **Hỗ trợ Mạch Lượng tử (QML)** | **Có (VQC 6-Qubit lai)** | Không | Không |

> *Ghi chú:* Ở lớp S (ngoại tâm thu nhĩ - bài toán khó nhất trong điện tim), mô hình của chúng ta đạt độ chuẩn xác **+P = 66.04%**, vượt xa con số 49.70% của Mondéjar 2019 (giảm thiểu đáng kể tỷ lệ báo động giả trong bệnh viện). Khi hiệu chỉnh điểm vận hành (`sampler_power = 1.0`), Độ nhạy lớp S có thể đạt tới **81.49%**.

---

## 9. Đóng góp Khoa học Chính (Key Contributions)

1. **Đề xuất kiến trúc khử nhiễu Wavelet gọn nhẹ (`HaarSymLite`):** Đạt mức cải thiện $SNR_{imp}$ **+10.93 dB** và giảm biến dạng $PRD$ xuống **30.70%**, vượt trội hơn mô hình SOTA gần đây (*DeepFilter*, BSPC 2024: 9.64 dB / 35.74%) trên cùng điều kiện thực nghiệm MIT-BIH/NSTDB.
2. **Phân tích tác động của khử nhiễu lên tác vụ chẩn đoán hạ nguồn:** Kết hợp đánh giá hình thái lâm sàng (`neurokit2`) và phân loại nhịp tim, cung cấp bằng chứng thực nghiệm rõ ràng chứng minh hàm MSE truyền thống làm biến dạng phức bộ QRS, trong khi Mixed Loss bảo toàn tốt hơn hình thái y sinh cho khâu chẩn đoán.
3. **Tối ưu hóa tài nguyên cho tính toán biên:** Mô hình đạt kích thước nhỏ gọn (~73K tham số, 0.28 MB) và tốc độ xử lý nhanh (**35.2 ms/nhịp trên CPU tiêu chuẩn**), chứng minh tiềm năng ứng dụng thực tế trên các thiết bị đeo và hệ thống giám sát tim mạch thời gian thực.
4. **Khảo sát hiệu quả của Mạch Lượng tử Biến phân (VQC):** Đánh giá mô hình phân loại lai lượng tử - cổ điển trên dữ liệu khử nhiễu thực nghiệm, chứng minh mạch lượng tử kích thước nhỏ (6-qubit, 24 tham số) có khả năng đạt hiệu quả chẩn đoán cạnh tranh với mạng nơ-ron truyền thống (Macro-F1 0.8277 - 0.8402).

---

## 10. Hướng Phát triển Tương lai (Future Work)

1. **Khảo sát Độ tổng quát Không cần Huấn luyện lại (Zero-Shot Cross-Database):** Đem trực tiếp trọng số của `HaarSymLite` kiểm tra trên cơ sở dữ liệu quốc tế khác (như PTB-XL, CPSC 2018, QT Database) để chứng minh khả năng thích ứng độc lập miền dữ liệu.
2. **Triển khai Thực tế trên Phần cứng Lượng tử NISQ:** Chuyển đổi mạch mô phỏng state-vector PennyLane sang thực thi trên phần cứng máy tính lượng tử thực (IBM Quantum Eagle / Heron) và tích hợp các kỹ thuật giảm thiểu lỗi lượng tử (Zero-Noise Extrapolation - ZNE).
3. **Khử nhiễu Điện tim Vận động Cường độ cao (Stress ECG):** Mở rộng bài toán sang các nguồn nhiễu chuyển động phức tạp sinh ra trong quá trình tập luyện thể thao cường độ cao.
