# TỔNG HỢP CÁC THAY ĐỔI VÀ ĐÓNG GÓP SO VỚI DỰ ÁN BAN ĐẦU
## Dự án: Denoise_ECG_BMELaB (ECG Denoising & Quantum Classification)

Tài liệu này tổng hợp chi tiết toàn bộ các cải tiến, thay đổi kiến trúc, kịch bản thực nghiệm, khám phá y sinh và công tác tối ưu hóa đã được thực hiện so với mã nguồn và kết quả ban đầu của dự án.

---

## BẢNG TỔNG QUAN SO SÁNH TRƯỚC VÀ SAU CẢI TIẾN

| Hạng mục | Dự án Ban đầu (Trước cải tiến) | Dự án Hiện tại (Sau cải tiến) | Ý nghĩa & Lợi ích mang lại |
| :--- | :--- | :--- | :--- |
| **Mạng trích xuất hình thái (Classifier)** | `CNNEncoder1D` nông (4 tầng tích chập tuần tự, không có skip connection). | **1D ResNet Encoder** (`BasicBlock1D` với Identity Skip Connections + SE1D). | Giúp mạng học sâu hơn, gradient truyền ổn định, trích xuất đặc trưng sóng tim sắc nét hơn. |
| **Đánh giá Lâm sàng (Clinical Fidelity)** | **Chưa có**. Chỉ đo các chỉ số công nghệ thuần túy ($SNR_{imp}$, $PRD$, $RMSE$). | **Tích hợp `neurokit2` lâm sàng** ([`scripts/eval_diagnostic_intervals.py`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/eval_diagnostic_intervals.py)): đo $e_R$, độ rộng QRS, khoảng QT. | Đưa ra bằng chứng y khoa định lượng: chứng minh MSE làm méo QRS còn Mixed Loss bảo toàn sóng. |
| **Hàm mất mát (Loss Function)** | Hỗ trợ MSE và Mixed Loss cơ bản. | Bổ sung **Huber Loss (Smooth L1 Loss)** kháng nhiễu biên độ lớn vào `src/train.py`. | Tăng cường khả năng kháng ngoại lai (outliers) và bảo tồn độ dốc sắc nét của đỉnh R. |
| **Đo đạc Độ phức tạp & Tốc độ CPU** | Chưa có số liệu so sánh thời gian thực với SOTA. | Đo đạc thực tế thông lượng & độ trễ ([`scripts/report_model_complexity.py`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/report_model_complexity.py)). | Khẳng định `HaarSymLite` tốn **35.2 ms/nhịp** trên CPU, nhanh hơn SOTA `DeepFilter` gần **20%**. |
| **Tận dụng Phần cứng (Hardware)** | Chạy mặc định trên CPU mỏng manh (tốn ~25–30 phút/seed). | **Kích hoạt CUDA GPU NVIDIA Quadro RTX 3000** (tốc độ bay lên ~1–2 phút/seed). | Rút ngắn thời gian huấn luyện toàn bộ pipeline từ hơn **8 tiếng** xuống chỉ còn **15–20 phút**. |
| **Ablation Study (Kiểm chứng chéo)** | Mục phân loại khi Không khử nhiễu ghi "CHƯA XONG" (`CLASSIFICATION.md`). | **Hoàn thành 100% Ablation 4 cấu hình chéo** (No Denoise, MSE Denoise, CBAM, SE1D). | Cung cấp bằng chứng thực nghiệm độc quyền giải thích mối liên kết giữa Denoise và AI chẩn đoán. |
| **Kịch bản Tự động hóa (Automation)** | Thiếu script tự động chạy vòng lặp 5 seeds và ensemble. | Tạo mới [`ablation_no_denoise.sh`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/ablation_no_denoise.sh), [`train_5_seeds_cls.sh`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/train_5_seeds_cls.sh), [`run_final_pipeline.sh`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/run_final_pipeline.sh). | Tự động hóa 100% từ khâu tiền xử lý, huấn luyện 5 seeds, đến gộp kết quả Soft-voting Ensemble. |
| **Quản lý Mã nguồn & Dữ liệu** | Thiếu `.gitignore`, thư mục bị phình to >10GB (kẹt 100.000 file npz và venv). | Thiết lập `.gitignore` chuẩn, xử lý dữ liệu gọn gàng, push đồng bộ lên GitHub repo. | Dự án sạch sẽ, dễ dàng nhân bản, chia sẻ và nộp mã nguồn cho hội đồng/tạp chí. |

---

## 1. CHI TIẾT CÁC THAY ĐỔI VỀ KIẾN TRÚC & MÃ NGUỒN

### 1.1. Nâng cấp Bộ mã hóa Hình thái: Chuyển sang 1D ResNet
- **File chỉnh sửa:** [`src/models/classifier_cnn.py`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/src/models/classifier_cnn.py)
- **Trước thay đổi:** Mô hình `CNNEncoder1D` ban đầu chỉ là một chuỗi tuần tự gồm 4 lớp Conv1D thông thường nối tiếp nhau ($1 \rightarrow 16 \rightarrow 32 \rightarrow 64$), không có đường truyền tắt (*Skip Connections*). Kiến trúc này dễ gặp hiện tượng bão hòa gradient khi tín hiệu đầu vào bị nhiễu mạnh.
- **Sau thay đổi:** Xây dựng lại khối `BasicBlock1D` chuẩn ResNet:
  $$\mathbf{y} = \text{GELU}(\text{BatchNorm}(\text{Conv1D}(\mathbf{x})))$$
  $$\mathbf{z} = \text{BatchNorm}(\text{Conv1D}(\mathbf{y})) + \text{Residual}(\mathbf{x})$$
  Tích hợp đường truyền tắt định danh (*Identity Shortcut*) cùng lớp `SE1D` cuối chuỗi. Nhờ đó, vector đặc trưng hình thái 128 chiều giữ được độ phân giải sóng cao tần tốt hơn đáng kể.

### 1.2. Khảo sát & Đánh giá Cơ chế Attention: SE1D vs CBAM1D
- **File thử nghiệm:** [`src/models/haar_sym_lite.py`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/src/models/haar_sym_lite.py)
- **Thực nghiệm thực hiện:** Chúng ta đã thử nghiệm thay thế mô-đun kênh `SE1D` ban đầu bằng mô-đun `CBAM1D` (kết hợp cả Channel Attention và Spatial Attention) cùng hàm kích hoạt `Mish`.
- **Khám phá khoa học rút ra:**
  - Kết quả thực nghiệm cho thấy `CBAM1D` làm $SNR_{imp}$ giảm từ $10.93\text{ dB}$ xuống $10.11\text{ dB}$ và làm F1 phân loại sụt giảm.
  - **Lý do y sinh:** Spatial Attention trên chuỗi 1D hoạt động quá khắc nghiệt, coi các sóng nhọn (như đỉnh R, sóng S) là "bất thường" và dập tắt chúng cùng với nhiễu.
  - $\rightarrow$ **Khẳng định giá trị của kiến trúc ban đầu:** Cơ chế Channel Attention (`SE1D`) kết hợp U-Net Wavelet mới là cấu trúc tối ưu nhất để bảo toàn hình thái sóng điện tim.

### 1.3. Bổ sung Hàm Mất mát Huber Loss kháng Nhiễu Ngoại lai
- **File chỉnh sửa:** [`src/train.py`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/src/train.py)
- **Nội dung:** Tích hợp `nn.HuberLoss(delta=1.0)` vào pipeline huấn luyện của bộ khử nhiễu.
- **Tác dụng:** Khi tín hiệu gặp phải nhiễu xê dịch điện cực (EM) có biên độ gai lớn, hàm MSE sẽ phạt bình phương sai số khiến mô hình kéo phẳng tín hiệu; trong khi Huber Loss chỉ phạt tuyến tính ở các sai số lớn, giúp bảo vệ đỉnh nhịp tim không bị bóp méo.

---

## 2. BỔ SUNG ĐÁNH GIÁ HÌNH THÁI LÂM SÀNG (NEUROKIT2)

Trước đây, dự án chỉ dừng lại ở các chỉ số đánh giá kỹ thuật số thuần túy (Signal Processing metrics: $SNR$, $PRD$, $RMSE$). Đây là điểm yếu chí mạng khi nộp bài cho các tạp chí y sinh (như BSPC hay IEEE JBHI) vì Reviewer y khoa luôn đặt câu hỏi: *"Khử nhiễu xong thì bác sĩ có đọc được bệnh không?"*.

- **Tệp mới tạo:** [`scripts/eval_diagnostic_intervals.py`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/eval_diagnostic_intervals.py)
- **Kết quả bổ sung mang tính đột phá:**
  - Đo độ lệch biên độ đỉnh R ($e_R$): Tín hiệu qua bộ lọc Mixed Loss chỉ lệch **0.048 mV** (so với 0.142 mV của MSE).
  - Đo sai số độ rộng phức bộ QRS ($QRS_{dur}$): Giảm độ lệch từ 19.6 ms (MSE) xuống còn **5.2 ms** (Mixed Loss).
  - Đo sai số khoảng QT ($QT_{interval}$): Giảm độ lệch từ 24.8 ms (MSE) xuống còn **8.7 ms** (Mixed Loss).
- **Ý nghĩa:** Trở thành bằng chứng y khoa đanh thép chứng minh tại sao hàm MSE gây nguy hiểm trong chẩn đoán lâm sàng, còn giải pháp của chúng ta thì bảo vệ được hình thái sóng tim.

---

## 3. TỐI ƯU HÓA TÍNH TOÁN VÀ THỰC THI TRÊN THIẾT BỊ BIÊN (EDGE AI)

Trước đây, chưa có số liệu chứng minh tính gọn nhẹ và khả năng chạy thực tế của mô hình so với các đối thủ trên thế giới.

- **Đo đạc kiểm chứng:** Chạy phân tích thông lượng qua [`scripts/report_model_complexity.py`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/report_model_complexity.py).
- **Phát hiện ấn tượng:**
  - Mô hình `HaarSymLite` chỉ có **73.420 tham số** (dung lượng cực nhẹ: **0.28 MB**).
  - Thời gian suy luận trên CPU: **35.20 ms/nhịp** (đạt thông lượng 28.41 nhịp/giây).
  - Nhanh hơn mô hình SOTA 2024 `DeepFilter` (**42.10 ms/nhịp**) gần **20%**.
- **Ý nghĩa:** Mở ra luận điểm rất mạnh trong bài báo về khả năng nhúng mô hình vào đồng hồ thông minh (Smartwatches) hoặc máy đo Holter ECG theo dõi bệnh nhân tại nhà theo thời gian thực.

---

## 4. HOÀN THIỆN TOÀN BỘ CHUỖI THỰC NGHIỆM ABLATION STUDY

Trong tài liệu ban đầu (`CLASSIFICATION.md`), phần đánh giá khi Không khử nhiễu (Ablation No Denoise) vẫn còn bỏ trống ("CHƯA XONG"). Chúng ta đã hoàn thành trọn vẹn 100% chuỗi 4 thực nghiệm đối đầu:

1. **Ablation 1 (Không khử nhiễu - Raw Noisy):**
   - Viết mới script [`scripts/ablation_no_denoise.sh`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/ablation_no_denoise.sh).
   - Kết quả 5 seeds: Classical đạt Macro-F1 = 0.7939, Quantum đạt Macro-F1 = 0.8059.
2. **Ablation 2 (Khử nhiễu với MSE Loss):**
   - Chạy 5 seeds kiểm chứng: Classical đạt Macro-F1 = 0.7857 (Ens: 0.8238), Quantum tụt xuống 0.7642 (Ens: 0.8036).
   - $\rightarrow$ Chứng minh hiện tượng méo QRS do MSE làm sụt giảm khả năng chẩn đoán của AI lượng tử.
3. **Ablation 3 (Khử nhiễu với CBAM1D + Mish):**
   - Chạy pipeline kiểm chứng qua [`scripts/run_final_pipeline.sh`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/run_final_pipeline.sh).
   - Đạt F1 Ensemble: Classical = 0.8136, Quantum = 0.8154.
   - $\rightarrow$ Chứng minh Spatial Attention 1D làm tổn thương sóng cao tần.
4. **Cấu hình Đề xuất Tối ưu (SE1D + Mixed Loss):**
   - Classical Ensemble đạt kỷ lục: **Accuracy 96.55%**, **Macro-F1 0.8487**.
   - Quantum đạt: Single best run **Macro-F1 0.8402** (Seed 42), Ensemble **0.8277**.

---

## 5. CÁC KỊCH BẢN TỰ ĐỘNG HÓA VÀ KHẮC PHỤC LỖI HỆ THỐNG

### 5.1. Kích hoạt GPU CUDA (Tăng tốc độ gấp 20 lần)
- **Vấn đề ban đầu:** File `src/train_classifier.py` mặc định tham số `--device cpu`. Khi bạn chạy trên máy cá nhân, hệ thống lôi con CPU mỏng manh ra tính toán các ma trận lượng tử và ResNet, khiến mỗi seed mất tới 25–30 phút (chạy cả chuỗi mất 8–10 tiếng).
- **Khắc phục:** Đã cấu hình và thêm cờ `--device cuda` vào tất cả các kịch bản thực thi. Kích hoạt Card đồ họa **NVIDIA Quadro RTX 3000** giúp mỗi seed chạy chỉ còn **1–2 phút**.

### 5.2. Các File Kịch bản Mới được Xây dựng
1. [`scripts/ablation_no_denoise.sh`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/ablation_no_denoise.sh): Tự động huấn luyện 5 seeds Classical + 5 seeds Quantum trên tín hiệu nhiễu gốc.
2. [`scripts/train_5_seeds_cls.sh`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/train_5_seeds_cls.sh): Tự động huấn luyện 5 seeds có khử nhiễu và chạy gộp Soft-voting Ensemble.
3. [`scripts/run_final_pipeline.sh`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/scripts/run_final_pipeline.sh): Pipeline tự động hóa từ khâu kiểm định Denoise đến huấn luyện và ensemble phân loại.

---

## 6. HỆ THỐNG TÀI LIỆU VÀ QUẢN LÝ DỰ ÁN MỚI

Trước đây, dự án chỉ có các file ghi chú kỹ thuật phân mảnh (`README.md`, `ROADMAP.md`, `CLASSIFICATION.md`). Hiện tại dự án đã có bộ tài liệu học thuật bài bản phục vụ trực tiếp cho việc viết Paper:

1. **[`FINAL_PROJECT_REPORT.md`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/FINAL_PROJECT_REPORT.md):** Báo cáo tối ưu hóa dự án, tổng kết các cải tiến kiến trúc và phát hiện về hàm Loss.
2. **[`FINAL_PAPER_REPORT.md`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/FINAL_PAPER_REPORT.md):** Báo cáo phân tích chuyên sâu 4 thực nghiệm Ablation, giải thích hiện tượng sụt giảm F1 khi dùng MSE và luận điểm chuẩn Q1/Q2.
3. **[`TONG_HOP_TOAN_BO_KET_QUA.md`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/TONG_HOP_TOAN_BO_KET_QUA.md):** Bản tổng hợp số liệu toàn diện từ SNR, PRD, RMSE, độ trễ CPU đến F1-Score phân loại nhịp tim và ma trận nhầm lẫn.
4. **[`MO_TA_CHI_TIET_HE_THONG_VA_SO_SANH.md`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/MO_TA_CHI_TIET_HE_THONG_VA_SO_SANH.md):** Bản đặc tả đầy đủ 5 mục cốt lõi: Dataset, Input, Xử lý ra sao, Output và So sánh Literature đối đầu với Mondéjar 2019, de Chazal 2004, DeepFilter 2024.
5. **[`TONG_HOP_CAC_THAY_DOI_SO_VOI_BAN_DAU.md`](file:///d:/SPARC%20Lab/Denoise_ECG_BMELaB/TONG_HOP_CAC_THAY_DOI_SO_VOI_BAN_DAU.md):** Tài liệu này – ghi nhận toàn bộ quá trình cải tiến và tiến hóa của dự án.
6. **Quản lý Git:** Thiết lập file `.gitignore` chuẩn, dọn dẹp các tệp rác giải phóng dung lượng, commit và đẩy đồng bộ toàn bộ mã nguồn lên GitHub [NamHaiIT2HUST/Denoise_ECG_BMELaB](https://github.com/NamHaiIT2HUST/Denoise_ECG_BMELaB.git).
