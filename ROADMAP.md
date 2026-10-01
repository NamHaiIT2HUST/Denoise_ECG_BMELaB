# ROADMAP — Việc cần làm để công bố bài báo

Kế hoạch triển khai cụ thể sau khi rà soát `PROJECT_OVERVIEW.md` và đối chiếu với literature
hiện tại (2024–2026). Sắp xếp theo **thứ tự ưu tiên thực hiện**, không phải theo mục 3/4 của
overview. Mỗi việc có: mục tiêu, lý do, cách làm cụ thể trong codebase hiện tại, file cần
sửa/tạo, tiêu chí hoàn thành (Definition of Done), và ước lượng công sức.

Cập nhật: 2026-09-24 · Base: PROJECT_OVERVIEW.md v26

---

## Bảng tổng hợp

| # | Việc | Nhánh | Công sức | Bắt buộc cho BSPC? |
|---|---|---|---|---|
| 1 | Hoàn thành `sampler_power` 5 seed × 4 mức | B (phân loại) | ~2–4 giờ chạy | Không, nhưng đang nợ (mục 10 overview) |
| 2 | Bổ sung seed cho so sánh quantum (đủ lực kiểm định) | B (phân loại) | ~1 ngày chạy | Không, nhưng nâng chất lượng claim |
| 3 | Đo QT/QRS interval error trước–sau denoise | A (denoiser) | ~1 ngày code + vài giờ chạy | Nên có |
| 4 | So sánh với **DeepFilter** (Romero 2021, BSPC) | A (denoiser) | ~2–3 ngày | **Có — rủi ro bị reject nếu thiếu** |
| 5 | Ablation kiến trúc (SE / detail gate / residual) | A (denoiser) | ~1–2 ngày | Nên có |
| 6 | Tổng quát hoá trên database khác (zero-shot) | A (denoiser) | ~2–3 ngày | Nên có |
| 7 | *(tuỳ chọn)* So sánh DeScoD-ECG (diffusion SOTA) | A (denoiser) | ~1–2 tuần | Không, nhưng nâng tầm bài |
| 8 | *(tuỳ chọn)* Lượng tử hoá INT8 + đo trên ESP32 thật | A (denoiser) | ~1–2 tuần, cần phần cứng | Không, chỉ cho IEEE Access |

---

## 1. Hoàn thành `sampler_power` 5 seed × 4 mức — ✅ XONG (2026-09-24)

**Kết quả (classical + denoise, 5 seed, script `scripts/sweep_sampler_power.sh`):**

| p | Accuracy | macro-F1 | Se(S) | +P(S) | F1(S) |
|---|---|---|---|---|---|
| 0.60 | 0.9582 ± 0.0026 | 0.8120 ± 0.0220 | 0.568 ± 0.098 | 0.610 ± 0.066 | 0.582 ± 0.060 |
| 0.75 | 0.9566 ± 0.0055 | 0.8013 ± 0.0153 | 0.527 ± 0.075 | 0.602 ± 0.041 | 0.556 ± 0.029 |
| 0.90 | 0.9414 ± 0.0139 | 0.7754 ± 0.0515 | 0.584 ± 0.147 | 0.492 ± 0.095 | 0.524 ± 0.100 |
| 1.00 | 0.9458 ± 0.0117 | 0.7977 ± 0.0180 | 0.677 ± 0.105 | 0.526 ± 0.071 | 0.582 ± 0.035 |

- ANOVA macro-F1 giữa 4 mức: F=1.025, **p=0.408** (không khác biệt); Se(S) p=0.299; +P(S) p=0.083.
- Welch p=1.0 vs p=0.6: Se(S) p=0.169, +P(S) p=0.124 → xu hướng đánh đổi Se↑/+P↓ khi p tăng
  có, nhưng **chưa đạt ý nghĩa thống kê** ở n=5.
- **Hệ quả cho paper:** `p=0.9` (mặc định hiện tại) **không phải mức tốt nhất** — thực ra có
  macro-F1 thấp nhất và phương sai lớn nhất (0.0515, seed0 chỉ 0.685). Bảng 1-seed cũ (0.9 là tốt nhất)
  bị bác bỏ. Không nên claim "p=0.9 tối ưu" trong bài; nên nêu "p không ảnh hưởng đo được macro-F1".
- Cân nhắc đổi mặc định sang p=0.6 (macro-F1 cao nhất, phương sai thấp), nhưng phải chạy lại
  các cấu hình A–D nếu đổi, nên chỉ ghi nhận, không đổi ngay.

**Mục tiêu.** Thay bảng quét `p` 1-seed đã bị huỷ (mục 4.6, 8 của overview — hai máy cho kết
quả ngược nhau) bằng bảng 5-seed đáng tin cậy.

**Cách làm.** Dùng lại logic mục 9 trong `Train_Classifier_Colab.ipynb`, mở rộng vòng lặp seed:

```python
SEEDS = [0, 1, 2, 3, 4]
P_VALUES = [0.6, 0.75, 0.9, 1.0]
for p in P_VALUES:
    for s in SEEDS:
        out = f'{R}/sp{int(p*100)}_seed{s}'
        # train_classifier với --sampler_power p --seed s --head classical --denoise_run DENOISE_RUN
```

Lưu kết quả theo `sp{p}_seed{s}/metrics.json` (khác với file cũ `sp{p}/metrics.json` không có
seed, giữ file cũ lại để đối chiếu). Sau khi xong, tính trung bình ± std mỗi mức `p`, chạy
Welch/ANOVA giữa các mức để xem chênh lệch có ý nghĩa không (đúng chuẩn nghiêm ngặt mục 6 của
overview).

**File cần sửa.** `Train_Classifier_Colab.ipynb` (mục 9) — hoặc tách thành script riêng
`scripts/sweep_sampler_power.sh` cho khớp quy ước `sweep_*.sh` sẵn có.

**Definition of Done.** Bảng `p × (Acc, macro-F1, Se(S), +P(S))` với mean±std trên 5 seed +
kết quả kiểm định thống kê giữa mức tốt nhất và mặc định (0.9).

---

## 2. Bổ sung seed cho so sánh quantum vs classical — ✅ XONG (2026-09-25)

**Kết quả 20 seed (quantum):**

| Cấu hình | n | macro-F1 | F1(S) | Se(S) | +P(S) |
|---|---|---|---|---|---|
| A. quantum + denoise | 20 | 0.7843 ± 0.0283 | 0.519 | 0.580 | 0.490 |
| C. quantum + nhiễu | 20 | 0.8030 ± 0.0138 | 0.580 | 0.696 | 0.503 |

- **Welch p = 0.013, Mann-Whitney p = 0.007, Cohen d = −0.84**: đây là khác biệt **CÓ ý nghĩa
  thống kê**, nhưng **ngược kỳ vọng** — quantum trên tín hiệu **đã khử nhiễu** thấp hơn 0.019
  macro-F1 so với trên tín hiệu nhiễu.
- Phương sai seed cũng khác: Levene p = 0.031 (denoise 0.028 vs nhiễu 0.014). Xác nhận giả thuyết
  mục 6.3 overview ("khử nhiễu làm tăng phương sai") ở head quantum.
- So với classical (chỉ n=5): Q+denoise vs C+denoise p=0.29; Q+nhiễu vs C+nhiễu p=0.89 →
  **quantum vs classical vẫn không phân biệt được**.
- **Cần cập nhật claim trong overview mục 7:** "không claim khử nhiễu giúp phân loại" nay mạnh
  hơn: ở head quantum, khử nhiễu có xu hướng **làm giảm** macro-F1. Kết luận này chỉ áp dụng cho
  quantum (classical n=5 chưa đủ lực). Giải thích khả dĩ (chưa kiểm chứng, không được viết như
  sự thật): denoiser train trên MSE/L1 làm mượt đặc trưng hình thái mà lớp S cần.
- Việc cần làm thêm: chạy classical 20 seed (c09b, c09n) để biết hiệu ứng có chung cho cả 2 head.

**Mục tiêu.** Mục 6.2 overview đã tính: so sánh **quantum denoise vs quantum nhiễu** (Cohen's
d=1.01) chỉ cần **~16 seed/nhóm** để đạt power 0.8 — khả thi. So sánh **classical vs quantum**
(d=0.68) cần ~35 seed/nhóm — nặng hơn nhưng vẫn khả thi với GPU hiện có.

**Cách làm.** Mở rộng `SEEDS` trong `Train_Classifier_Colab.ipynb` (mục 6.0) từ `[0..4]` sang
`[0..19]` (20 seed) cho **tối thiểu 2 cấu hình**: `q09_seed` (quantum+denoise) và `q09n_seed`
(quantum+nhiễu) — đây là cặp so sánh rẻ nhất mà có thể kết luận **có ý nghĩa thống kê**. Cấu
hình classical (`c09b`, `c09n`) có thể giữ ít seed hơn (10 seed) vì so sánh với chúng cần cỡ
mẫu lớn hơn nhiều nên khó đạt được trong thời gian ngắn — nêu rõ giới hạn này trong bài thay vì
cố chạy 35 seed × 4 cấu hình (140 run, quá tốn thời gian).

Toàn bộ hạ tầng skip-nếu-đã-xong đã có sẵn (`train_one` kiểm tra `metrics.json`), chỉ cần đổi
`SEEDS` và chạy lại `train_config(*CONFIGS[0])` / `train_config(*CONFIGS[2])`.

**File cần sửa.** `Train_Classifier_Colab.ipynb` (mục 6.0, biến `SEEDS`).

**Definition of Done.** Welch t-test (hoặc Mann-Whitney nếu phân phối không chuẩn) giữa
quantum+denoise và quantum+nhiễu với n≥16/nhóm, báo cáo **kết luận dứt khoát** (có hoặc không
có ý nghĩa) thay vì "chưa đủ lực kiểm định" như hiện tại.

---

## 3. Đo QT/QRS interval error trước–sau khử nhiễu

**Mục tiêu.** Hiện tại đánh giá denoiser chỉ bằng chỉ số dạng sóng chung (RMSE/PRD/ΔSNR/cosine
— mục 3.5 overview). Literature xác nhận **đo lường sai số khoảng QT/QRS** là chỉ số đặc trưng
chẩn đoán được công nhận nhưng phần lớn paper denoising bỏ qua. Đây là bằng chứng định lượng
cho câu claim "không làm méo hình thái P–QRS–T" trong title, thay vì khẳng định suông.

**Cách làm.**
1. Thêm `neurokit2` vào `requirements.txt` (thư viện delineation ECG chuẩn, tách được đỉnh
   P/Q/R/S/T qua wavelet-based method — không cần tự viết detector từ đầu).
2. Viết `scripts/eval_diagnostic_intervals.py`: với mỗi segment test (theo đúng noise×SNR đã
   có trong `eval.py`), chạy delineation trên (a) tín hiệu sạch gốc → nhãn tham chiếu, (b) tín
   hiệu nhiễu, (c) tín hiệu đã khử nhiễu bởi `haar_sym_lite`. Tính sai số tuyệt đối khoảng QT
   và độ rộng QRS (ms) so với tham chiếu.
3. Gộp kết quả theo noise-type × SNR giống định dạng bảng mục 3.5, dùng lại
   `scripts/merge_noise_snr_summary.py` làm mẫu.

**File cần tạo/sửa.** `requirements.txt` (+neurokit2), `scripts/eval_diagnostic_intervals.py`
(mới), tái dùng `src/preprocess/beats.py` để lấy vị trí R-peak đã có.

**Definition of Done.** Bảng "QT-interval error (ms)" và "QRS-width error (ms)" theo
noise×SNR, so sánh tín hiệu nhiễu vs đã khử nhiễu — kỳ vọng sai số giảm rõ rệt sau khử nhiễu.

---

## 4. So sánh với DeepFilter (Romero et al. 2021, BSPC) — BẮT BUỘC

**Mục tiêu.** DeepFilter đăng **chính trên Biomedical Signal Processing and Control** — tạp
chí đích của bạn — cùng dùng NSTDB để đánh giá baseline wander. Reviewer BSPC gần như chắc
chắn sẽ hỏi tại sao không so sánh. Thiếu bước này là rủi ro reject lớn nhất hiện tại.

**Cách làm (khuyến nghị: tái hiện lại kiến trúc trong codebase hiện tại, không tích hợp
chéo framework).** Code gốc của DeepFilter dùng Keras/TensorFlow
(`github.com/fperdigon/DeepFilter`); tích hợp chéo với pipeline PyTorch hiện tại sẽ rất mất
công. Thay vào đó:
1. Đọc kiến trúc DeepFilter (Inception-style multi-branch FCN, dilated convolution) từ paper/
   code tham khảo, cài đặt lại bằng PyTorch thành `src/models/deepfilter.py`, theo đúng khuôn
   các baseline khác (`dw_cnn.py`, `fcn.py`, `dnn_dan.py`) — cùng interface `forward(x)`.
2. Thêm vào `scripts/train_baselines.sh` để train với **cả hai loss** (`mixed α=0.8` và
   `mse`) giống giao thức công bằng đã áp dụng cho 5 baseline hiện có (mục 3.6 overview).
3. Ghi rõ trong bài đây là **tái hiện lại theo kiến trúc mô tả trong paper gốc**, trích dẫn
   đầy đủ, không tuyên bố là bản chính thức của tác giả.
4. **Lưu ý phạm vi:** DeepFilter trong paper gốc chỉ nhắm baseline wander (bw). Khi so sánh,
   hoặc (a) chỉ so trên nhiễu `bw` cho công bằng đúng phạm vi gốc, hoặc (b) so trên cả 3 loại
   nhiễu (bw/ma/em) và nêu rõ đây là mở rộng đánh giá ngoài phạm vi thiết kế gốc của họ.

**File cần tạo/sửa.** `src/models/deepfilter.py` (mới), `scripts/train_baselines.sh`,
bảng so sánh mục 3.6 (`PROJECT_OVERVIEW.md`) thêm 1 hàng.

**Definition of Done.** DeepFilter xuất hiện trong bảng so sánh baseline (mục 3.6) với đầy đủ
ΔSNR/PRD/RMSE/cosine/latency/tham số, kèm kiểm định Wilcoxon ghép cặp với `optimal` giống các
baseline khác.

---

## 5. Ablation kiến trúc: SE block / detail gate / residual learning

**Mục tiêu.** Hiện tại đã chứng minh "số tham số không dự báo chất lượng" (Spearman, mục 3.6.3)
nhưng chưa chứng minh **từng thành phần kiến trúc đề xuất** (SE, detail gate, học phần dư) thực
sự có đóng góp — đây là câu hỏi ablation kinh điển reviewer sẽ hỏi.

**Cách làm.** Thêm cờ ablation vào `src/models/haar_sym_lite.py` (hoặc tạo biến thể riêng):
- `no_se`: bỏ SE block trong ISE, chỉ giữ depthwise-separable conv
- `no_detail_gate`: giữ nguyên `c_D` không qua gating (`ĉ_D = c_D`)
- `no_residual`: bỏ residual, học trực tiếp `x̂ = Δ` thay vì `x̂ = x + Δ`

Train mỗi biến thể ở cấu hình tối ưu hiện tại (`db4`, `mid_depth=5`, `mixed α=0.8`), 3 seed đủ
để thấy hiệu ứng lớn (khác với ablation phân loại — đây là ablation kiến trúc, hiệu ứng dự
kiến lớn hơn nhiều so với phương sai seed 0.087 macro-F1 bên nhánh B).

**File cần sửa.** `src/models/haar_sym_lite.py` (thêm cờ), `src/train.py` (thêm CLI args),
bảng ablation mới trong `PROJECT_OVERVIEW.md` mục 3.

**Definition of Done.** Bảng 4 dòng (full model, −SE, −detail gate, −residual) × ΔSNR/PRD/RMSE,
mỗi dòng giảm rõ rệt so với full model — nếu dòng nào KHÔNG giảm đáng kể, ghi nhận trung thực
(giống tinh thần null-result đã làm ở nhánh B) thay vì giấu đi.

---

## 6. Tổng quát hoá trên database khác (zero-shot)

**Mục tiêu.** Toàn bộ đánh giá hiện tại chỉ trên MIT-BIH — reviewer sẽ hỏi "có generalize
không". Chứng minh denoiser **đã train xong, đóng băng**, áp trực tiếp lên bản ghi từ database
khác vẫn hoạt động tốt là bằng chứng mạnh về tính tổng quát.

**Cách làm.**
1. Chọn 1 database PhysioNet khác có ECG sạch tương tự (khuyến nghị: **INCART** — 12-lead,
   dân số khác MIT-BIH, hoặc **QT Database** — đã được DeepFilter dùng nên tiện so sánh chéo
   ở mục 4).
2. Tải qua `wfdb.dl_database` (đã có sẵn pattern trong `Train_Classifier_Colab.ipynb`), viết
   config mới `configs/incart.yaml` kế thừa `configs/default.yaml`, chỉ đổi `dataset_root`.
3. Trộn nhiễu NSTDB y hệt quy trình mục 2.3 overview, chạy `src/eval.py` với checkpoint
   `optimal` đã có — **không train lại**, đúng nghĩa zero-shot.

**File cần tạo/sửa.** `configs/incart.yaml` (mới), tái dùng `src/eval.py` không đổi.

**Definition of Done.** Bảng ΔSNR/PRD/RMSE/cosine trên database mới, so sánh với kết quả trên
MIT-BIH (mục 3.5) để thấy mức độ giảm hiệu năng (nếu có) khi chuyển domain.

---

## 7. *(Tuỳ chọn, nặng)* So sánh với DeScoD-ECG (diffusion model SOTA)

**Mục tiêu.** DeScoD-ECG là SOTA hiện tại cho baseline wander removal (hơn baseline cũ ~20%),
nhưng là mô hình diffusion — nặng, chậm hơn nhiều so với `haar_sym_lite` (93K tham số, 9.12ms).
So sánh trực tiếp biến điểm yếu latency (mục 3.6 overview) thành **lợi thế đóng khung**:
*"đạt fidelity cạnh tranh với SOTA diffusion nhưng nhanh hơn hàng trăm/nghìn lần, khả thi
nhúng — điều diffusion model không làm được."*

**Cách làm.** Kiểm tra code công khai của DeScoD-ECG (arXiv 2208.00542) có sẵn không; nếu có,
chạy inference (không cần train lại từ đầu nếu có checkpoint) trên đúng tập test hiện tại, đo
cùng bộ chỉ số + latency. Nếu không có checkpoint công khai, cân nhắc bỏ qua việc này (chi phí
train diffusion từ đầu quá cao so với lợi ích) và chỉ **thảo luận bằng số liệu trích dẫn từ
paper gốc** trong phần related work + latency comparison định tính.

**Definition of Done.** Biểu đồ latency-vs-fidelity (trục X: ms/segment, trục Y: ΔSNR hoặc
PRD) đặt `haar_sym_lite` cạnh DeScoD-ECG và các baseline khác — minh hoạ trực quan trade-off.

---

## 8. *(Tuỳ chọn)* Lượng tử hoá INT8 thật + đo trên ESP32

**Mục tiêu.** Hiện tại "91 KB INT8, vừa flash ESP32" là **ước tính**, chưa đo thật (mục 3.6
overview). Số đo thật trên phần cứng thật sẽ mạnh hơn nhiều cho hướng IEEE Access (ưu tiên
ứng dụng thực tế) — nhưng cần có board ESP32 vật lý.

**Cách làm.** Dùng `torch.ao.quantization` (post-training static quantization) lượng tử hoá
`haar_sym_lite` xuống INT8, export ONNX → chuyển sang TFLite Micro hoặc dùng `esp-dl`, đo trực
tiếp: kích thước file thật, RAM runtime, latency trên board.

**Rủi ro/phụ thuộc.** Cần có board ESP32 trong tay; nếu không có, có thể bỏ qua bước đo thật
và chỉ tinh chỉnh lại câu chữ trong bài từ "vừa flash ESP32" (khẳng định) sang "ước tính kích
thước sau lượng tử hoá INT8, khả thi nhúng trên MCU lớp ESP32" (thận trọng hơn, không cần đo
thật vẫn hợp lệ về mặt khoa học).

**Definition of Done.** Bảng kích thước model thật (KB) + latency đo trên board, HOẶC (nếu
không có phần cứng) câu chữ trong bài được điều chỉnh về mức thận trọng phù hợp.

---

## Timeline gợi ý

| Giai đoạn | Việc | Ghi chú |
|---|---|---|
| **Tuần 1** | #1, #2, #3 | Toàn bộ dùng hạ tầng có sẵn (GPU đã setup), không cần code mới nhiều |
| **Tuần 2** | #4, #5 | #4 là việc quan trọng nhất, nên bắt đầu sớm vì có thể phát sinh vấn đề khi tái hiện kiến trúc |
| **Tuần 3** | #6 | Có thể làm song song với #4/#5 nếu có người thứ hai |
| **Sau đó, nếu còn thời gian** | #7, #8 | Không bắt buộc, cân nhắc theo deadline nộp bài |

## Sau khi hoàn thành #1–#6

Cập nhật lại `PROJECT_OVERVIEW.md` (thêm kết quả mới vào mục 3.6, 4.6, 5), viết lại phần
Related Work của `paper_IEEE_v1.tex` / `paper_ICCE_style.tex` để trích dẫn DeepFilter,
DeScoD-ECG, và các paper quantum-ECG 2024–2026 đã tìm được, sau đó rà lại toàn bộ claim ở
mục 7 overview (Được/Không được kết luận) xem có gì cần cập nhật theo số liệu mới không.
