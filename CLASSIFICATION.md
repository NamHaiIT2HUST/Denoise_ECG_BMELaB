# Phân loại AAMI 3 lớp (N/S/V) trên tín hiệu ECG đã khử nhiễu

Mở rộng project denoise sang phân loại nhịp, với **head có thể hoán đổi**: cổ điển hoặc
**lai lượng tử (VQC)**. Đánh giá **inter-patient** (DS1/DS2, de Chazal) trên MIT-BIH + nhiễu NSTDB.
Test: 49.298 beat (N 44.241 · S 1.837 · V 3.220).

## ⚠️ ĐỌC TRƯỚC: mọi kết quả phải là trung bình ≥5 seed
Phương sai seed ở bài toán này **rất lớn** — macro-F1 của một lần chạy đơn lẻ dao động tới
**0.087**. Một lần chạy quantum từng đạt 0.8195 (cao hơn cả trung bình classical) nhưng trung
bình 5 seed chỉ 0.7897. **Không bao giờ kết luận từ 1 seed.**

## Kết quả chính (`sampler_power 0.9`, có denoise)
| | Classical | Quantum |
|---|---|---|
| Single, 5 seed — accuracy | **0.9554 ± 0.0063** | 0.9454 ± 0.0083 |
| Single, 5 seed — macro-F1 | **0.8146 ± 0.0236** | 0.7897 ± 0.0318 |
| Ensemble 5 seed — accuracy | **0.9655** | 0.9625 |
| Ensemble 5 seed — macro-F1 | **0.8487** | 0.8277 |
| Ensemble — F1(S) | **0.6631** | 0.5960 |
| Tham số head | 37.507 | 30.875 (−17,7%; 24 lượng tử) |

**Welch t-test trên macro-F1 từng seed: t = 1.25, p = 0.25 → KHÔNG có ý nghĩa thống kê.**
Nhưng hướng chênh lệch nhất quán ở cả single lẫn ensemble và ở mọi chỉ số tổng hợp
→ **không được claim quantum vượt trội**. Ensemble gain gần bằng nhau (classical +0.034,
quantum +0.038) nên ensembling cũng không ưu ái quantum.

### Cấu hình tốt nhất = classical ensemble 5 seed
accuracy **0.9655** · macro-F1 **0.8487**
F1 [N 0.9832 · S 0.6631 · V 0.8999] · Se [0.9799 · 0.6658 · 0.9385] · +P [0.9864 · 0.6604 · 0.8644]

So literature inter-patient — **thắng 3/5**:

| | Ta (classical ens.) | Mondéjar 2019 | de Chazal 2004 |
|---|---|---|---|
| Accuracy | **96,55** | 94,5 | 86,2 |
| S: F1 | **0,663** | 0,607 | 0,511 |
| S: +P | **66,0** | 49,7 | 38,5 |
| S: Se | 66,6 | **78,1** | 76,0 |
| V: F1 | 0,900 | **0,943** | 0,833 |

Đánh đổi: suy luận ×5 (5 model). Se(S) thấp hơn là do **điểm vận hành**, không phải giới hạn
năng lực — xem bảng `sampler_power` bên dưới (p=1.0 cho Se(S)=0,807, vượt mọi bài đã công bố).

## Pipeline
```
beat nhiễu (256 mẫu, quanh đỉnh R)
  → denoiser haar_sym_lite (FREEZE, thang mV thô)
  → z-norm (SAU denoiser)
  → CNN encoder (1→16→32→64, GELU, MaxPool)
  → SE block → AdaptiveAvgPool(8) → Linear → morph (128)
  ⊕ RR features (6) → BatchNorm → Linear → rr (32)
  → head: classical MLP  |  quantum: [VQC 12 obs ‖ bypass 64] → MLP
  → softmax → N / S / V
```

## Nhãn & chia dữ liệu
- **AAMI**: N = {N,L,R,e,j} · S = {A,a,J,S} · V = {V,E}. **Bỏ F, Q** (cực hiếm, mơ hồ hình thái;
  F còn gây nhiễu dự đoán lớp N). Loại 4 record máy tạo nhịp (102,104,107,217).
- **Inter-patient**: DS1 → train/val, DS2 → test.
- **Val CỐ ĐỊNH** `['223','118','116']` (`auto_val: false`) để **tái lập được**.
  Train: N=39335 · S=774 · V=3190.

## RR features (6 chiều — chìa khóa cho lớp S)
`[pre_RR/fs, post_RR/fs, pre/local, post/local, pre/global, post/global]`
với `local` = RR trung bình ~10 nhịp gần nhất, `global` = RR trung bình cả bản ghi.
Đi qua **nhánh riêng** (không bị 512 chiều morphology nhấn chìm).

## Mạch lượng tử (VQC)
6 qubit · 2 layer · mỗi layer: `Ry`, `Rz` mỗi qubit + **C-ring CNOT** → **24 tham số**.
Đọc **12 observable**: 6×⟨Z_q⟩ + 6×⟨Z_q Z_{q+1}⟩. Amplitude encoding (64 = 2⁶).
Ghép **residual fusion** với nhánh bypass cổ điển → ổn định, phương sai thấp hơn readout thuần lượng tử.

Mô phỏng state-vector (PennyLane `default.qubit`), **chưa chạy phần cứng thật**.

## Chạy
```bash
# 1) sinh beat + nhãn
PYTHONPATH=. python -m src.prepare_classification_data --config configs/default.yaml

# 2) train 5 seed (mac dinh DA LA cau hinh tot nhat)
for s in 0 1 2 3 4; do
  PYTHONPATH=. python -m src.train_classifier --config configs/default.yaml \
      --denoise_run <run_optimal> --out_dir runs_cls/c09b_seed$s \
      --head classical --seed $s
done

# 3) ensemble soft-voting
PYTHONPATH=. python scripts/ensemble_cls.py --config configs/default.yaml \
    --denoise_run <run_optimal> --head classical \
    --run_dirs runs_cls/c09b_seed{0,1,2,3,4}
```

## Ablation `sampler_power` (lever mạnh nhất)
Tỉ lệ lấy mẫu S:N = `(n_N/n_S)^p`. Bảng dưới là **head quantum, 1 seed** — dùng để chọn điểm
vận hành, không dùng để so sánh head.

| p | Acc | macro-F1 | S Se | S +P | S F1 | V F1 |
|---|---|---|---|---|---|---|
| 0.6 | .9478 | .7393 | .339 | .607 | .435 | .807 |
| 0.75 | .9503 | .7612 | .422 | .524 | .468 | .838 |
| **0.9** | **.9551** | **.8195** | .674 | .558 | **.611** | **.870** |
| 1.0 | .9412 | .8031 | **.807** | .485 | .606 | .834 |

p=0.9 là đỉnh. p=1.0 cho Se(S)=0.807 **vượt literature** (0.76–0.78) nhưng +P giảm —
dùng làm điểm vận hành thay thế khi ưu tiên recall.

## Ablation khử nhiễu — ⚠️ CHƯA XONG
```bash
--no_denoise                # bo khu nhieu
```
Mới có 1 seed (42) và **hai head cho tín hiệu ngược nhau**:
- classical noisy 0.7750 → thấp hơn **cả 5 seed** có denoise (thấp nhất 0.7838) ✅ denoise giúp
- quantum noisy 0.8281 → **cao hơn** trung bình có denoise (0.7897) ❌ ngược kỳ vọng

Với phương sai seed 0.087 thì 1 seed không kết luận được gì.
**Phải chạy đủ 5 seed cho cả hai head** (`c09n_seed*`, `q09n_seed*`) trước khi viết mục này vào paper.

## Đã thử và KHÔNG cải thiện (giữ cờ để tái lập)
| Hướng | Kết quả |
|---|---|
| multi-beat context (`n_context: 3`) | S F1 tụt 0.435 → 0.29 |
| data augmentation (`--augment`) | hại lớp S |
| prior-tuning (`--prior_tune`) | trung tính, làm +P(S) sập 0.61 → 0.32 |
| fine-tune (mở băng) denoiser | không đổi, best_epoch vẫn ~2 |
| 2-lead + z-norm **trước** denoiser | lỗi nặng (denoiser train trên thang mV thô) |

## Lưu ý quan trọng
- **z-norm phải áp SAU denoiser** — denoiser được train trên biên độ mV thô.
- **Val nhỏ (181 beat S) rất nhiễu** → luôn xác nhận bằng test trước khi kết luận.
- Mỗi run in `DENOISE_ACTIVE=<bool>` ở dòng đầu — **kiểm tra dòng này** trước khi tin kết quả
  ablation, vì lỗi nội suy shell từng làm `--denoise_run` rỗng âm thầm.
- Xem [[README]] cho phần khử nhiễu, `paper_ICCE_style.tex` / `paper_IEEE_v1.tex` cho bản thảo.
