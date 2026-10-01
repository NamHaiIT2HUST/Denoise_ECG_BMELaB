# Denoise_ECG_BMELaB

Project ECG denoising 1D dùng lại dữ liệu đã preprocessing sẵn. Folder này bỏ pipeline WPD 2D, tập trung vào 5 baseline 1D và model mới `haar_sym_lite`.

## 1. Đường Dẫn Dữ Liệu

Mặc định code đọc dữ liệu đã preprocessing ở:

```bash
/home/sparc/workdir/sonnguyen/PPG_PCG_ECG/data/processed
```

Trong đó cần có:

```bash
manifest.csv
segments/*.npz
```

Không cần chạy lại preprocessing nếu folder trên đã tồn tại.

## 2. Cài Môi Trường

```bash
cd /home/sparc/workdir/sonnguyen/Denoise_ECG_BMELaB
pip install -r requirements.txt
```

Nếu dùng conda env có sẵn:

```bash
conda activate pytorch_py3.14
```

## 3. Model Mới

Model chính:

```bash
haar_sym_lite
```

Cấu hình mặc định:

```text
base = 32
expansion = 4
mid_depth = 3
wavelet = haar
loss = mse
```

Ý tưởng:

- DWT/IDWT 2 tầng theo wavelet
- Inverted depthwise-separable SE block
- Detail-gated skip connection ở nhánh high-frequency
- Residual output learning: `prediction = noisy + delta`
- Có thể sweep `mid_depth`, `alpha`, và `wavelet`

## 4. Chạy Riêng Model Mới

```bash
bash scripts/train_all_haar.sh
```

Hoặc chạy trực tiếp:

```bash
python -m src.train \
  --config configs/default.yaml \
  --model haar_sym_lite \
  --wavelet haar \
  --loss mse \
  --base 32 \
  --expansion 4 \
  --mid_depth 3
```

## 5. Chạy 5 Baseline

```bash
bash scripts/train_baselines.sh
```

Baseline gồm:

```text
dw_cnn
dw_se
dnn_dan
fcn
liwave
```

## 6. Chạy Tất Cả Model Chính

```bash
bash scripts/train_all_models.sh
```

Lệnh này chạy 5 baseline và `haar_sym_lite` cấu hình chính.

## 7. Sweep Alpha Cho Loss

Loss mixed:

```text
L = alpha * MSE + (1 - alpha) * L1
```

Chạy:

```bash
bash scripts/sweep_alpha.sh
```

Alpha được sweep:

```text
0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0
```

## 8. Sweep Depth

Ở đây `depth` là độ sâu bottleneck `mid_depth` của model mới.

```bash
bash scripts/sweep_depth.sh
```

Các giá trị:

```text
1, 2, 3, 4, 5
```

## 9. Sweep Wavelet

```bash
bash scripts/sweep_wavelets.sh
```

Wavelet mặc định:

```text
haar, db2, db4, sym4, coif1, bior2.2
```

Chạy sweep cả wavelet và alpha:

```bash
bash scripts/sweep_loss_wavelet.sh
```

## 10. Chạy Full Experiment

```bash
bash scripts/run_full_experiment.sh
```

Lệnh này chạy:

```text
train_all_models
sweep_alpha
sweep_depth
sweep_wavelets
eval_all
make_all_figures
```

## 11. Evaluate Và Tổng Hợp CSV

Evaluate toàn bộ run:

```bash
bash scripts/eval_all.sh
```

File sinh ra:

```text
runs/summary_all_models.csv
runs/all_segment_metrics.csv
runs/all_models_by_noise_snr.csv
```

Ý nghĩa:

| File | Nội dung |
|---|---|
| `summary_all_models.csv` | Trung bình toàn bộ test theo từng run |
| `all_segment_metrics.csv` | Toàn bộ kết quả từng segment |
| `all_models_by_noise_snr.csv` | Tổng hợp từng model, từng noise, từng SNR |

Các file tổng hợp có đủ metadata:

```text
run_name, model_name, wavelet, loss, alpha, base, expansion, mid_depth, noise_type, noise_snr
```

## 12. In Thông Số Model

Chỉ in tham số:

```bash
PYTHONPATH=. python scripts/count_params.py
```

In tham số + latency:

```bash
PYTHONPATH=. python scripts/report_model_complexity.py --latency
```

In nhiều wavelet:

```bash
PYTHONPATH=. python scripts/report_model_complexity.py \
  --models haar_sym_lite \
  --wavelets haar db2 db4 sym4 coif1 bior2.2 \
  --base 32 \
  --expansion 4 \
  --mid_depth 3 \
  --latency
```

Output:

```text
runs/model_params.csv
runs/model_complexity_report.csv
```

## 13. Vẽ Hình

Sau khi evaluate xong:

```bash
bash scripts/make_all_figures.sh
```

Hoặc vẽ waveform:

```bash
python -m src.visualize --config configs/default.yaml --run_glob "runs/*" --segment_index 0
```

Vẽ biểu đồ metric:

```bash
python scripts/plot_results.py \
  --summary runs/summary_all_models.csv \
  --noise_snr runs/all_models_by_noise_snr.csv \
  --out_dir figures/metrics
```

Output:

```text
figures/comparison_all_models/same_segment.png
figures/metrics/summary_rmse.png
figures/metrics/summary_prd.png
figures/metrics/summary_snrimp.png
figures/metrics/noise_model_snrimp.png
figures/metrics/noise_model_prd.png
```

## 14. Chạy Nhanh Sau Khi Sửa Code

```bash
PYTHONPATH=. python scripts/report_model_complexity.py --models haar_sym_lite --wavelets haar --latency
bash scripts/train_all_haar.sh
python -m src.eval --config configs/default.yaml --run_glob "runs/haar_sym_lite*"
python scripts/summarize_runs.py --run_glob "runs/haar_sym_lite*" --out runs/summary_haar_sym_lite.csv
```

## 15. Ghi Chú

- `config_effective.json` được lưu trong từng run để eval tự dựng lại đúng kiến trúc.
- Nếu đổi `base`, `expansion`, `mid_depth`, `wavelet`, run name sẽ tự ghi lại cấu hình.
- Với non-Haar wavelet, DWT/IDWT dùng filter từ `PyWavelets`.
