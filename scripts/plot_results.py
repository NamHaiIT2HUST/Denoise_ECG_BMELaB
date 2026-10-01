import argparse
from pathlib import Path
import pandas as pd

from src.utils.plots import save_metric_bar, save_pivot_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="runs/summary_all_models.csv")
    parser.add_argument("--noise_snr", default="runs/all_models_by_noise_snr.csv")
    parser.add_argument("--out_dir", default="figures/metrics")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.summary)
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        if not df.empty:
            save_metric_bar(df, "rmse", out_dir / "summary_rmse.png", higher_is_better=False, title="Average RMSE")
            save_metric_bar(df, "prd", out_dir / "summary_prd.png", higher_is_better=False, title="Average PRD")
            save_metric_bar(df, "snr_imp", out_dir / "summary_snrimp.png", higher_is_better=True, title="Average SNR improvement")

            if "wavelet" in df.columns and df["wavelet"].nunique() > 1:
                save_pivot_heatmap(df, "wavelet", "loss", "snr_imp", out_dir / "wavelet_loss_snrimp.png", title="SNRimp by wavelet and loss")
            if "alpha" in df.columns and df["alpha"].nunique() > 1:
                save_pivot_heatmap(df, "wavelet", "alpha", "snr_imp", out_dir / "wavelet_alpha_snrimp.png", title="SNRimp by wavelet and alpha")
            if "mid_depth" in df.columns and df["mid_depth"].nunique() > 1:
                save_pivot_heatmap(df, "wavelet", "mid_depth", "snr_imp", out_dir / "wavelet_depth_snrimp.png", title="SNRimp by wavelet and depth")

    noise_path = Path(args.noise_snr)
    if noise_path.exists():
        df = pd.read_csv(noise_path)
        if not df.empty:
            save_pivot_heatmap(df, "noise_type", "model_name", "snr_imp", out_dir / "noise_model_snrimp.png", title="SNRimp by noise and model")
            save_pivot_heatmap(df, "noise_type", "model_name", "prd", out_dir / "noise_model_prd.png", title="PRD by noise and model")

    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
