import argparse
from pathlib import Path
import glob
import json
import pandas as pd


def load_run_meta(run_dir):
    path = run_dir / "config_effective.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return {
        "wavelet": cfg.get("data", {}).get("wavelet"),
        "loss": cfg.get("loss", {}).get("name"),
        "alpha": cfg.get("loss", {}).get("alpha"),
        "base": cfg.get("model", {}).get("base"),
        "expansion": cfg.get("model", {}).get("expansion"),
        "mid_depth": cfg.get("model", {}).get("mid_depth"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_glob", default="runs/*")
    parser.add_argument("--out", default="runs/all_models_by_noise_snr.csv")
    args = parser.parse_args()

    rows = []
    for run_path in sorted(glob.glob(args.run_glob)):
        run_dir = Path(run_path)
        csv_path = run_dir / "test_summary_by_noise_snr.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.insert(0, "run_name", run_dir.name)
        if "model_name" not in df.columns:
            df.insert(1, "model_name", run_dir.name.split("_wav")[0])
        for key, value in load_run_meta(run_dir).items():
            if key not in df.columns:
                df[key] = value
        rows.append(df)

    if not rows:
        raise FileNotFoundError("No test_summary_by_noise_snr.csv found.")

    out_df = pd.concat(rows, ignore_index=True)
    first = ["run_name", "model_name", "wavelet", "loss", "alpha", "base", "expansion", "mid_depth", "noise_type", "noise_snr"]
    out_df = out_df[first + [c for c in out_df.columns if c not in first]]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
