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
    parser.add_argument("--out", default="runs/summary_all_models.csv")
    args = parser.parse_args()

    rows = []
    for run_path in sorted(glob.glob(args.run_glob)):
        run_dir = Path(run_path)
        metrics_path = run_dir / "test_segment_metrics.csv"
        if not metrics_path.exists():
            continue
        df = pd.read_csv(metrics_path)
        row = df.mean(numeric_only=True).to_dict()
        row["run_name"] = run_dir.name
        row["model_name"] = df["model_name"].iloc[0] if "model_name" in df else run_dir.name.split("_wav")[0]
        row.update(load_run_meta(run_dir))
        rows.append(row)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        first = ["model_name", "run_name", "wavelet", "loss", "alpha", "base", "expansion", "mid_depth"]
        summary = summary[first + [c for c in summary.columns if c not in first]]
    summary.to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
