import argparse
from pathlib import Path
import pandas as pd
import torch

from src.models import build_model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model):
    return sum(p.numel() for p in model.parameters())


def forward_latency_ms(model, length, device, warmup=10, repeats=50):
    model = model.to(device).eval()
    x = torch.randn(1, 1, length, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                _ = model(x)
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / repeats
        import time
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        return (time.perf_counter() - t0) * 1000.0 / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/model_complexity_report.csv")
    parser.add_argument("--models", nargs="+", default=["dw_cnn", "dw_se", "dnn_dan", "fcn", "liwave", "haar_sym_lite"])
    parser.add_argument("--wavelets", nargs="+", default=["haar"])
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--expansion", type=int, default=4)
    parser.add_argument("--mid_depth", type=int, default=3)
    parser.add_argument("--length", type=int, default=8192)
    parser.add_argument("--latency", action="store_true")
    args = parser.parse_args()

    rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for wavelet in args.wavelets:
        for name in args.models:
            try:
                model = build_model(
                    name,
                    wavelet=wavelet,
                    base=args.base,
                    expansion=args.expansion,
                    mid_depth=args.mid_depth,
                )
                trainable = count_trainable(model)
                row = {
                    "model_name": name,
                    "wavelet": wavelet,
                    "base": args.base if name == "haar_sym_lite" else None,
                    "expansion": args.expansion if name == "haar_sym_lite" else None,
                    "mid_depth": args.mid_depth if name == "haar_sym_lite" else None,
                    "trainable_params": trainable,
                    "total_params": count_total(model),
                    "param_size_mb_fp32": trainable * 4 / (1024 ** 2),
                }
                if args.latency:
                    ms = forward_latency_ms(model, args.length, device)
                    row["device"] = device
                    row["forward_ms_batch1"] = ms
                    row["segments_per_second_batch1"] = 1000.0 / ms
                rows.append(row)
            except Exception as exc:
                rows.append({"model_name": name, "wavelet": wavelet, "error": str(exc)})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
