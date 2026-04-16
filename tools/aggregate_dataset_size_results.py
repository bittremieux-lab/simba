#!/usr/bin/env python
"""Aggregate dataset-size benchmark results into a CSV and a PNG plot.

Usage:
    python tools/aggregate_dataset_size_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = Path("experiments/dataset_size/results")
OUTPUT_DIR = Path("experiments/dataset_size")


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*_samples_*_results.json")):
        with open(path) as f:
            data = json.load(f)
        n_m = data["n_samples"] / 1_000_000
        rows.append({
            "n_samples_M": n_m,
            "n_samples": data["n_samples"],
            "steps": data["steps_measured"],
            "wall_time_s": data["wall_time_s"],
            "wall_time_hms": data["wall_time_hms"],
            "throughput_samples_per_s": data["throughput_samples_per_s"],
            "max_gpu_mem_gb": data["max_gpu_mem_gb"],
            "max_cpu_ram_gb": data["max_cpu_ram_gb"],
            "gpu_type": data.get("gpu_type", "A100"),
        })
    df = pd.DataFrame(rows).sort_values("n_samples").reset_index(drop=True)
    return df


def plot_results(df: pd.DataFrame, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Wall time vs n_samples
    ax = axes[0]
    ax.plot(df["n_samples_M"], df["wall_time_s"] / 60, "o-", color="steelblue", linewidth=2)
    # Ideal linear line from 1M
    ref_x = df["n_samples_M"].iloc[0]
    ref_y = df["wall_time_s"].iloc[0] / 60
    ideal = [ref_y * (x / ref_x) for x in df["n_samples_M"]]
    ax.plot(df["n_samples_M"], ideal, "--", color="gray", label="Ideal (linear)")
    ax.set_xlabel("Dataset size (M samples)")
    ax.set_ylabel("Wall time (minutes)")
    ax.set_title("Training time vs dataset size\n(1× A100 80GB)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Throughput (should be flat)
    ax = axes[1]
    ax.plot(df["n_samples_M"], df["throughput_samples_per_s"], "s-", color="darkorange", linewidth=2)
    ax.axhline(df["throughput_samples_per_s"].mean(), color="gray", linestyle="--",
               label=f"Mean: {df['throughput_samples_per_s'].mean():.0f} samples/s")
    ax.set_xlabel("Dataset size (M samples)")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title("Throughput vs dataset size\n(should be constant)")
    ax.set_ylim(0, df["throughput_samples_per_s"].max() * 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle("SimBA Dataset-Size Scalability — accelgor (A100)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = output_dir / "dataset_size_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    df = load_results(RESULTS_DIR)
    if df.empty:
        print("No results found — check experiments/dataset_size/results/")
        return

    csv_path = OUTPUT_DIR / "dataset_size_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    print("\nResults table:")
    print(df[["n_samples_M", "steps", "wall_time_hms", "throughput_samples_per_s",
              "max_gpu_mem_gb"]].to_string(index=False))

    plot_results(df, OUTPUT_DIR)


if __name__ == "__main__":
    main()
