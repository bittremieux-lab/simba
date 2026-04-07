#!/usr/bin/env python
"""Aggregate scaling benchmark results into a summary table."""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        default="experiments/scaling_efficiency/results",
        help="Directory containing *_results.json files",
    )
    p.add_argument(
        "--output-dir",
        default="experiments/scaling_efficiency",
        help="Directory to write summary CSV and plot PNG",
    )
    return p.parse_args()


def load_results(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("*_results.json"),
                   key=lambda f: int(f.name.split("gpu")[0]))
    if not files:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for fp in files:
        with open(fp) as f:
            rows.append(json.load(f))
    return rows


def compute_table(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["num_gpus"])

    baseline = next((r for r in rows if r["num_gpus"] == 1), None)
    if baseline is None:
        print("Warning: no 1-GPU baseline found; using first entry as baseline",
              file=sys.stderr)
        baseline = rows[0]

    baseline_tp = baseline["throughput_samples_per_s"]

    table = []
    for r in rows:
        tp = r["throughput_samples_per_s"]
        n = r["num_gpus"]
        speedup = tp / baseline_tp
        efficiency = speedup / n
        table.append({
            "GPUs": n,
            "nodes": r["num_nodes"],
            "time (s)": r["time_s"],
            "samples": r["samples"],
            "throughput (samples/s)": round(tp, 1),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 3),
            "max gpu mem (GB)": r.get("max_gpu_mem_gb", r.get("max_mem_gb", "-")),
            "max cpu ram (GB)": r.get("max_cpu_ram_gb", "-"),
            "gpu_util (%)": r.get("gpu_util_pct", "-"),
        })
    return table


def print_table(table: list[dict]):
    """Pretty-print the results table."""
    headers = list(table[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in table))
                  for h in headers}

    sep = "+-" + "-+-".join("-" * col_widths[h] for h in headers) + "-+"
    header_row = "| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |"

    print(sep)
    print(header_row)
    print(sep)
    for r in table:
        row = "| " + " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers) + " |"
        print(row)
    print(sep)


def save_csv(table: list[dict], output_dir: Path):
    import csv
    out_path = output_dir / "scaling_results.csv"
    headers = list(table[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table)
    print(f"\nCSV saved to: {out_path}")


def save_plot(table: list[dict], output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    gpus = [r["GPUs"] for r in table]
    throughput = [r["throughput (samples/s)"] for r in table]
    efficiency = [r["efficiency"] for r in table]
    speedup = [r["speedup"] for r in table]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("SimBA DDP Scaling Efficiency — accelgor A100 80GB", fontsize=14)

    ax = axes[0]
    ax.bar([str(g) for g in gpus], throughput, color="steelblue")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Throughput (samples/s)")
    ax.set_title("Throughput")
    for i, (g, tp) in enumerate(zip(gpus, throughput)):
        ax.text(i, tp * 1.01, f"{tp:.0f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    ax.plot(gpus, speedup, "o-", color="steelblue", linewidth=2, label="Actual")
    ax.plot(gpus, gpus, "--", color="gray", linewidth=1, label="Linear ideal")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Linear Ideal")
    ax.legend()
    ax.set_xticks(gpus)

    ax = axes[2]
    ax.plot(gpus, [e * 100 for e in efficiency], "o-", color="steelblue", linewidth=2)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, label="100%")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Scaling Efficiency")
    ax.set_ylim(0, 110)
    ax.set_xticks(gpus)
    for i, (g, e) in enumerate(zip(gpus, efficiency)):
        ax.text(g, e * 100 + 1.5, f"{e*100:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "scaling_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to:  {out_path}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    rows = load_results(results_dir)
    print(f"Found {len(rows)} result file(s)")

    table = compute_table(rows)

    print("\n" + "=" * 80)
    print("SCALING EFFICIENCY RESULTS")
    print("=" * 80)
    print_table(table)

    save_csv(table, output_dir)
    save_plot(table, output_dir)


if __name__ == "__main__":
    main()
