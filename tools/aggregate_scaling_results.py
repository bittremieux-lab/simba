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


def _is_cpu(row: dict) -> bool:
    return row.get("accelerator", "gpu") == "cpu"


def load_results(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """Return (gpu_rows, cpu_rows) sorted by unit count."""
    gpu_files = sorted(
        [f for f in results_dir.glob("*gpu_*_results.json")
         if not f.name.startswith("0")],
        key=lambda f: int(f.name.split("gpu")[0]),
    )
    cpu_files = sorted(
        results_dir.glob("*cpu_*_results.json"),
        key=lambda f: int(f.name.split("cpu")[0]),
    )
    if not gpu_files and not cpu_files:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    def _load(files):
        rows = []
        for fp in files:
            with open(fp) as f:
                rows.append(json.load(f))
        return rows

    return _load(gpu_files), _load(cpu_files)


def compute_table(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["num_gpus"])

    baseline = next((r for r in rows if r["num_gpus"] == 1), None)
    if baseline is None:
        print("Warning: no 1-unit baseline found; using first entry as baseline",
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


def compute_cpu_table(rows: list[dict]) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["num_gpus"])

    baseline = next((r for r in rows if r["num_gpus"] == 1), None)
    if baseline is None:
        baseline = rows[0]

    baseline_tp = baseline["throughput_samples_per_s"]

    table = []
    for r in rows:
        tp = r["throughput_samples_per_s"]
        n = r["num_gpus"]
        speedup = tp / baseline_tp
        efficiency = speedup / n
        table.append({
            "cores": n,
            "nodes": r["num_nodes"],
            "time (s)": r["time_s"],
            "samples": r["samples"],
            "throughput (samples/s)": round(tp, 1),
            "speedup": round(speedup, 3),
            "efficiency": round(efficiency, 3),
            "max cpu ram (GB)": r.get("max_cpu_ram_gb", "-"),
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


def save_csv(table: list[dict], output_dir: Path, filename: str = "scaling_results.csv"):
    import csv
    out_path = output_dir / filename
    headers = list(table[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table)
    print(f"\nCSV saved to: {out_path}")


def save_plot(table: list[dict], output_dir: Path):
    try:
        import matplotlib
        import matplotlib.ticker
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
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpus)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax = axes[2]
    ax.plot(gpus, [e * 100 for e in efficiency], "o-", color="steelblue", linewidth=2)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, label="100%")
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Scaling Efficiency")
    ax.set_ylim(0, 110)
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpus)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for i, (g, e) in enumerate(zip(gpus, efficiency)):
        ax.text(g, e * 100 + 1.5, f"{e*100:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "scaling_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to:  {out_path}")


def save_cpu_plot(cpu_table: list[dict], gpu_table,
                  output_dir: Path):
    """CPU strong-scaling plot, plus optional GPU-vs-CPU speedup bar."""
    try:
        import matplotlib
        import matplotlib.ticker
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping CPU plot")
        return

    cores = [r["cores"] for r in cpu_table]
    throughput_cpu = [r["throughput (samples/s)"] for r in cpu_table]
    efficiency_cpu = [r["efficiency"] for r in cpu_table]
    speedup_cpu = [r["speedup"] for r in cpu_table]

    ncols = 3 if gpu_table else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle("SimBA CPU Scaling — doduo (cpu node)", fontsize=14)

    ax = axes[0]
    ax.plot(cores, speedup_cpu, "o-", color="darkorange", linewidth=2, label="Actual")
    ax.plot(cores, cores, "--", color="gray", linewidth=1, label="Linear ideal")
    ax.set_xlabel("CPU cores")
    ax.set_ylabel("Speedup")
    ax.set_title("Speedup vs Linear Ideal")
    ax.legend()
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax = axes[1]
    ax.plot(cores, [e * 100 for e in efficiency_cpu], "o-", color="darkorange", linewidth=2)
    ax.axhline(y=70, color="red", linestyle=":", linewidth=1, label="70% target")
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, label="100%")
    ax.set_xlabel("CPU cores")
    ax.set_ylabel("Efficiency (%)")
    ax.set_title("Scaling Efficiency")
    ax.set_ylim(0, 110)
    ax.set_xscale("log", base=2)
    ax.set_xticks(cores)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8)
    for c, e in zip(cores, efficiency_cpu):
        ax.text(c, e * 100 + 1.5, f"{e*100:.1f}%", ha="center", va="bottom", fontsize=9)

    if gpu_table:
        ax = axes[2]
        # Best CPU throughput vs each GPU config
        best_cpu_tp = max(throughput_cpu)
        gpu_labels = [str(r["GPUs"]) + " GPU" + ("s" if r["GPUs"] > 1 else "")
                      for r in gpu_table]
        gpu_tps = [r["throughput (samples/s)"] for r in gpu_table]
        gpu_speedups = [tp / best_cpu_tp for tp in gpu_tps]
        colors = ["steelblue"] * len(gpu_speedups)
        bars = ax.bar(gpu_labels, gpu_speedups, color=colors)
        ax.axhline(y=1.0, color="darkorange", linestyle="--", linewidth=1,
                   label=f"Best CPU ({int(best_cpu_tp)} samp/s)")
        ax.set_xlabel("GPU configuration")
        ax.set_ylabel("Speedup over best CPU run")
        ax.set_title("GPU vs Best CPU Speedup")
        ax.legend(fontsize=8)
        for bar, s in zip(bars, gpu_speedups):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    f"{s:.1f}×", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = output_dir / "cpu_scaling_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"CPU plot saved to: {out_path}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {results_dir}")
    gpu_rows, cpu_rows = load_results(results_dir)
    print(f"Found {len(gpu_rows)} GPU file(s), {len(cpu_rows)} CPU file(s)")

    if gpu_rows:
        gpu_table = compute_table(gpu_rows)
        print("\n" + "=" * 80)
        print("GPU SCALING EFFICIENCY RESULTS")
        print("=" * 80)
        print_table(gpu_table)
        save_csv(gpu_table, output_dir, "scaling_results.csv")
        save_plot(gpu_table, output_dir)
    else:
        gpu_table = None

    if cpu_rows:
        cpu_table = compute_cpu_table(cpu_rows)
        print("\n" + "=" * 80)
        print("CPU SCALING EFFICIENCY RESULTS")
        print("=" * 80)
        print_table(cpu_table)
        save_csv(cpu_table, output_dir, "cpu_scaling_results.csv")
        save_cpu_plot(cpu_table, gpu_table, output_dir)
    else:
        print("\nNo CPU results found — skipping CPU table/plot.")
        print("Run cpu_scaling_experiment.slurm.sh on doduo to generate them.")


if __name__ == "__main__":
    main()
