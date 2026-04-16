#!/usr/bin/env python
"""Aggregate model-size benchmark results into a summary table and heatmap plots."""

import argparse
import json
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="experiments/model_size/results")
    p.add_argument("--output-dir", default="experiments/model_size")
    return p.parse_args()


def load_results(results_dir: Path) -> list:
    files = sorted(results_dir.glob("layers*_results.json"))
    if not files:
        print(f"No result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    rows = []
    for fp in files:
        with open(fp) as f:
            rows.append(json.load(f))
    return rows


def compute_table(rows: list) -> list:
    rows = sorted(rows, key=lambda r: (r["n_layers"], r["d_model"]))
    table = []
    for r in rows:
        n_params_m = r["n_params"] / 1e6
        size_label = _size_label(r["n_layers"], r["d_model"])
        table.append({
            "size": size_label,
            "n_layers": r["n_layers"],
            "d_model": r["d_model"],
            "emb_dim": r["embeddings_dim"],
            "n_params (M)": round(n_params_m, 2),
            "walltime (s)": r["time_s"],
            "throughput (samp/s)": r["throughput_samples_per_s"],
            "GPU mem (GB)": r["max_gpu_mem_gb"],
            "GPU util (%)": r.get("gpu_util_pct", "-"),
            "CPU RAM (GB)": r["max_cpu_ram_gb"],
        })
    return table


def _size_label(n_layers: int, d_model: int) -> str:
    """Assign small/medium/large label based on position in 3×3 grid."""
    score = n_layers * d_model
    thresholds = sorted(set(
        nl * dm for nl in [2, 5, 8] for dm in [128, 256, 512]
    ))
    idx = thresholds.index(score) if score in thresholds else -1
    if idx <= 2:
        return "small"
    elif idx <= 5:
        return "medium"
    else:
        return "large"


def print_table(table: list):
    headers = list(table[0].keys())
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in table))
                  for h in headers}
    sep = "+-" + "-+-".join("-" * col_widths[h] for h in headers) + "-+"
    print(sep)
    print("| " + " | ".join(h.ljust(col_widths[h]) for h in headers) + " |")
    print(sep)
    for r in table:
        print("| " + " | ".join(str(r[h]).ljust(col_widths[h]) for h in headers) + " |")
    print(sep)


def save_csv(table: list, output_dir: Path):
    import csv
    out = output_dir / "model_size_results.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)
    print(f"CSV saved to: {out}")


def save_plots(rows: list, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    layers = sorted(set(r["n_layers"] for r in rows))
    dmodels = sorted(set(r["d_model"] for r in rows))

    def make_matrix(key):
        m = np.full((len(layers), len(dmodels)), np.nan)
        for r in rows:
            i = layers.index(r["n_layers"])
            j = dmodels.index(r["d_model"])
            m[i, j] = r[key]
        return m

    walltime_mat = make_matrix("time_s")
    mem_mat = make_matrix("max_gpu_mem_gb")
    tp_mat = make_matrix("throughput_samples_per_s")
    params_mat = make_matrix("n_params")
    params_mat_m = params_mat / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("SimBA Model-Size Benchmark — 1× A100 80GB, accelgor", fontsize=13)

    xl = [str(d) for d in dmodels]
    yl = [str(l) for l in layers]

    def heatmap(ax, data, title, fmt, cmap="YlOrRd"):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(dmodels)))
        ax.set_xticklabels(xl)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(yl)
        ax.set_xlabel("d_model")
        ax.set_ylabel("n_layers")
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        for i in range(len(layers)):
            for j in range(len(dmodels)):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=9, color="black")

    heatmap(axes[0], walltime_mat, "Walltime (s)", "{:.0f}", cmap="YlOrRd")
    heatmap(axes[1], mem_mat, "Peak GPU Memory (GB)", "{:.1f}", cmap="YlOrRd")
    heatmap(axes[2], params_mat_m, "Parameters (M)", "{:.1f}", cmap="Blues")

    plt.tight_layout()
    out = output_dir / "model_size_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {out}")

    # Second figure: throughput vs params scatter
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    fig2.suptitle("SimBA: Throughput vs Model Size", fontsize=13)
    colors = plt.cm.viridis(np.linspace(0, 1, len(layers)))
    for i, nl in enumerate(layers):
        sub = [(r["n_params"] / 1e6, r["throughput_samples_per_s"])
               for r in rows if r["n_layers"] == nl]
        sub.sort()
        xs, ys = zip(*sub)
        ax2.plot(xs, ys, "o-", color=colors[i], label=f"{nl} layers")
        for x, y in zip(xs, ys):
            ax2.text(x, y + 0.5, f"{y:.0f}", ha="center", fontsize=8)
    ax2.set_xlabel("Parameters (M)")
    ax2.set_ylabel("Throughput (samples/s)")
    ax2.legend(title="n_layers")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out2 = output_dir / "model_size_throughput.png"
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Throughput plot saved to: {out2}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_results(results_dir)
    print(f"Found {len(rows)} result file(s)\n")

    table = compute_table(rows)
    print_table(table)
    save_csv(table, output_dir)
    save_plots(rows, output_dir)


if __name__ == "__main__":
    main()
