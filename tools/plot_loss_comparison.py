#!/usr/bin/env python3
"""Compare Adam and Muon experiments using saved loss history and wall-clock time."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_experiment_data(exp_dir: Path) -> dict:
    history_path = exp_dir / "checkpoints" / "loss_history.json"
    history = json.loads(history_path.read_text())
    return {
        "name": exp_dir.name,
        "history": history,
    }


def latest_experiment(results_dir: Path, name_prefix: str) -> Path | None:
    matches = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith(name_prefix)
        and (d / "checkpoints" / "loss_history.json").exists()
    ]
    return sorted(matches)[-1] if matches else None


def trim_at_early_stop(losses: list[float], patience: int | None, min_delta: float = 0.0) -> int:
    if patience is None or patience <= 0 or not losses:
        return len(losses)

    best = losses[0]
    no_improve = 0

    for i in range(1, len(losses)):
        current = losses[i]
        if current < best - min_delta:
            best = current
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                return i + 1

    return len(losses)


def trimmed_xy(
    x: list[float],
    y: list[float],
    patience: int | None,
    min_delta: float = 0.0,
) -> tuple[list[float], list[float], int]:
    max_len = min(len(x), len(y))
    x_cut = x[:max_len]
    y_cut = y[:max_len]
    end_idx = trim_at_early_stop(y_cut, patience, min_delta)
    return x_cut[:end_idx], y_cut[:end_idx], end_idx


def plot_loss_curves(
    adam: dict,
    muon: dict,
    output_dir: Path,
    early_stop_patience: int | None,
    min_delta: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    adam_h = adam["history"]
    muon_h = muon["history"]

    adam_train_step_x, adam_train_step_y, adam_train_step_n = trimmed_xy(
        list(range(1, len(adam_h.get("train_loss_step", [])) + 1)),
        adam_h.get("train_loss_step", []),
        early_stop_patience,
        min_delta,
    )
    muon_train_step_x, muon_train_step_y, muon_train_step_n = trimmed_xy(
        list(range(1, len(muon_h.get("train_loss_step", [])) + 1)),
        muon_h.get("train_loss_step", []),
        early_stop_patience,
        min_delta,
    )

    adam_train_time_x, adam_train_time_y, adam_train_time_n = trimmed_xy(
        adam_h.get("train_step_times_seconds", []),
        adam_h.get("train_loss_step", []),
        early_stop_patience,
        min_delta,
    )
    muon_train_time_x, muon_train_time_y, muon_train_time_n = trimmed_xy(
        muon_h.get("train_step_times_seconds", []),
        muon_h.get("train_loss_step", []),
        early_stop_patience,
        min_delta,
    )

    adam_val_step_x, adam_val_step_y, adam_val_step_n = trimmed_xy(
        adam_h.get("validation_global_steps", []),
        adam_h.get("validation_loss_step", []),
        early_stop_patience,
        min_delta,
    )
    muon_val_step_x, muon_val_step_y, muon_val_step_n = trimmed_xy(
        muon_h.get("validation_global_steps", []),
        muon_h.get("validation_loss_step", []),
        early_stop_patience,
        min_delta,
    )

    adam_val_time_x, adam_val_time_y, adam_val_time_n = trimmed_xy(
        adam_h.get("validation_times_seconds", []),
        adam_h.get("validation_loss_step", []),
        early_stop_patience,
        min_delta,
    )
    muon_val_time_x, muon_val_time_y, muon_val_time_n = trimmed_xy(
        muon_h.get("validation_times_seconds", []),
        muon_h.get("validation_loss_step", []),
        early_stop_patience,
        min_delta,
    )

    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Adam vs Muon: training and validation loss comparison", fontsize=14)

    ax = axs[0, 0]
    ax.plot(adam_train_time_x, adam_train_time_y, label="Adam", linewidth=1.5)
    ax.plot(muon_train_time_x, muon_train_time_y, label="Muon", linewidth=1.5)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[0, 1]
    ax.plot(adam_train_step_x, adam_train_step_y, label="Adam", linewidth=1.5)
    ax.plot(muon_train_step_x, muon_train_step_y, label="Muon", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Training loss")
    ax.set_title("Training loss vs step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[1, 0]
    ax.plot(adam_val_time_x, adam_val_time_y, "o-", label="Adam", linewidth=1.5, markersize=3)
    ax.plot(muon_val_time_x, muon_val_time_y, "s-", label="Muon", linewidth=1.5, markersize=3)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axs[1, 1]
    ax.plot(adam_val_step_x, adam_val_step_y, "o-", label="Adam", linewidth=1.5, markersize=3)
    ax.plot(muon_val_step_x, muon_val_step_y, "s-", label="Muon", linewidth=1.5, markersize=3)
    ax.set_xlabel("Global training step")
    ax.set_ylabel("Validation loss")
    ax.set_title("Validation loss vs step")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = output_dir / "loss_comparison_4panel.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")

    print()
    print("Used points after optional early-stop trimming:")
    print(f"  Adam train (step/time): {adam_train_step_n}/{adam_train_time_n}")
    print(f"  Muon train (step/time): {muon_train_step_n}/{muon_train_time_n}")
    print(f"  Adam val   (step/time): {adam_val_step_n}/{adam_val_time_n}")
    print(f"  Muon val   (step/time): {muon_val_step_n}/{muon_val_time_n}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Adam and Muon experiments with a 4-panel loss figure."
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help=(
            "Optional trim threshold in number of consecutive non-improving points. "
            "If omitted, no trimming is applied."
        ),
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0,
        help="Minimum loss decrease required to count as an improvement (default: 0.0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path("experiments/results")
    adam_dir = latest_experiment(results_dir, "adam_baseline")
    muon_dir = latest_experiment(results_dir, "muon_experiment")

    if adam_dir is None or muon_dir is None:
        print("Could not find both Adam and Muon experiments in experiments/results")
        return

    adam = load_experiment_data(adam_dir)
    muon = load_experiment_data(muon_dir)
    output_dir = results_dir / "comparison_plots"
    plot_loss_curves(
        adam,
        muon,
        output_dir,
        early_stop_patience=args.early_stop_patience,
        min_delta=args.min_delta,
    )

    adam_h = adam["history"]
    muon_h = muon["history"]
    adam_val = adam_h.get("validation_loss_step", [])
    muon_val = muon_h.get("validation_loss_step", [])
    adam_steps = adam_h.get("train_step_times_seconds", [])
    muon_steps = muon_h.get("train_step_times_seconds", [])

    print()
    print(f"Adam experiment : {adam_dir.name}")
    print(f"Muon experiment : {muon_dir.name}")
    if args.early_stop_patience is not None:
        print(f"Early-stop trimming: enabled (patience={args.early_stop_patience}, min_delta={args.min_delta})")
    else:
        print("Early-stop trimming: disabled")
    print()
    print(f"Adam val events : {len(adam_val)}, final val loss: {adam_val[-1]:.4f}" if adam_val else "Adam: no val data")
    print(f"Muon val events : {len(muon_val)}, final val loss: {muon_val[-1]:.4f}" if muon_val else "Muon: no val data")
    print()

    if adam_steps and muon_steps:
        def mean_step_dur(t: list[float]) -> float:
            durs = [t[0]] + [t[i] - t[i - 1] for i in range(1, len(t))]
            return sum(durs) / len(durs)
        adam_msd = mean_step_dur(adam_steps)
        muon_msd = mean_step_dur(muon_steps)
        print(f"Adam mean step time : {adam_msd:.3f}s ({len(adam_steps)} steps, total {adam_steps[-1]:.0f}s)")
        print(f"Muon mean step time : {muon_msd:.3f}s ({len(muon_steps)} steps, total {muon_steps[-1]:.0f}s)")
        print(f"Muon overhead vs Adam: {(muon_msd / adam_msd - 1) * 100:+.1f}%")


if __name__ == "__main__":
    main()
