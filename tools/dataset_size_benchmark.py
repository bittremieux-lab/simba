#!/usr/bin/env python
"""Dataset size (samples per epoch) runtime benchmark for SimBA.

Runs exactly ceil(n_samples / batch_size) training steps on 1 GPU and records
the measured wall time. Since runtime scales linearly with steps, this directly
shows how training time scales with dataset/epoch size.

Usage:
    python tools/dataset_size_benchmark.py --n-samples 1000000
"""

import argparse
import json
import math
import time
from pathlib import Path

import psutil
import torch
import lightning.pytorch as pl


def parse_args():
    p = argparse.ArgumentParser(description="SimBA dataset-size scalability benchmark")
    p.add_argument("--preprocessing-dir",
                   default="./preprocessed_massspecgym_22k_speedup")
    p.add_argument("--n-samples", type=int, required=True,
                   help="Number of samples to process (= 1 epoch of this size)")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--warmup-steps", type=int, default=50,
                   help="Warmup steps before timing starts")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output-dir",
                   default="./experiments/dataset_size/results")
    p.add_argument("--gpu-type", type=str, default="A100")
    return p.parse_args()


def load_config(preprocessing_dir, batch_size, num_workers):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    repo_root = Path(__file__).resolve().parent.parent
    config_dir = repo_root / "simba" / "configs"

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.preprocessing_dir={preprocessing_dir}",
                f"paths.preprocessing_dir_train={preprocessing_dir}",
                f"training.batch_size={batch_size}",
                "hardware.accelerator=gpu",
                "hardware.devices=1",
                f"hardware.num_workers={num_workers}",
                "training.epochs=9999",
            ],
        )
    return cfg


class StepTimerCallback(pl.Callback):
    def __init__(self, warmup_steps: int, measure_steps: int, batch_size: int):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.batch_size = batch_size
        self._step_times: list[float] = []
        self._step_start = None
        self._max_gpu_mem_gb: float = 0.0
        self._max_cpu_ram_gb: float = 0.0
        self._proc = psutil.Process()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._step_start

        if torch.cuda.is_available():
            self._max_gpu_mem_gb = max(self._max_gpu_mem_gb,
                                       torch.cuda.max_memory_allocated() / 1e9)
        self._max_cpu_ram_gb = max(self._max_cpu_ram_gb,
                                   self._proc.memory_info().rss / 1e9)

        step = trainer.global_step
        if step > self.warmup_steps:
            self._step_times.append(elapsed)

        if len(self._step_times) >= self.measure_steps:
            trainer.should_stop = True

    def get_results(self):
        n = len(self._step_times)
        total_time = sum(self._step_times)
        samples = n * self.batch_size
        throughput = samples / total_time if total_time > 0 else 0.0
        return {
            "steps_measured": n,
            "wall_time_s": round(total_time, 4),
            "samples_processed": samples,
            "throughput_samples_per_s": round(throughput, 2),
            "max_gpu_mem_gb": round(self._max_gpu_mem_gb, 3),
            "max_cpu_ram_gb": round(self._max_cpu_ram_gb, 3),
        }


def main():
    args = parse_args()
    measure_steps = math.ceil(args.n_samples / args.batch_size)

    print("=" * 60)
    print("SimBA Dataset-Size Benchmark")
    print(f"  n_samples      : {args.n_samples:,}")
    print(f"  batch_size     : {args.batch_size}")
    print(f"  warmup steps   : {args.warmup_steps}")
    print(f"  measure steps  : {measure_steps}")
    print(f"  dataset        : {args.preprocessing_dir}")
    print("=" * 60)

    cfg = load_config(args.preprocessing_dir, args.batch_size, args.num_workers)

    from simba.workflows.training import load_dataset, prepare_data, setup_model
    from torch.utils.data import DataLoader, RandomSampler
    import numpy as np

    print("Loading dataset ...")
    molecule_pairs_train, molecule_pairs_val, _, _ = load_dataset(cfg)
    (dataset_train, _, _, _, _, _) = prepare_data(
        molecule_pairs_train, molecule_pairs_val, None, None, cfg
    )

    total_steps = args.warmup_steps + measure_steps + 20
    sampler = RandomSampler(
        dataset_train, replacement=True,
        num_samples=total_steps * args.batch_size
    )
    dataloader = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    print("Building model ...")
    weights_mces = np.ones(cfg.model.tasks.edit_distance.n_classes)
    model = setup_model(cfg, weights_mces)

    timer_cb = StepTimerCallback(
        warmup_steps=args.warmup_steps,
        measure_steps=measure_steps,
        batch_size=args.batch_size,
    )

    trainer = pl.Trainer(
        max_steps=total_steps,
        limit_val_batches=0.0,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=max(1, measure_steps // 10),
        callbacks=[timer_cb],
    )

    print("Starting benchmark run ...")
    trainer.fit(model, dataloader)

    results = timer_cb.get_results()
    t = int(results["wall_time_s"])
    wall_hms = f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"

    output = {
        "n_samples": args.n_samples,
        "batch_size": args.batch_size,
        "gpu_type": args.gpu_type,
        "warmup_steps": args.warmup_steps,
        "measure_steps": measure_steps,
        **results,
        "wall_time_hms": wall_hms,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_m = args.n_samples // 1_000_000
    out_path = output_dir / f"{n_m}M_samples_{args.gpu_type}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results (n_samples={args.n_samples:,}):")
    print(f"  Steps measured  : {results['steps_measured']}")
    print(f"  Wall time       : {wall_hms} ({results['wall_time_s']:.1f} s)")
    print(f"  Throughput      : {results['throughput_samples_per_s']:.1f} samples/s")
    print(f"  Peak GPU memory : {results['max_gpu_mem_gb']:.2f} GB")
    print(f"  Peak CPU RAM    : {results['max_cpu_ram_gb']:.2f} GB")
    print(f"  Saved to        : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
