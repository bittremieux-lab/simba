#!/usr/bin/env python
"""Distributed training scaling efficiency benchmark for SimBA."""

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import psutil

import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies import DDPStrategy

class GpuUtilMonitor:
    """Polls nvidia-smi every *interval* seconds in a daemon thread."""

    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._utilizations: list[float] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5)

    def mean_utilization(self) -> float:
        if not self._utilizations:
            return 0.0
        return sum(self._utilizations) / len(self._utilizations)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    timeout=3,
                ).decode().strip()
                vals = [float(v) for v in out.splitlines() if v.strip().isdigit() or v.strip().replace(".", "", 1).isdigit()]
                if vals:
                    self._utilizations.append(sum(vals) / len(vals))
            except Exception:
                pass
            self._stop_event.wait(self.interval)


class ThroughputCallback(pl.Callback):
    """Measures steady-state step throughput, ignoring warmup.

    Timing is done on rank 0 only (global_zero), using CUDA synchronisation
    for accurate GPU-inclusive measurement.  ``trainer.should_stop`` is set
    once *measure_steps* have been collected; PyTorch Lightning broadcasts
    this to all ranks.
    """

    def __init__(
        self,
        warmup_steps: int,
        measure_steps: int,
        batch_size_per_gpu: int,
        num_gpus: int,
    ):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.batch_size_per_gpu = batch_size_per_gpu
        self.num_gpus = num_gpus

        self._step_times: list[float] = []
        self._step_start: float | None = None
        self._max_mem_gb: float = 0.0
        self._max_cpu_ram_gb: float = 0.0
        self._proc = psutil.Process()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not trainer.is_global_zero:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._step_start = time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._step_start  # type: ignore[operator]

        # Track peak GPU memory
        if torch.cuda.is_available():
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            self._max_mem_gb = max(self._max_mem_gb, mem_gb)

        # Track peak CPU RAM (RSS of this process)
        cpu_ram_gb = self._proc.memory_info().rss / 1e9
        self._max_cpu_ram_gb = max(self._max_cpu_ram_gb, cpu_ram_gb)

        # global_step increments *after* this hook in newer PL, check timing:
        # warmup_steps=100 → ignore steps 1..100, measure from step 101 onward
        step = trainer.global_step  # 1-based after first step
        if step > self.warmup_steps:
            self._step_times.append(elapsed)

        if len(self._step_times) >= self.measure_steps:
            trainer.should_stop = True

    def get_results(self) -> dict:
        n = len(self._step_times)
        total_time = sum(self._step_times)
        total_samples = n * self.batch_size_per_gpu * self.num_gpus
        throughput = total_samples / total_time if total_time > 0 else 0.0
        return {
            "steps_measured": n,
            "time_s": round(total_time, 4),
            "samples": total_samples,
            "throughput_samples_per_s": round(throughput, 2),
            "max_gpu_mem_gb": round(self._max_mem_gb, 3),
            "max_cpu_ram_gb": round(self._max_cpu_ram_gb, 3),
        }


def parse_args():
    p = argparse.ArgumentParser(description="SimBA DDP scaling benchmark")
    p.add_argument(
        "--preprocessing-dir",
        default="./preprocessed_massspecgym_22k_speedup",
        help="Path to preprocessed dataset directory",
    )
    p.add_argument("--batch-size", type=int, default=2048,
                   help="Batch size *per GPU*")
    p.add_argument("--warmup-steps", type=int, default=150,
                   help="Steps to skip before measuring")
    p.add_argument("--measure-steps", type=int, default=1500,
                   help="Steps to measure for throughput")
    p.add_argument("--num-workers", type=int, default=8,
                   help="DataLoader workers per GPU process")
    p.add_argument(
        "--output-dir",
        default="./experiments/scaling_efficiency/results",
        help="Directory to write JSON result files",
    )
    # GPU/node overrides (auto-detected from SLURM env when available)
    p.add_argument("--gpus-per-node", type=int, default=None,
                   help="GPUs per node (defaults to SLURM_NTASKS_PER_NODE)")
    p.add_argument("--num-nodes", type=int, default=None,
                   help="Number of nodes (defaults to SLURM_NNODES)")
    p.add_argument("--gpu-type", type=str, default="gpu",
                   help="GPU type label for output filename, e.g. A100, H100")
    return p.parse_args()


def load_config(preprocessing_dir: str, batch_size: int, num_workers: int,
                gpus_per_node: int, num_nodes: int):
    """Load Hydra config via compose API."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    script_dir = Path(__file__).resolve().parent          # tools/
    repo_root = script_dir.parent                          # simba/
    config_dir = repo_root / "simba" / "configs"

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(config_dir), version_base=None
    ):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.preprocessing_dir={preprocessing_dir}",
                f"paths.preprocessing_dir_train={preprocessing_dir}",
                f"training.batch_size={batch_size}",
                f"hardware.accelerator=gpu",
                f"hardware.devices={gpus_per_node}",
                f"hardware.num_workers={num_workers}",
                "training.epochs=9999",     # stopped by max_steps
            ],
        )
    return cfg


def main():
    args = parse_args()

    num_nodes = args.num_nodes or int(os.environ.get("SLURM_NNODES", 1))
    gpus_per_node = args.gpus_per_node or int(
        os.environ.get("SLURM_NTASKS_PER_NODE", 1)
    )
    total_gpus = num_nodes * gpus_per_node

    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    global_rank = int(os.environ.get("SLURM_PROCID", 0))
    is_rank0 = (global_rank == 0)

    if is_rank0:
        print("=" * 60)
        print(f"SimBA scaling benchmark")
        print(f"  Nodes        : {num_nodes}")
        print(f"  GPUs / node  : {gpus_per_node}")
        print(f"  Total GPUs   : {total_gpus}")
        print(f"  Batch / GPU  : {args.batch_size}")
        print(f"  Warmup steps : {args.warmup_steps}")
        print(f"  Measure steps: {args.measure_steps}")
        print(f"  Dataset      : {args.preprocessing_dir}")
        print("=" * 60)

    cfg = load_config(
        preprocessing_dir=args.preprocessing_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        gpus_per_node=gpus_per_node,
        num_nodes=num_nodes,
    )

    from simba.workflows.training import (
        load_dataset,
        prepare_data,
        setup_model,
    )
    from torch.utils.data import DataLoader, RandomSampler
    from simba.core.data.weighted_sampling import CustomWeightedRandomSampler

    if is_rank0:
        print("Loading dataset …")
    molecule_pairs_train, molecule_pairs_val, _, _ = load_dataset(cfg)

    (
        dataset_train,
        train_sampler,
        _dataset_val,
        _val_sampler,
        weights_ed,
        bins_ed,
    ) = prepare_data(
        molecule_pairs_train,
        molecule_pairs_val,
        None,       # test not needed
        None,
        cfg,
    )

    total_steps = args.warmup_steps + args.measure_steps + 20
    big_num_samples = total_steps * args.batch_size * total_gpus * 2
    big_sampler = RandomSampler(
        dataset_train,
        replacement=True,
        num_samples=big_num_samples,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=big_sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    if is_rank0:
        print(f"Dataset: {len(dataset_train)} samples, {len(dataloader_train)} batches/epoch")
        print("Building model …")

    import numpy as np
    weights_mces = np.ones(cfg.model.tasks.edit_distance.n_classes)
    model = setup_model(cfg, weights_mces)

    throughput_cb = ThroughputCallback(
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        batch_size_per_gpu=args.batch_size,
        num_gpus=total_gpus,
    )

    gpu_monitor = None
    if is_rank0:
        gpu_monitor = GpuUtilMonitor(interval=2.0)
        gpu_monitor.start()

    if total_gpus > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = "auto"
    max_steps = args.warmup_steps + args.measure_steps + 20

    trainer = pl.Trainer(
        max_steps=max_steps,
        limit_val_batches=0.0,          # skip validation entirely
        accelerator="gpu",
        devices=gpus_per_node,
        num_nodes=num_nodes,
        strategy=strategy,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=is_rank0,
        log_every_n_steps=50,
        callbacks=[throughput_cb],
    )

    if is_rank0:
        print("Starting training run …")
    trainer.fit(model, dataloader_train)

    if gpu_monitor is not None:
        gpu_monitor.stop()
        mean_util = gpu_monitor.mean_utilization()
    else:
        mean_util = 0.0

    if not trainer.is_global_zero:
        return  # only rank 0 writes results

    results = throughput_cb.get_results()

    output = {
        "num_gpus": total_gpus,
        "num_nodes": num_nodes,
        "gpus_per_node": gpus_per_node,
        "gpu_type": args.gpu_type,
        "batch_size_per_gpu": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        **results,
        "gpu_util_pct": round(mean_util, 1),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{total_gpus}gpu_{num_nodes}node_{args.gpu_type}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results ({total_gpus} GPU{'s' if total_gpus > 1 else ''}):")
    print(f"  Steps measured  : {results['steps_measured']}")
    print(f"  Wall-clock time : {results['time_s']:.2f} s")
    print(f"  Samples         : {results['samples']}")
    print(f"  Throughput      : {results['throughput_samples_per_s']:.1f} samples/s")
    print(f"  Peak GPU memory : {results['max_gpu_mem_gb']:.2f} GB")
    print(f"  Peak CPU RAM    : {results['max_cpu_ram_gb']:.2f} GB")
    print(f"  Mean GPU util   : {mean_util:.1f}%")
    print(f"  Saved to        : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
