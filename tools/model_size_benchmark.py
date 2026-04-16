#!/usr/bin/env python
"""Model size (depth × width) runtime and memory benchmark for SimBA.

Runs a fixed number of training steps on 1 GPU and records walltime,
throughput, GPU memory, and parameter count for a given (n_layers, d_model)
configuration.
"""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import psutil
import torch
import lightning.pytorch as pl


class GpuUtilMonitor:
    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self._utilizations: list = []
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
                vals = [float(v) for v in out.splitlines()
                        if v.strip().replace(".", "", 1).isdigit()]
                if vals:
                    self._utilizations.append(sum(vals) / len(vals))
            except Exception:
                pass
            self._stop_event.wait(self.interval)


class ThroughputCallback(pl.Callback):
    def __init__(self, warmup_steps: int, measure_steps: int,
                 batch_size: int):
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.batch_size = batch_size
        self._step_times: list = []
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
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            self._max_gpu_mem_gb = max(self._max_gpu_mem_gb, mem_gb)

        cpu_gb = self._proc.memory_info().rss / 1e9
        self._max_cpu_ram_gb = max(self._max_cpu_ram_gb, cpu_gb)

        step = trainer.global_step
        if step > self.warmup_steps:
            self._step_times.append(elapsed)

        if len(self._step_times) >= self.measure_steps:
            trainer.should_stop = True

    def get_results(self) -> dict:
        n = len(self._step_times)
        total_time = sum(self._step_times)
        total_samples = n * self.batch_size
        throughput = total_samples / total_time if total_time > 0 else 0.0
        return {
            "steps_measured": n,
            "time_s": round(total_time, 4),
            "samples": total_samples,
            "throughput_samples_per_s": round(throughput, 2),
            "max_gpu_mem_gb": round(self._max_gpu_mem_gb, 3),
            "max_cpu_ram_gb": round(self._max_cpu_ram_gb, 3),
        }


def parse_args():
    p = argparse.ArgumentParser(description="SimBA model-size benchmark")
    p.add_argument("--preprocessing-dir",
                   default="./preprocessed_massspecgym_22k_speedup")
    p.add_argument("--effective-batch-size", type=int, default=2048,
                   help="Effective batch size (held constant via grad accumulation)")
    p.add_argument("--micro-batch-size", type=int, default=None,
                   help="Override micro-batch size. If None, auto-determined by binary search.")
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--measure-steps", type=int, default=200)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--output-dir",
                   default="./experiments/model_size/results")
    p.add_argument("--gpu-type", type=str, default="A100")
    # Model architecture
    p.add_argument("--n-layers", type=int, default=5,
                   help="Number of transformer layers")
    p.add_argument("--d-model", type=int, default=256,
                   help="Transformer hidden dimension (d_model)")
    p.add_argument("--embeddings-dim", type=int, default=None,
                   help="Output embedding dimension (defaults to 2 × d_model)")
    p.add_argument("--n-heads", type=int, default=None,
                   help="Number of attention heads (auto: largest divisor of d_model ≤ 8)")
    return p.parse_args()


def find_max_micro_batch(model, dataset, effective_batch: int,
                         num_workers: int, cfg) -> int:
    """Binary-search the largest micro-batch that fits in GPU memory.

    Tries descending powers of 2 from effective_batch down to 1.
    Returns the largest successful size.
    """
    from torch.utils.data import DataLoader, RandomSampler

    device = torch.device("cuda")
    model = model.to(device)
    model.train()

    candidates = []
    b = effective_batch
    while b >= 1:
        candidates.append(b)
        b = b // 2

    best = 1
    for candidate in candidates:
        try:
            torch.cuda.empty_cache()
            sampler = RandomSampler(dataset, replacement=True,
                                    num_samples=candidate * 3)
            dl = DataLoader(dataset, batch_size=candidate, sampler=sampler,
                            num_workers=0, pin_memory=True)
            batch = next(iter(dl))
            batch = _batch_to_device(batch, device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            optimizer.zero_grad()
            out = model.training_step(batch, 0)
            loss = out if isinstance(out, torch.Tensor) else out.get("loss", next(iter(out.values())))
            loss.backward()
            optimizer.zero_grad(set_to_none=True)
            best = candidate
            break  # largest that fits found
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                continue
            raise

    torch.cuda.empty_cache()
    model = model.cpu()
    print(f"  Auto micro-batch: {best} "
          f"(accumulation={effective_batch // best}x to reach effective={effective_batch})")
    return best


def _batch_to_device(batch, device):
    """Recursively move a batch (dict/list/tensor) to device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(_batch_to_device(x, device) for x in batch)
    return batch


def _auto_n_heads(d_model: int, requested: int = None) -> int:
    if requested is not None:
        return requested
    # Largest power-of-two that divides d_model, capped at 8
    for h in [8, 4, 2, 1]:
        if d_model % h == 0:
            return h
    return 1


def load_config(preprocessing_dir: str, batch_size: int, num_workers: int,
                n_layers: int, d_model: int, embeddings_dim: int,
                n_heads: int):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    config_dir = repo_root / "simba" / "configs"

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="config",
            overrides=[
                f"paths.preprocessing_dir={preprocessing_dir}",
                f"paths.preprocessing_dir_train={preprocessing_dir}",
                f"training.batch_size={batch_size}",
                f"hardware.accelerator=gpu",
                f"hardware.devices=1",
                f"hardware.num_workers={num_workers}",
                "training.epochs=9999",
                f"model.transformer.n_layers={n_layers}",
                f"model.transformer.d_model={d_model}",
                f"model.transformer.n_heads={n_heads}",
                f"model.embeddings.dim={embeddings_dim}",
                f"model.n_layers={n_layers}",
                f"model.d_model={d_model}",
            ],
        )
    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    embeddings_dim = args.embeddings_dim if args.embeddings_dim else 2 * args.d_model
    n_heads = _auto_n_heads(args.d_model, args.n_heads)

    print("=" * 60)
    print("SimBA Model-Size Benchmark")
    print(f"  n_layers        : {args.n_layers}")
    print(f"  d_model         : {args.d_model}")
    print(f"  embeddings_dim  : {embeddings_dim}")
    print(f"  n_heads         : {n_heads}")
    print(f"  effective batch : {args.effective_batch_size}")
    print(f"  warmup steps    : {args.warmup_steps}")
    print(f"  measure steps   : {args.measure_steps}")
    print(f"  dataset         : {args.preprocessing_dir}")
    print("=" * 60)

    cfg = load_config(
        preprocessing_dir=args.preprocessing_dir,
        batch_size=args.effective_batch_size,
        num_workers=args.num_workers,
        n_layers=args.n_layers,
        d_model=args.d_model,
        embeddings_dim=embeddings_dim,
        n_heads=n_heads,
    )

    from simba.workflows.training import load_dataset, prepare_data, setup_model
    from torch.utils.data import DataLoader, RandomSampler
    import numpy as np

    print("Loading dataset …")
    molecule_pairs_train, molecule_pairs_val, _, _ = load_dataset(cfg)
    (dataset_train, train_sampler, _, _, weights_ed, bins_ed) = prepare_data(
        molecule_pairs_train, molecule_pairs_val, None, None, cfg
    )

    print("Building model …")
    weights_mces = np.ones(cfg.model.tasks.edit_distance.n_classes)
    model = setup_model(cfg, weights_mces)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Determine micro-batch size
    if args.micro_batch_size is not None:
        micro_batch = min(args.micro_batch_size, args.effective_batch_size)
        print(f"  Micro-batch: {micro_batch} (manual override)")
    else:
        print("  Finding max micro-batch that fits in GPU memory …")
        micro_batch = find_max_micro_batch(
            model, dataset_train, args.effective_batch_size, args.num_workers, cfg
        )

    accumulation = max(1, args.effective_batch_size // micro_batch)
    effective = micro_batch * accumulation
    print(f"  Micro-batch={micro_batch} | Accumulation={accumulation}x | Effective={effective}")

    # Rebuild config/dataloader with actual micro-batch size
    cfg = load_config(
        preprocessing_dir=args.preprocessing_dir,
        batch_size=micro_batch,
        num_workers=args.num_workers,
        n_layers=args.n_layers,
        d_model=args.d_model,
        embeddings_dim=embeddings_dim,
        n_heads=n_heads,
    )

    total_steps = (args.warmup_steps + args.measure_steps + 20)
    big_sampler = RandomSampler(
        dataset_train,
        replacement=True,
        num_samples=total_steps * accumulation * micro_batch * 2,
    )
    dataloader = DataLoader(
        dataset_train,
        batch_size=micro_batch,
        shuffle=False,
        sampler=big_sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=True,
    )

    # Callback counts samples per optimizer step (= effective batch)
    throughput_cb = ThroughputCallback(
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        batch_size=effective,
    )

    gpu_monitor = GpuUtilMonitor(interval=2.0)
    gpu_monitor.start()

    trainer = pl.Trainer(
        max_steps=total_steps,
        limit_val_batches=0.0,
        accelerator="gpu",
        devices=1,
        strategy="auto",
        accumulate_grad_batches=accumulation,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=True,
        log_every_n_steps=50,
        callbacks=[throughput_cb],
    )

    print("Starting benchmark run …")
    trainer.fit(model, dataloader)

    gpu_monitor.stop()
    mean_util = gpu_monitor.mean_utilization()

    results = throughput_cb.get_results()

    output = {
        "n_layers": args.n_layers,
        "d_model": args.d_model,
        "embeddings_dim": embeddings_dim,
        "n_heads": n_heads,
        "n_params": n_params,
        "gpu_type": args.gpu_type,
        "effective_batch_size": effective,
        "micro_batch_size": micro_batch,
        "accumulate_grad_batches": accumulation,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        **results,
        "gpu_util_pct": round(mean_util, 1),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"layers{args.n_layers}_dmodel{args.d_model}_{args.gpu_type}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Results (n_layers={args.n_layers}, d_model={args.d_model}):")
    print(f"  Parameters      : {n_params:,}")
    print(f"  Micro-batch     : {micro_batch}  (accumulation={accumulation}x, effective={effective})")
    print(f"  Steps measured  : {results['steps_measured']}")
    print(f"  Wall-clock time : {results['time_s']:.2f} s")
    print(f"  Throughput      : {results['throughput_samples_per_s']:.1f} samples/s")
    print(f"  Peak GPU memory : {results['max_gpu_mem_gb']:.2f} GB")
    print(f"  Peak CPU RAM    : {results['max_cpu_ram_gb']:.2f} GB")
    print(f"  Mean GPU util   : {mean_util:.1f}%")
    print(f"  Saved to        : {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
