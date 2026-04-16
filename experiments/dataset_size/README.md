# Dataset-Size Scalability Experiment

## Goal

Demonstrate that SimBA training time scales **linearly with dataset (epoch) size**, and measure the per-sample throughput constant for compute-budget estimation.

## Setup

- **Cluster**: accelgor (UGent Tier-2)
- **GPU**: 1× NVIDIA A100-SXM4-80GB
- **Batch size**: 2048 samples/step
- **Warmup**: 50 steps (excluded from timing)
- **Measured steps**: ⌈n_samples / batch_size⌉ (so the full requested sample count is covered)

## Dataset sizes tested

| Dataset size | Steps | Time limit |
|---|---|---|
| 1 M | 489 | 00:15:00 |
| 2 M | 977 | 00:25:00 |
| 4 M | 1 954 | 00:50:00 |
| 8 M | 3 907 | 01:40:00 |
| 16 M | 7 813 | 03:15:00 |
| 32 M | 15 625 | 06:00:00 |

## Results

| Dataset size (M) | Steps | Walltime | Throughput (samples/s) | GPU memory (GB) |
|---|---|---|---|---|
| 1 | 489 | 00:07:12 | 2313 | 41.6 |
| 2 | 977 | 00:14:25 | 2312 | 41.6 |
| 4 | 1 954 | 00:28:50 | 2312 | 41.6 |
| 8 | 3 907 | 00:57:40 | 2312 | 41.6 |
| 16 | 7 813  | 01:55:18 | 2313 | 41.6 |
| 32 | 15 625 | 03:50:50 | 2310 | 41.6 |

**Mean throughput**: 2312 samples/s across 1 M – 32 M samples

GPU memory is constant across all sizes — the model footprint does not depend on epoch size.

## Key finding

Training time scales **perfectly linearly** with dataset size. Throughput is constant at ~2312 samples/s on a single A100. This allows straightforward compute-budget estimation:

```
walltime (s) = n_samples / 2312
```

## Reproducing

```bash
# Single run
python tools/dataset_size_benchmark.py --n-samples 4000000

# Full sweep (accelgor)
declare -A TIMES=([1000000]=00:15:00 [2000000]=00:25:00 [4000000]=00:50:00 \
                  [8000000]=01:40:00 [16000000]=03:15:00 [32000000]=06:00:00)
for N in 1000000 2000000 4000000 8000000 16000000 32000000; do
  sbatch --clusters=accelgor --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
    --time=${TIMES[$N]} --export=N_SAMPLES=$N \
    tools/slurm/dataset_size_experiment.slurm.sh
done

# Aggregate results
python tools/aggregate_dataset_size_results.py
```
