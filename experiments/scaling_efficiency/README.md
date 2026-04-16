# SimBA Distributed Training Scaling Efficiency

Measures DDP throughput scaling across 1–16 A100 GPUs on the UGent HPC accelgor cluster.

## Hardware

- **Cluster**: accelgor (UGent HPC)
- **GPUs**: NVIDIA A100-SXM4-80GB (4 per node)
- **Nodes**: 9 total (node3900–node3908), InfiniBand interconnect
- **CPUs**: 48 per node, 432 GB RAM per node

## Methodology

Each benchmark run:
1. Loads the preprocessed MassSpecGym dataset
2. Uses a uniform `RandomSampler`
3. Runs `WARMUP_STEPS` steps (excluded from timing), then times `MEASURE_STEPS` steps
4. Reports throughput as `(steps_measured × batch_size_per_gpu × total_gpus) / wall_time`

**Key design choice — constant total data volume**: steps are scaled inversely with GPU count so that all configurations process approximately the same number of samples (~3.07M). This means wall-clock time is the meaningful comparison metric: perfect linear scaling → identical wall-clock time.

| GPUs | Warmup steps | Measure steps | Total samples |
|------|-------------|---------------|---------------|
| 1    | 150         | 1500          | 3,072,000     |
| 2    | 75          | 750           | 3,072,000     |
| 4    | 50          | 375           | 3,072,000     |
| 8    | 50          | 187           | 3,063,808     |
| 16   | 50          | 93            | 3,047,424     |

**Settings**: batch size = 2048 per GPU, 8 DataLoader workers per GPU process.

## Reproducing the experiment

All GPU jobs were submitted to accelgor using SLURM script at `tools/slurm/scaling_experiment.slurm.sh`.

Run from the repo root:

```bash
# 1 GPU (1 node)
sbatch --clusters=accelgor --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
    tools/slurm/scaling_experiment.slurm.sh

# 2 GPUs (1 node)
sbatch --clusters=accelgor --nodes=1 --ntasks-per-node=2 --gres=gpu:2 \
    tools/slurm/scaling_experiment.slurm.sh

# 4 GPUs (1 node, full node)
sbatch --clusters=accelgor --nodes=1 --ntasks-per-node=4 --gres=gpu:4 \
    tools/slurm/scaling_experiment.slurm.sh

# 8 GPUs (2 nodes)
sbatch --clusters=accelgor --nodes=2 --ntasks-per-node=4 --gres=gpu:4 \
    tools/slurm/scaling_experiment.slurm.sh

# 16 GPUs (4 nodes)
sbatch --clusters=accelgor --nodes=4 --ntasks-per-node=4 --gres=gpu:4 \
    tools/slurm/scaling_experiment.slurm.sh
```

CPU baseline jobs are submitted to `doduo` via `tools/slurm/cpu_scaling_experiment.slurm.sh`.
Sweep core counts 1, 2, 4, 8, 16, 32 on a single node:

```bash
for N in 1 2 4 8 16 32; do
    sbatch --clusters=doduo --cpus-per-task=$N \
        tools/slurm/cpu_scaling_experiment.slurm.sh
done
```

Results are written to `experiments/scaling_efficiency/results/<N>gpu_<M>node_A100_results.json`
and `experiments/scaling_efficiency/results/<N>cpu_1node_doduo_results.json`.

After all jobs complete, regenerate the summary tables and plots:

```bash
source .venv/bin/activate
python tools/aggregate_scaling_results.py
```

This produces:
- `scaling_results.csv` / `scaling_results.png` — GPU strong-scaling (log-scale x-axis)
- `cpu_scaling_results.csv` / `cpu_scaling_results.png` — CPU strong-scaling + GPU-vs-CPU speedup bar

## Results

### GPU strong-scaling — accelgor (partition: `accelgor`)

| GPUs | nodes | time (s) | samples | throughput (samples/s) | speedup | efficiency | max gpu mem (GB) | max cpu ram (GB) | gpu_util (%) |
|------|-------|----------|---------|------------------------|---------|------------|-----------------|-----------------|--------------|
| 1    | 1     | 1321.98  | 3072000 | 2323.8                 | 1.000   | 1.000      | 41.617          | 2.014           | 99.1         |
| 2    | 1     | 664.08   | 3072000 | 4626.0                 | 1.991   | 0.995      | 41.636          | 2.203           | 97.7         |
| 4    | 1     | 325.56   | 3072000 | 9436.1                 | 4.061   | 1.015      | 41.636          | 2.203           | 92.6         |
| 8    | 2     | 163.50   | 3063808 | 18739.2                | 8.064   | 1.008      | 41.636          | 2.345           | 93.6         |
| 16   | 4     | 81.86    | 3047424 | 37227.4                | 16.020  | 1.001      | 41.636          | 2.309           | 88.7         |

Plots (speedup and efficiency panels use log₂ x-axis as required): `scaling_results.png`

### CPU baseline — doduo (partition: `doduo`)

Benchmarks run on a single node of the doduo CPU cluster (UGent HPC). Training uses PyTorch with OpenMP threading only (no MPI/DDP — a single process). Memory requirement: ~12.5 GB per run (fixed regardless of core count).

| cores | nodes | time (s) | samples | throughput (samples/s) | speedup | efficiency | max cpu ram (GB) |
|-------|-------|----------|---------|------------------------|---------|------------|-----------------|
| 1     | 1     | 798.79   | 5120    | 6.4                    | 1.000   | 1.000      | 12.514           |
| 2     | 1     | 437.58   | 5120    | 11.7                   | 1.825   | 0.913      | 12.422           |
| 4     | 1     | 391.71   | 5120    | 13.1                   | 2.039   | 0.510      | 12.650           |
| 8     | 1     | 242.52   | 5120    | 21.1                   | 3.293   | 0.412      | 12.694           |
| 16    | 1     | 216.97   | 5120    | 23.6                   | 3.682   | 0.230      | 12.676           |
| 32    | 1     | 170.74   | 5120    | 30.0                   | 4.679   | 0.146      | 12.674           |

CPU scaling efficiency drops below 70% after 2 cores, indicating the model is memory-bandwidth bound for CPU threading. The best CPU throughput (32 cores, 30 samples/s) is **~77× slower than a single A100 GPU** (2324 samples/s).

Plots: `cpu_scaling_results.png` (CPU speedup, CPU efficiency with 70% target line, GPU-vs-CPU speedup bar).
