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

All jobs were submitted to accelgor using SLURM script at `tools/slurm/scaling_experiment.slurm.sh`.

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

Results are written to `experiments/scaling_efficiency/results/<N>gpu_<M>node_A100_results.json`.

After all jobs complete, regenerate the summary table and plots:

```bash
source .venv/bin/activate
python tools/aggregate_scaling_results.py
```

## Results

| GPUs | nodes | time (s) | samples | throughput (samples/s) | speedup | efficiency | max gpu mem (GB) | max cpu ram (GB) | gpu_util (%) |
|------|-------|----------|---------|------------------------|---------|------------|-----------------|-----------------|--------------|
| 1    | 1     | 1321.98  | 3072000 | 2323.8                 | 1.000   | 1.000      | 41.617          | 2.014           | 99.1         |
| 2    | 1     | 664.08   | 3072000 | 4626.0                 | 1.991   | 0.995      | 41.636          | 2.203           | 97.7         |
| 4    | 1     | 325.56   | 3072000 | 9436.1                 | 4.061   | 1.015      | 41.636          | 2.203           | 92.6         |
| 8    | 2     | 163.50   | 3063808 | 18739.2                | 8.064   | 1.008      | 41.636          | 2.345           | 93.6         |
| 16   | 4     | 81.86    | 3047424 | 37227.4                | 16.020  | 1.001      | 41.636          | 2.309           | 88.7         |
