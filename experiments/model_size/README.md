# SimBA Model-Size Benchmark

Measures throughput, GPU memory, and parameter count across a 3×3 grid of transformer
architectures (n_layers × d_model) on a single A100 GPU.

## Hardware

- **Cluster**: accelgor, partition `accelgor` (UGent HPC)
- **GPU**: 1× NVIDIA A100-SXM4-80GB per job
- **CPUs**: 8 per job, 5 GB RAM per CPU

## Methodology

Each job runs on 1 GPU with:
- **Effective batch size**: 2048 (held constant via gradient accumulation when the
  micro-batch does not fit at full size)
- **Warmup**: 50 steps (excluded from timing)
- **Measure**: 200 steps (timed)
- **Throughput**: `(steps_measured × effective_batch_size) / wall_time`
- **Memory**: `torch.cuda.max_memory_allocated()` peak over all measured steps

Auto-detection of micro-batch: binary search from `effective_batch_size` downward
to find the largest power-of-two that fits in GPU memory; gradient accumulation is
set accordingly.

**Architecture grid**:

| n_layers | d_model | embeddings_dim | n_heads |
|----------|---------|----------------|---------|
| 2        | 128     | 256            | 8       |
| 2        | 256     | 512            | 8       |
| 2        | 512     | 1024           | 8       |
| 5        | 128     | 256            | 8       |
| 5        | 256     | 512            | 8       |
| 5        | 512     | 1024           | 8       |
| 8        | 128     | 256            | 8       |
| 8        | 256     | 512            | 8       |
| 8        | 512     | 1024           | 8       |

## Reproducing the experiment

All jobs were submitted manually from the repo root:

```bash
for L in 2 5 8; do
  for D in 128 256 512; do
    sbatch --clusters=accelgor --nodes=1 --ntasks-per-node=1 --gres=gpu:1 \
      --export=N_LAYERS=$L,D_MODEL=$D \
      tools/slurm/model_size_experiment.slurm.sh
  done
done
```

Results are written to `experiments/model_size/results/layers{N}_dmodel{D}_A100_results.json`.

After all 9 jobs complete, regenerate the summary table and plots:

```bash
source .venv/bin/activate
python tools/aggregate_model_size_results.py
```

This produces:
- `model_size_results.csv` — full results table
- `model_size_results.png` — heatmaps: walltime, peak GPU memory, parameter count
- `model_size_throughput.png` — throughput vs parameter count scatter by depth

## Results

| size   | n_layers | d_model | emb_dim | n_params (M) | walltime (s) | throughput (samp/s) | GPU mem (GB) | GPU util (%) | CPU RAM (GB) |
|--------|----------|---------|---------|--------------|--------------|----------------------|--------------|--------------|--------------|
| small  | 2        | 128     | 256     | 0.79         | 45.40        | 9021.1               | 13.33        | 54.7         | 1.963        |
| small  | 2        | 256     | 512     | 2.11         | 79.67        | 5141.0               | 17.97        | 84.8         | 1.984        |
| medium | 2        | 512     | 1024    | 6.31         | 169.07       | 2422.6               | 27.30        | 96.6         | 2.018        |
| small  | 5        | 128     | 256     | 1.78         | 102.35       | 4002.0               | 30.94        | 92.3         | 1.999        |
| medium | 5        | 256     | 512     | 4.48         | 177.11       | 2312.7               | 41.62        | 96.7         | 2.012        |
| large  | 5        | 512     | 1024    | 12.62        | 401.61       | 1019.9               | 63.10        | 98.8         | 2.038        |
| medium | 8        | 128     | 256     | 2.77         | 160.18       | 2557.1               | 48.55        | 95.4         | 2.013        |
| medium | 8        | 256     | 512     | 6.85         | 279.92       | 1463.3               | 65.27        | 97.2         | 2.020        |
| large  | 8        | 512     | 1024    | 18.93        | 318.92       | 1284.3               | 83.90        | 98.1         | 2.004        |

### Key observations

- **Throughput vs size**: ranges from 9021 samples/s (smallest: 0.79M params) down to
  1020 samples/s (largest intra-node: 12.62M params, 5 layers × d512).  Throughput
  drops roughly 9× across the grid.
- **GPU memory**: scales from 13.3 GB (2L×d128) to 83.9 GB (8L×d512), the latter using
  ~105% of the 80 GB limit (auto-batch-finding stepped down to a smaller micro-batch via
  gradient accumulation in this case).
- **GPU utilisation**: the smallest model (2L×d128) only reaches 55% utilisation —
  compute kernels finish faster than data can be loaded.  All medium/large configs reach
  92–99%, confirming the GPU is compute-bound there.
- **CPU RAM**: flat at ~2 GB regardless of model size — the dataset, not the model,
  dominates host memory.
- **Current SimBA default** (5L×d256): 4.48M params, 2313 samples/s, 41.6 GB GPU memory,
  97% utilisation — a well-balanced operating point on this hardware.
