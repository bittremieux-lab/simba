# Quick Start

This guide will get you up and running with SIMBA.

## Overview

SIMBA provides a pretrained model trained on spectra from **MassSpecGym**. The model operates in positive ionization mode for protonated adducts.

A typical SIMBA workflow consists of:

1. **Computing Structural Similarities**: Predict edit distance and MCES between spectra
2. **Analog Discovery**: Find structurally similar molecules in a reference library
3. **Training Custom Models**: Train SIMBA on your own MS/MS data (optional)

## Computing Structural Similarities

Follow the [Run Inference Notebook](https://github.com/bittremieux-lab/simba/tree/main/notebooks/final_tutorials/run_inference.ipynb) for a comprehensive tutorial:

- **Runtime:** < 10 minutes (including model/data download)
- **Example data:** data folder
- **Supported format:** `.mgf`

### Performance

Using an Apple M3 Pro (36 GB RAM):

- **Embedding computation:** ~100,000 spectra in ~1 minute
- **Similarity computation:** 1 query vs. 100,000 spectra in ~10 seconds

SIMBA caches computed embeddings, significantly speeding repeated library searches.

## Analog Discovery

Perform analog discovery to find structurally similar molecules:

```bash
simba analog-discovery \
  --model-path /path/to/model.ckpt \
  --query-spectra /path/to/query.mgf \
  --reference-spectra /path/to/reference_library.mgf \
  --output-dir /path/to/output \
  analog_discovery.query_index=0 \
  analog_discovery.top_k=10 \
  analog_discovery.device=cpu \
  analog_discovery.compute_ground_truth=true
```

**CLI flags** (required):

- `--model-path`: Path to trained SIMBA model checkpoint (.ckpt file)
- `--query-spectra`: Path to query spectra file (.mgf format)
- `--reference-spectra`: Path to reference library spectra file (.mgf format)
- `--output-dir`: Directory where results will be saved

**Hydra overrides** (optional, with defaults):

- `analog_discovery.query_index=0`: Index of the query spectrum to analyze (default: null = all)
- `analog_discovery.top_k=10`: Number of top matches to return
- `analog_discovery.device=cpu`: Hardware device: `cpu` or `gpu`
- `analog_discovery.batch_size=32`: Batch size for processing
- `analog_discovery.cache_embeddings=true`: Cache embeddings for faster repeated searches
- `analog_discovery.compute_ground_truth=false`: Compute ground truth edit distance and MCES
- `analog_discovery.save_rankings=true`: Save complete ranking matrix to file

**Output:**

The command generates several files in the output directory:

- `results.json`: Summary of top matches with predictions and ground truth
- `matches.csv`: Detailed table of all matches
- `query_molecule.png`: Structure of the query molecule
- `match_N_molecule.png`: Structures of matched molecules
- `mirror_plot_match_N.png`: Mirror plots comparing query and matched spectra
- `rankings.npy`: Complete ranking matrix (if `--save-rankings` is used)

For interactive exploration, use the [Run Analog Discovery Notebook](https://github.com/bittremieux-lab/simba/tree/main/notebooks/final_tutorials/run_analog_discovery.ipynb).

## Training Custom Models

### Step 1: Preprocess Data

```bash
simba preprocess \
  paths.spectra_path=/path/to/your/spectra.mgf \
  paths.preprocessing_dir=/path/to/preprocessed_data \
  preprocessing.max_spectra_train=10000 \
  preprocessing.num_workers=60
```

Output: `mapping_unique_smiles.pkl` (default name, override with `paths.preprocessing_pickle_file=...`).

### Step 2: Train Model

```bash
simba train \
  paths.preprocessing_dir=/path/to/preprocessed_data \
  paths.checkpoint_dir=checkpoints/ \
  training.epochs=50 \
  hardware.accelerator=gpu \
  training.batch_size=64
```

### Step 3: Run Inference

```bash
simba inference \
  paths.checkpoint_dir=checkpoints/ \
  paths.preprocessing_dir=/path/to/preprocessed_data \
  inference.batch_size=128 \
  hardware.accelerator=gpu
```
