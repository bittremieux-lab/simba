#!/bin/bash
#SBATCH -J simba_mces_calib_rerun_014_2
#SBATCH -p zen4_h200
#SBATCH --account=zen4-h200-2026_053-1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=nvidia_h200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH -o /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots/%x_%j.out
#SBATCH -e /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots/%x_%j.err

# Rerun of mces_calibration_plots.py only -- the main test_to_test_binned_box.png
# now uses molecule-level scoring matching validation's own protocol
# (ood_generalization_check.score_test_to_test_molecule_level), instead of
# the exhaustive spectrum-vs-spectrum comparison it used before. Steps 1-4
# of the original pipeline (embeddings/intermediates, cosine intermediates,
# GT-MCES-from-matrices, comparison table) and the other 3 plotting scripts
# are unaffected by this fix and don't need rerunning.
#
# --force_recompute is required: the existing scored_pairs_cache.pkl
# predates today's new cache keys (tt_mol_*) and would KeyError without it.

set -euo pipefail
module load uv
SIMBA_DIR=/sofia/projects/2026_053/simba_project/simba
cd "${SIMBA_DIR}"

echo "===== Rerun mces_calibration_plots.py (014_2, Gaetan, molecule-level test-to-test fix) ====="
date

uv run python tools/mces_calibration_plots.py \
    --intermediates_dir /sofia/projects/2026_053/simba_project/experiments/retrieval_split_comparison/gaetan_test/simba_iceberg_014_2_regression \
    --gt_mces_dir /sofia/projects/2026_053/simba_project/data/gt_mces_gaetan_test \
    --test_to_test_prepro_dir /sofia/projects/2026_053/simba_project/data/massspecgym/preprocessing_gaetan_split_max_lb_hdf5_v2 \
    --output_dir /sofia/projects/2026_053/simba_project/experiments/014_2_gaetan_diagnostic_plots \
    --force_recompute

echo "===== Done: $(date) ====="
