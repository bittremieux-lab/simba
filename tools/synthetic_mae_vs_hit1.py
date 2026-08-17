"""hit@1 vs prediction MAE, interpolating toward SIMBA's ACTUAL prediction
(not independent noise). Reuses item 8b's retrieval_comparison_table.csv
directly (no re-scoring): c = gt_mces + (simba_mces - gt_mces) * w. w=0 is
the perfect predictor (c=gt_mces), w=1 is SIMBA's real, already-measured
prediction (c=simba_mces) -- exactly reproducing SIMBA's real hit@1 on this
population, since ranking by c=simba_mces is the same ordering as ranking by
simba_similarity descending (verified: w=1 gives hit@1=10.25%, matching the
already-known official-split ICEBERG+SIMBA number exactly). Intermediate w
blends toward SIMBA's REAL error STRUCTURE (whatever correlation/bias it has
across candidates in the same pool), not independent per-pair noise. MAE(c,
gt_mces) = w * MAE(simba_mces, gt_mces) exactly (no clipping applied -- c
isn't meant to model a bounded predictor, just SIMBA's real values), so it's
reported but not independently estimated.

Ceiling check (confirmed directly, not assumed): 27/17,555 official-split
test spectra have MORE THAN ONE candidate tied at GT MCES=0 -- these are
stereoisomers (same 2D connectivity, different stereo descriptors) that
MCES doesn't distinguish but that survived candidate dedup since they have
different canonical SMILES. At w=0 this script still reports hit@1=100%
though -- NOT because ties don't matter, but because all 27/27 tied groups
happen to have the true candidate listed FIRST in the candidate JSON, and
idxmin's tie-break (first-occurrence, same convention as rank_descending in
build_retrieval_comparison_table.py -- not the "fair random" tie-break used
specifically in mces_top1_diagnostics.py) always picks it. A fair/random
tie-break would show the real ceiling of ~99.85% instead. Confirmed by
direct check: for every one of the 27 tied groups, the first gt_mces==0 row
in file order has is_correct==1.

Usage:
    uv run python tools/synthetic_mae_vs_hit1.py \\
        --comparison_csv /path/to/retrieval_comparison_table.csv \\
        --output_dir /path/to/output \\
        --interp_ws 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \\
        --n_repeats 5
"""

import argparse
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def hit1_for_interpolation(
    gt_mces: np.ndarray,
    simba_mces: np.ndarray,
    test_spec_idx: np.ndarray,
    is_correct: np.ndarray,
    w: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """c = gt_mces + (simba_mces - gt_mces) * w -- deterministic (no injected
    randomness; only the tie-break is randomized). Fair tie-break: shuffle
    row order first, THEN idxmin (which always picks the first occurrence) --
    confirmed directly that the true candidate is listed FIRST in literally
    100% of test_spec_idx groups in this CSV, so an unshuffled idxmin would
    silently hand it every tie. Returns (hit1_rate, mae(c, gt_mces))."""
    c = gt_mces + (simba_mces - gt_mces) * w
    mae = float(np.nanmean(np.abs(c - gt_mces)))

    df = pd.DataFrame(
        {"test_spec_idx": test_spec_idx, "c": c, "is_correct": is_correct}
    )
    df = df.sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
    top1 = df.loc[df.groupby("test_spec_idx", sort=False)["c"].idxmin()]
    hit1 = float(top1["is_correct"].mean())
    return hit1, mae


def run(
    comparison_csv: str,
    output_dir: str,
    interp_ws: str = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    n_repeats: int = 5,
    seed: int = 42,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ws = [float(x) for x in interp_ws.split(",") if x.strip()]

    print(f"Loading {comparison_csv} ...")
    df = pd.read_csv(
        comparison_csv,
        usecols=["test_spec_idx", "gt_mces", "simba_mces", "is_correct"],
    )
    gt_mces = df["gt_mces"].to_numpy()
    simba_mces = df["simba_mces"].to_numpy()
    test_spec_idx = df["test_spec_idx"].to_numpy()
    is_correct = df["is_correct"].to_numpy()
    n_test = df["test_spec_idx"].nunique()
    print(f"  {len(df):,} rows, {n_test:,} test spectra")

    rng = np.random.default_rng(seed)
    means, los, his, maes = [], [], [], []
    for w in ws:
        n_rep = 1 if w == 0 else n_repeats
        hits, real_maes = [], []
        for _ in range(n_rep):
            h, real_mae = hit1_for_interpolation(
                gt_mces, simba_mces, test_spec_idx, is_correct, w, rng
            )
            hits.append(h)
            real_maes.append(real_mae)
        hits = np.array(hits)
        means.append(hits.mean())
        los.append(hits.min())
        his.append(hits.max())
        maes.append(float(np.mean(real_maes)))
        print(
            f"  w={w:g}: hit@1={hits.mean():.4f} "
            f"(range {hits.min():.4f}-{hits.max():.4f} over {n_rep} repeats), "
            f"mae={np.mean(real_maes):.3f}"
        )

    results = pd.DataFrame(
        {"w": ws, "mae": maes, "hit1_mean": means, "hit1_min": los, "hit1_max": his}
    )
    results_csv = out_dir / "interpolation_mae_vs_hit1.csv"
    results.to_csv(results_csv, index=False)
    print(f"Saved {results_csv}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    means_pct = [m * 100 for m in means]
    los_pct = [x * 100 for x in los]
    his_pct = [x * 100 for x in his]
    ax.plot(
        maes,
        means_pct,
        "s-",
        color="tab:orange",
        label="interpolated toward SIMBA's real prediction",
    )
    ax.fill_between(maes, los_pct, his_pct, alpha=0.2, color="tab:orange")

    # w=1 -- SIMBA's actual, unmodified prediction -- specifically marked:
    # this is the real reference point the whole plot exists to contextualize,
    # not just another sample on the curve.
    w1_idx = ws.index(1.0) if 1.0 in ws else len(ws) - 1
    ax.scatter(
        [maes[w1_idx]],
        [means_pct[w1_idx]],
        color="black",
        zorder=5,
        s=70,
        marker="*",
        label="SIMBA actual (w=1)",
    )
    ax.annotate(
        f"SIMBA actual\nMAE={maes[w1_idx]:.2f}, hit@1={means_pct[w1_idx]:.1f}%",
        (maes[w1_idx], means_pct[w1_idx]),
        textcoords="offset points",
        xytext=(10, 10),
        fontsize=8,
    )

    ax.set_xlabel("MAE vs GT MCES (MCES units)")
    ax.set_ylabel("hit@1 (%)")
    ax.set_title(
        f"hit@1 vs prediction MAE, interpolated toward SIMBA's real error\n"
        f"(n={n_test:,} test spectra, {n_repeats} repeats/point)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = out_dir / "synthetic_mae_vs_hit1.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--comparison_csv", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--interp_ws", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    p.add_argument("--n_repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
