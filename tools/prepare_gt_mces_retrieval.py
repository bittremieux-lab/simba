"""Prepare exact-MCES pairs for SIMBA's official-test retrieval candidates (3c/3e).

Builds the full (test molecule, candidate molecule) pair set — one row per
unique official-test-fold molecule crossed with every molecule in its
MassSpecGym-formula-grouped PubChem candidate list — so a single exact-MCES
computation run covers both:
  3c: GT MCES of the retrieved top-k candidates vs the true molecule, to
      extend hit@1/5/20 with "how close was the best wrong guess".
  3e: GT MCES on the broader test-to-candidate pool, to check whether SIMBA's
      MCES predictions hold up outside its (narrower, MassSpecGym-only)
      training distribution the way Gaetan described.

~3,170 unique official-test molecules x their candidate lists (min 1, median
256, mean ~185 candidates each) = ~585k molecule pairs total (self/true-match
pairs excluded — their GT MCES is trivially 0, no need to compute).

Two-machine workflow (this repo has no metabo_depthcharge install, and
asimov2 doesn't mount /sofia/projects — so prepare and compute must happen on
different machines):
  1. HERE (sofia — has the source MGF + candidates json):
       uv run python tools/prepare_gt_mces_retrieval.py prepare
  2. Copy OUTPUT_DIR (meta.json, smiles.txt, pairs.npy — the blocks/ dir
     doesn't exist yet) to asimov2, e.g.:
       rsync -av <output_dir>/ asimov2:/mnt/data2/nkubrakov/mces_exact_retrieval_candidates/
  3. THERE (asimov2 — has metabo_depthcharge + spare CPUs):
       sbatch tools/slurm/mces_exact_retrieval_candidates.slurm.sh
       uv run python tools/prepare_gt_mces_retrieval.py --output_dir <path> status
       uv run python tools/prepare_gt_mces_retrieval.py --output_dir <path> combine

compute_block/status/combine are reused UNMODIFIED from
compute_mces_exact_1020.py (same block/meta.json/smiles.txt/pairs.npy schema,
same watchdog, same threshold=20 exact-MCES convention used everywhere else
in SIMBA). combine() writes its output as "mces_exact_10_20.npy" — a name
inherited from that tool's own [10,20]-lb mining use case, not a filter
applied here (there is none: every test-to-candidate pair is included, capped
at threshold=20 same as everywhere else) — this script's own `combine`
subcommand renames it to "mces_exact.npy" right after, so nothing confusing
survives into the file you'd actually use downstream.

Paths stored in meta.json are re-derived from --output_dir on every
compute_block/status/combine call, so it's safe to move the directory
between machines (or if the local copy lands at a different absolute path).

Note: MGF parsing here uses a plain manual scan, not matchms.importing.
load_from_mgf() — the matchms loader took 15+ minutes on this filesystem for
this same 300+MB file (Spectrum-object construction overhead per entry,
apparently very slow here for large files); the manual scan does the same
job in ~3 seconds since it only needs FOLD/SMILES fields, not full spectra.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from rdkit import Chem

from compute_mces_exact_1020 import THRESHOLD, combine, compute_block, status

MGF_PATH = "/sofia/projects/2026_053/simba_project/data/massspecgym/data/auxiliary/MassSpecGym.mgf"
CANDIDATES_JSON = "/sofia/projects/2026_053/spectrawl_project/data/massspecgym/MassSpecGym_retrieval_candidates_formula.json"
DEFAULT_OUTPUT_DIR = Path(
    "/sofia/projects/2026_053/simba_project/data/gt_mces_retrieval_candidates"
)
COMBINED_NPY_NAME = "mces_exact.npy"


def canon(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def _load_test_fold_smiles(mgf_path: str) -> set[str]:
    """Fast manual MGF scan for fold=test SMILES — see module docstring for
    why this is used instead of matchms.importing.load_from_mgf."""
    smis = set()
    fold = None
    smi = None
    with open(mgf_path) as fh:
        for line in fh:
            line = line.strip()
            if line == "BEGIN IONS":
                fold, smi = None, None
            elif line.upper().startswith("FOLD="):
                fold = line.split("=", 1)[1].strip()
            elif line.upper().startswith("SMILES="):
                smi = line.split("=", 1)[1].strip()
            elif line == "END IONS":
                if fold == "test" and smi:
                    smis.add(smi)
    return smis


def prepare(output_dir: Path, n_blocks: int) -> None:
    """Extract all test-to-candidate molecule pairs, write meta + smiles.txt + pairs.npy."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "blocks").mkdir(exist_ok=True)

    print("Loading test-fold query molecules from MGF ...")
    raw_query_smis = _load_test_fold_smiles(MGF_PATH)
    query_canon: dict[str, str] = {}
    for s in raw_query_smis:
        c = canon(s)
        if c:
            query_canon[c] = s
    print(
        f"  {len(raw_query_smis)} unique raw SMILES -> "
        f"{len(query_canon)} canonical test molecules"
    )

    print("Loading candidate pools ...")
    with open(CANDIDATES_JSON) as fh:
        cand_json = json.load(fh)
    cand_json_canon: dict[str, list[str]] = {}
    for k, v in cand_json.items():
        c = canon(k)
        if c:
            cand_json_canon[c] = v
    print(f"  {len(cand_json)} keys -> {len(cand_json_canon)} canonical")

    print("Building unified molecule index + pair list ...")
    mol_to_idx: dict[str, int] = {}
    smiles_list: list[str] = []

    def get_idx(smi: str) -> int:
        idx = mol_to_idx.get(smi)
        if idx is None:
            idx = len(smiles_list)
            mol_to_idx[smi] = idx
            smiles_list.append(smi)
        return idx

    pairs: list[tuple[int, int]] = []
    missing = 0
    n_candidate_canon_fail = 0
    for q_canon in query_canon:
        cands = cand_json_canon.get(q_canon)
        if cands is None:
            missing += 1
            continue
        q_idx = get_idx(q_canon)
        seen_cands: set[str] = set()
        for c in cands:
            c_canon = canon(c)
            if c_canon is None:
                n_candidate_canon_fail += 1
                continue
            if c_canon == q_canon or c_canon in seen_cands:
                continue  # skip trivial true-match (GT MCES=0) + intra-list dups
            seen_cands.add(c_canon)
            c_idx = get_idx(c_canon)
            pairs.append((q_idx, c_idx))

    print(f"  {missing} test molecules missing from candidates json")
    if n_candidate_canon_fail:
        print(
            f"  {n_candidate_canon_fail} candidate SMILES failed to canonicalize, skipped"
        )
    print(f"  {len(smiles_list)} unique molecules (test + candidates)")
    print(f"  {len(pairs)} test-to-candidate molecule pairs (excl. trivial true-match)")

    smiles_path = output_dir / "smiles.txt"
    smiles_path.write_text("\n".join(smiles_list) + "\n")

    pairs_arr = np.asarray(pairs, dtype=np.int32)
    rng = np.random.default_rng(seed=42)
    rng.shuffle(pairs_arr)
    pairs_path = output_dir / "pairs.npy"
    np.save(pairs_path, pairs_arr)

    n_pairs = len(pairs_arr)
    bounds = [int(round(i * n_pairs / n_blocks)) for i in range(n_blocks + 1)]
    bounds[-1] = n_pairs
    block_sizes = [bounds[i + 1] - bounds[i] for i in range(n_blocks)]

    meta = {
        "n_pairs": n_pairs,
        "n_mols": len(smiles_list),
        "threshold": THRESHOLD,
        "n_blocks": n_blocks,
        "bounds": bounds,
        "pairs_path": str(pairs_path.resolve()),
        "smiles_path": str(smiles_path.resolve()),
        "output_dir": str(output_dir.resolve()),
    }
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    size_mb = pairs_path.stat().st_size / 1e6
    print(f"Wrote {smiles_path}, {pairs_path} ({size_mb:.1f} MB), meta.json")
    print(f"\nReady. {n_blocks} blocks of {min(block_sizes):,}-{max(block_sizes):,} pairs each.")
    print(
        "Copy this directory to asimov2, then: "
        "sbatch tools/slurm/mces_exact_retrieval_candidates.slurm.sh"
    )


def _fix_meta_paths(output_dir: Path) -> None:
    """Re-derive meta.json's stored absolute paths from the CURRENT --output_dir,
    so a directory copied to a different machine/path still works."""
    meta_path = output_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["output_dir"] = str(output_dir.resolve())
    meta["smiles_path"] = str((output_dir / "smiles.txt").resolve())
    meta["pairs_path"] = str((output_dir / "pairs.npy").resolve())
    meta_path.write_text(json.dumps(meta, indent=2))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser(
        "prepare", help="Build smiles.txt/pairs.npy/meta.json (run on sofia)."
    )
    pp.add_argument(
        "--n_blocks", type=int, default=60, help="Number of SLURM array tasks."
    )

    pb = sub.add_parser(
        "compute_block", help="Compute one block, one SLURM array task (run on asimov2)."
    )
    pb.add_argument("--task_id", type=int, required=True)
    pb.add_argument("--n_jobs", type=int, default=-1, help="-1 = all CPUs.")
    pb.add_argument("--timeout", type=float, default=None, help="Per-pair solver timeout (s).")

    sub.add_parser("status", help="Print completion progress.")
    sub.add_parser("combine", help=f"Merge blocks into {COMBINED_NPY_NAME}.")

    a = p.parse_args()
    if a.cmd == "prepare":
        prepare(a.output_dir, a.n_blocks)
    elif a.cmd == "compute_block":
        _fix_meta_paths(a.output_dir)
        compute_block(a.output_dir, a.task_id, a.n_jobs, a.timeout)
    elif a.cmd == "status":
        _fix_meta_paths(a.output_dir)
        status(a.output_dir)
    elif a.cmd == "combine":
        _fix_meta_paths(a.output_dir)
        combine(a.output_dir)
        # combine() is reused unmodified from compute_mces_exact_1020.py, so it
        # always writes "mces_exact_10_20.npy" — a name inherited from that
        # tool's own [10,20]-lb mining use case, not ours (all test-to-candidate
        # pairs, no lb filter). Rename to something that doesn't lie about content.
        legacy_path = a.output_dir / "mces_exact_10_20.npy"
        final_path = a.output_dir / COMBINED_NPY_NAME
        legacy_path.replace(final_path)
        print(f"Renamed {legacy_path.name} -> {final_path.name}")


if __name__ == "__main__":
    main()
