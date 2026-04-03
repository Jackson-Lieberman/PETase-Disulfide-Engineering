from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# -------------------------
# EDIT THESE SETTINGS
# -------------------------
EQ_PS = 500       # equilibration length (ps)
NS = 10           # production length (ns)
REPORT_PS = 10    # trajectory/log interval (ps)
CHECKPOINT_PS = 100
SEEDS = [1]       # replicates: [1,2] for 2 replicates, etc.

# 30, 40, 50, 60, 70, 80 °C
TEMPS_C = [30, 40, 50, 60, 70, 80]
# -------------------------

BASE = Path(__file__).resolve().parent

MD_PY = BASE / "md.py"
SOLV_DIR = BASE / "01_structures_raw" / "03_structures_solvated"
OUT_BASE = BASE / "md"

# Your 7 solvated PDB files (exact names as you provided)
PROTEINS = [
    "FASTPETASE_WT_raw_solv.pdb",
    "FASTPETASE_S54C_Y69C_raw_solv.pdb",
    "FASTPETASE_A82C_A130C_raw_solv.pdb",
    "FASTPETASE_M156C_S166C_raw_solv.pdb",
    "FASTPETASE_S54C_Y69C_A82C_A130C_raw_solv.pdb",
    "ThemoPETASE_WT_raw_solv.pdb",
    "WTPETASE_WT_raw_solv.pdb",
]


def c_to_k(c: float) -> float:
    return c + 273.15


def run_one(pdb_path: Path, out_dir: Path, temp_k: float, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already finished
    final_pdb = out_dir / "final.pdb"
    if final_pdb.exists() and final_pdb.stat().st_size > 0:
        print(f"SKIP (already done): {out_dir}")
        return

    cmd = [
        sys.executable, str(MD_PY),
        "--pdb", str(pdb_path),
        "--out", str(out_dir),
        "--temp", str(temp_k),
        "--eq_ps", str(EQ_PS),
        "--ns", str(NS),
        "--seed", str(seed),
        "--report_ps", str(REPORT_PS),
        "--checkpoint_ps", str(CHECKPOINT_PS),
    ]

    print("\n" + "=" * 80)
    print(f"RUN  pdb={pdb_path.name} | T={temp_k:.2f}K ({temp_k-273.15:.0f}C) | seed={seed}")
    print(f"OUT  {out_dir}")
    print("=" * 80)

    # Run and stream output to your terminal
    subprocess.run(cmd, check=True)


def main():
    if not MD_PY.exists():
        raise FileNotFoundError(f"Could not find md.py at: {MD_PY}")

    if not SOLV_DIR.exists():
        raise FileNotFoundError(f"Could not find solvated folder at: {SOLV_DIR}")

    # Validate all PDBs exist
    missing = [p for p in PROTEINS if not (SOLV_DIR / p).exists()]
    if missing:
        print("Missing solvated PDB(s):")
        for m in missing:
            print("  -", SOLV_DIR / m)
        raise SystemExit(1)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    for pdb_name in PROTEINS:
        pdb_path = SOLV_DIR / pdb_name
        protein_name = pdb_name.replace("_raw_solv.pdb", "")

        for seed in SEEDS:
            for c in TEMPS_C:
                temp_k = c_to_k(c)
                out_dir = OUT_BASE / protein_name / f"T{c}C_rep{seed}"
                run_one(pdb_path, out_dir, temp_k, seed)

    print("\nALL RUNS COMPLETE.")


if __name__ == "__main__":
    main()
