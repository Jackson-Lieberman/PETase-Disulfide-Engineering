#!/usr/bin/env bash
set -e

PY="/opt/anaconda3/envs/petase/bin/python"
MD="/Users/jacksonlieberman/Desktop/PETase/Mutations/md.py"
BASE="/Users/jacksonlieberman/Desktop/PETase/Mutations"
SOLV="$BASE/03_structures_solvated"
OUTROOT="$BASE/md"

# replicate settings
SEED=3
NS=5
EQ_PS=500
PLATFORM="OpenCL"   # optional; remove if you want md.py default

# temps: "label Kelvin"
temps=(
  "30C 303.15"
  "40C 313.15"
  "50C 323.15"
  "60C 333.15"
  "70C 343.15"
  "80C 353.15"
)

# proteins: "PROTNAME PDBFILENAME"
proteins=(
  "FASTPETASE_WT FASTPETASE_WT_raw_solv.pdb"
  "WTPETASE_WT WTPETASE_WT_raw_solv.pdb"
  "ThemoPETASE_WT ThemoPETASE_WT_raw_solv.pdb"
  "FASTPETASE_A82C_A130C FASTPETASE_A82C_A130C_raw_solv.pdb"
  "FASTPETASE_M156C_S166C FASTPETASE_M156C_S166C_raw_solv.pdb"
  "FASTPETASE_S54C_Y69C FASTPETASE_S54C_Y69C_raw_solv.pdb"
  "FASTPETASE_S54C_Y69C_A82C_A130C FASTPETASE_S54C_Y69C_A82C_A130C_raw_solv.pdb"
  "FASTPETASE_S54C_Y69C_A82C_A130C_M156C_S166C FASTPETASE_S54C_Y69C_A82C_A130C_M156C_S166C_raw_solv.pdb"
)

for p in "${proteins[@]}"; do
  set -- $p
  prot="$1"
  pdbfile="$2"
  pdb="$SOLV/$pdbfile"

  if [ ! -f "$pdb" ]; then
    echo "ERROR: Missing input PDB: $pdb"
    exit 1
  fi

  for t in "${temps[@]}"; do
    set -- $t
    label="$1"
    kelvin="$2"

    out="$OUTROOT/$prot/T${label}_rep3"
    mkdir -p "$out"

    echo "================================================================================"
    echo "RUN rep3  pdb=$pdbfile | T=${kelvin}K ($label) | seed=$SEED"
    echo "OUT       $out"
    echo "================================================================================"

    "$PY" "$MD" \
      --pdb "$pdb" \
      --out "$out" \
      --temp "$kelvin" \
      --seed "$SEED" \
      --ns "$NS" \
      --eq_ps "$EQ_PS" \
      --platform "$PLATFORM"
  done
done

echo "All rep3 runs launched/completed."
