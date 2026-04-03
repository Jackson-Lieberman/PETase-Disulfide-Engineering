#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ----------------------------
# Topology selection utilities
# ----------------------------

def find_topology_in_top_dir(top_dir: Path, protein_name: str) -> Path | None:
    """
    Old behavior: find solvated topology PDB for a protein in 03_structures_solvated/.
    Expected: {protein}_raw_solv.pdb, with fallbacks.
    """
    exact = top_dir / f"{protein_name}_raw_solv.pdb"
    if exact.exists():
        return exact

    hits = sorted(top_dir.glob(f"{protein_name}*raw_solv*.pdb"))
    if hits:
        return hits[0]

    pname_low = protein_name.lower()
    for p in sorted(top_dir.glob("*.pdb")):
        s = p.name.lower()
        if pname_low in s and "raw_solv" in s:
            return p

    return None


def find_run_topology(run_dir: Path, top_dir: Path, protein_name: str) -> tuple[Path | None, str]:
    """
    Robust behavior: choose a topology that matches the DCD for THIS run.

    Priority:
      1) run_dir/topology.pdb  (if you choose to save input topology per run)
      2) run_dir/final.pdb     (written by your MD code; matches the DCD atom-for-atom)
      3) fallback: top_dir/{protein}_raw_solv.pdb (only if needed)
    """
    cand1 = run_dir / "topology.pdb"
    if cand1.exists():
        return cand1, "run_dir/topology.pdb"

    cand2 = run_dir / "final.pdb"
    if cand2.exists():
        return cand2, "run_dir/final.pdb"

    # fallback
    top = find_topology_in_top_dir(top_dir, protein_name)
    if top is not None:
        return top, "top_dir/*raw_solv*.pdb"

    return None, "none"


def looks_like_atom_mismatch(stderr_text: str) -> bool:
    """
    Detect the common MDTraj error for mismatched topology vs trajectory.
    """
    t = (stderr_text or "").lower()
    return (
        "topology and the trajectory files might not contain the same atoms" in t
        or "must be shape" in t
        or "contain the same atoms" in t
    )


# ----------------------------
# Your existing helpers
# ----------------------------

def disulfide_pairs_from_name(name: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    if ("S54C" in name) or ("Y69C" in name):
        pairs.append((54, 69))
    if ("A82C" in name) or ("A130C" in name):
        pairs.append((82, 130))
    if ("M156C" in name) or ("S166C" in name):
        pairs.append((156, 166))

    # de-dup preserve order
    out = []
    for pr in pairs:
        if pr not in out:
            out.append(pr)
    return out


def pdb_has_resseq(pdb_path: Path, resseq: int) -> bool:
    rs = f"{resseq:4d}"
    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[22:26] == rs:
                return True
    return False


def parse_temp_and_rep(run_name: str) -> tuple[str | None, str | None]:
    m = re.match(r"^T(\d+)(C)?_rep(\d+)$", run_name)
    if not m:
        return (None, None)
    t = m.group(1)
    cflag = m.group(2)
    rep = m.group(3)
    if cflag:
        return (f"{t}C", f"rep{rep}")
    else:
        return (f"{t}K", f"rep{rep}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch analyze all MD run folders using analyze_run.py (robust topology selection)")
    ap.add_argument("--md_root", default="md", help="Root MD folder (default: md)")
    ap.add_argument("--top_dir", default="03_structures_solvated", help="Folder with *_raw_solv.pdb (fallback only)")
    ap.add_argument("--analyze_script", default="analyze_run.py", help="Path to analyze_run.py")
    ap.add_argument("--frame_ps", type=float, default=1.0, help="Time per saved frame in ps (must match your report interval)")
    ap.add_argument("--prod_fraction", type=float, default=0.5, help="Analyze last fraction of frames (default: 0.5)")
    ap.add_argument("--use_triad", action="store_true", help="Attempt to compute catalytic triad distances if residues exist")
    ap.add_argument("--triad", nargs=3, type=int, default=[160, 206, 237], help="Triad resSeq: Ser Asp His (default: 160 206 237)")
    ap.add_argument("--only", nargs="*", default=None, help="Optional list of protein folder names to analyze")
    ap.add_argument("--skip_existing", action="store_true", help="Skip runs that already have analysis/summary.json")
    args = ap.parse_args()

    md_root = Path(args.md_root).resolve()
    top_dir = Path(args.top_dir).resolve()
    analyze_script = Path(args.analyze_script).resolve()

    if not md_root.exists():
        raise FileNotFoundError(f"md_root not found: {md_root}")
    if not top_dir.exists():
        raise FileNotFoundError(f"top_dir not found: {top_dir}")
    if not analyze_script.exists():
        raise FileNotFoundError(f"analyze_run.py not found: {analyze_script}")

    protein_dirs = [p for p in sorted(md_root.iterdir()) if p.is_dir()]
    if args.only:
        want = set(args.only)
        protein_dirs = [p for p in protein_dirs if p.name in want]

    runs: list[tuple[str, Path]] = []
    for pdir in protein_dirs:
        for rdir in sorted(pdir.iterdir()):
            if not rdir.is_dir():
                continue
            if not list(rdir.glob("traj*.dcd")):
                continue
            if args.skip_existing and (rdir / "analysis" / "summary.json").exists():
                continue
            runs.append((pdir.name, rdir))

    if not runs:
        print(f"No run folders found under {md_root} (expected md/<protein>/TxxC_repY with traj*.dcd)")
        sys.exit(0)

    print(f"Found {len(runs)} run folders to analyze.")

    ok = 0
    failed = 0

    for protein_name, run_dir in runs:
        top_path, top_source = find_run_topology(run_dir, top_dir, protein_name)
        if top_path is None:
            print(f"[SKIP] No topology found for {protein_name} in run_dir or {top_dir}")
            failed += 1
            continue

        cmd = [
            sys.executable,
            str(analyze_script),
            "--run_dir", str(run_dir),
            "--top", str(top_path),
            "--frame_ps", str(args.frame_ps),
            "--prod_fraction", str(args.prod_fraction),
        ]

        for r1, r2 in disulfide_pairs_from_name(protein_name):
            cmd += ["--disulfide", str(r1), str(r2)]

        if args.use_triad:
            ser_r, asp_r, his_r = args.triad
            if pdb_has_resseq(top_path, ser_r) and pdb_has_resseq(top_path, asp_r) and pdb_has_resseq(top_path, his_r):
                cmd += ["--triad", str(ser_r), str(asp_r), str(his_r)]
            else:
                print(f"[WARN] Triad residues not found in {top_path.name}; skipping triad for {protein_name}")

        print("\n" + "-" * 80)
        print("ANALYZE:", protein_name, "|", run_dir.name)
        print("TOPOLOGY:", top_path.name, f"({top_source})")
        print("CMD:", " ".join(cmd))
        print("-" * 80)

        # Run and capture output so we can auto-retry on atom mismatch
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode == 0:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
            ok += 1
            continue

        # Print the failure output
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")

        # If it looks like an atom mismatch, retry using run_dir/final.pdb (if not already)
        final_top = run_dir / "final.pdb"
        if looks_like_atom_mismatch(proc.stderr) and final_top.exists() and final_top != top_path:
            print(f"[RETRY] Atom mismatch suspected. Retrying with {final_top} ...")
            cmd_retry = cmd.copy()
            # swap the --top argument
            top_i = cmd_retry.index("--top") + 1
            cmd_retry[top_i] = str(final_top)

            proc2 = subprocess.run(cmd_retry, text=True, capture_output=True)
            if proc2.returncode == 0:
                print(proc2.stdout, end="" if proc2.stdout.endswith("\n") else "\n")
                ok += 1
                continue
            else:
                if proc2.stdout:
                    print(proc2.stdout, end="" if proc2.stdout.endswith("\n") else "\n")
                if proc2.stderr:
                    print(proc2.stderr, end="" if proc2.stderr.endswith("\n") else "\n")

        print(f"[FAIL] {protein_name} {run_dir.name}")
        failed += 1

    print(f"\nBatch analysis finished: OK={ok} FAIL/SKIP={failed}")

    # Compile summaries
    rows = []
    for protein_name, run_dir in runs:
        s = run_dir / "analysis" / "summary.json"
        if not s.exists():
            continue
        with open(s, "r") as f:
            d = json.load(f)

        temp_label, rep_label = parse_temp_and_rep(run_dir.name)
        d["protein"] = protein_name
        d["run"] = run_dir.name
        d["temp_label"] = temp_label
        d["rep_label"] = rep_label
        rows.append(d)

    if rows:
        df = pd.DataFrame(rows)
        out_csv = md_root / "analysis_master_summary.csv"
        df.to_csv(out_csv, index=False)
        print("Wrote master summary CSV:", out_csv)
    else:
        print("No summary.json files found to compile (did analyses run successfully?)")


if __name__ == "__main__":
    main()
