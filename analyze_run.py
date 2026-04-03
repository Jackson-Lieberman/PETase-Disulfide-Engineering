#!/usr/bin/env python3
from __future__ import annotations                                                                                     #imports

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import mdtraj as md


def _pick_atom(top: md.Topology, resseq: int, atom_names: list[str]) -> int:
    """
    Pick first matching atom index for given PDB residue number (resSeq) and allowed atom names.
    Raises ValueError if not found.
    """
    for nm in atom_names:                                                                                              #for every atom
        sel = top.select(f"protein and resSeq {resseq} and name {nm}")                                                 #use MDTraj to find atoms in the protein with the requested residue number and atom name.
        if len(sel) == 1:                                                                                              #if one exact atom was requested
            return int(sel[0])                                                                                         #give the index
        if len(sel) > 1:                                                                                               #if more than one atom matched what was requested
            return int(sel[0])                                                                                         #return the first index
    raise ValueError(f"Could not find any of atoms {atom_names} in protein residue resSeq={resseq}")                   #if it couldnt be found give a value error


def native_contacts_pairs(traj_prot: md.Trajectory, ca_only: bool = True, cutoff_nm: float = 0.45):
    """
    Define native contacts from a reference frame (frame 0):
    - Choose CA-CA pairs separated by >= 3 residues (to avoid neighbors).
    - Keep those within cutoff in the reference.
    Returns (pairs, cutoff_nm).
    """
    ref = traj_prot[0]                                                                                                 #gets the first frame of the protein as the reference structure.

    if ca_only:                                                                                                        #if you want only the backbone
        ca = ref.topology.select("protein and name CA")                                                                #gets all the CA resiudes from their names
        res_idx = np.array([ref.topology.atom(i).residue.index for i in ca])                                           #map CA atoms -> residue indices

        pairs = []
        for i in range(len(ca)):                                                                                       #for every CA atom
            for j in range(i + 1, len(ca)):                                                                            #for every CA atom after the current one (to avoid double counting)
                if abs(res_idx[i] - res_idx[j]) < 3:                                                                   #if the residue indices are less than 3 apart, they are neighbors and we skip them
                    continue                                                                                          
                pairs.append((int(ca[i]), int(ca[j])))                                                                 #if they are not neighbors then store the atom-index pair as a candidate native-contact pair

        pairs = np.array(pairs, dtype=int)                                                                             #converts the Python list into a NumPy integer array for MDTraj.
        d0 = md.compute_distances(ref, pairs)[0]                                                                       # get distance for every candidate pair in nanometers.
        native = pairs[d0 < cutoff_nm]                                                                                 #keep only those pairs whose distance in the reference structure is less than the cutoff
        return native, cutoff_nm                                                                                       #returns the native pairs and their cutoff
    else:
        raise NotImplementedError("Use CA-only native contacts for now.")                                              #Pretty much says you cant to heavy-atom contacts


def compute_q(traj_prot: md.Trajectory, native_pairs: np.ndarray, cutoff_nm: float) -> np.ndarray:
    """
    Q(t) = fraction of native contacts present per frame.
    """
    d = md.compute_distances(traj_prot, native_pairs)                                                                   # computes all native-contact distances in all frames... the result looks like (n_frames, n_pairs).
    return (d < cutoff_nm).mean(axis=1)                                                                                 #returns T/F if a native contact is present in each frame, then averages over all pairs to get the fraction of native contacts present in each frame. The result is a 1D array of length n_frames with values between 0 and 1.


def try_dssp(traj_prot: md.Trajectory) -> pd.DataFrame | None:
    """
    Returns per-frame secondary structure fractions if DSSP is available.
    """
    try:
        dssp = md.compute_dssp(traj_prot)                                                                              # computes DSSP assignments for every residue in every frame.          
    except Exception as e:                                                                                             #catches failure
        print(f"[WARN] DSSP not available / failed: {e}")
        return None

    # DSSP codes: H,G,I = helix; E,B = sheet; everything else = coil/turn/bend
    helix = np.isin(dssp, ["H", "G", "I"]).mean(axis=1)                                                                #fraction of residues that are helix-like in each frame.
    sheet = np.isin(dssp, ["E", "B"]).mean(axis=1)                                                                     #fraction of residues that are sheet-like in each frame.
    coil = 1.0 - helix - sheet                                                                                         #takes everything not helix or sheet as coil-like.
    return pd.DataFrame({"helix_frac": helix, "sheet_frac": sheet, "coil_frac": coil})                                 #DataFrame with three columns: helix fraction, sheet fraction, and coil fraction


def main(): 
    ap = argparse.ArgumentParser(description="Compute MD metrics for a single run folder.")                            #creates the command-line parser and gives the script a description.
    ap.add_argument("--run_dir", required=True, help="Run directory containing traj*.dcd/log*.csv/final.pdb")          #arg for run directory containing trajectory and log outputs.
    ap.add_argument("--top", required=True, help="Topology PDB used for the DCD (your *_raw_solv.pdb)")                #arg for topology PDB file and matches DCD trajectory.
    ap.add_argument("--frame_ps", type=float, default=1.0, help="Time per saved frame in ps (matches your DCD report interval)")           #arg for time per saved frame- default: 1 ps
    ap.add_argument("--prod_fraction", type=float, default=0.5, help="Analyze last fraction of frames (e.g., 0.5 = last 50%)")             #arg for fraction of run used the analysis window
    ap.add_argument("--disulfide", nargs=2, action="append", type=int,                                                 #finds the SG atoms for the specified residue numbers and computes their distance across all frames (repeatable)
                    help="Disulfide residue pair by PDB numbering: e.g. --disulfide 54 69 (repeatable)")
    ap.add_argument("--triad", nargs=3, type=int, default=None,                                                        #finds the catalytic triad residues by their residue numbers and computes distances across all frames
                    help="Catalytic triad residues resSeq: Ser Asp His (e.g., 160 206 237) to compute geometry distances")
    args = ap.parse_args()                                                                                             #puts the command-line inputs into args

    run_dir = Path(args.run_dir).resolve()                                                                             #takes run directory and makes it a string .
    top_path = Path(args.top).resolve()                                                                                #takes topology path and makes it a string

    traj_files = sorted(run_dir.glob("traj*.dcd"))                                                                     #gets the trajectories and allows for continuations
    if not traj_files:                                                                                                 #error if not found
        raise FileNotFoundError(f"No traj*.dcd found in {run_dir}")

    print("Loading topology:", top_path.name)                                                                          #prints the name of the topology file being loaded
    print("Loading trajectories:", [p.name for p in traj_files])                                                       #prints the names of the trajectory files being loaded

    traj = md.load([str(p) for p in traj_files], top=str(top_path))                                                    #loads the trajectory files together using the topology

    prot_idx = traj.topology.select("protein")                                                                         #keeps only the protein  
    traj_prot = traj.atom_slice(prot_idx)                                                                              #makes it into a new trajectory with only the protein

    try:                                                                                                               #tries to image molecules
        traj_prot.image_molecules(inplace=True)
    except Exception:                                                                                                  #if not it just passes
        pass

    # Choose analysis frames: last X% of production frames
    n = traj_prot.n_frames                                                                                             #gets the number of frames in trajectory
    start = int(np.floor(n * (1.0 - args.prod_fraction)))                                                              #gets first frame of the analysis window
    start = max(0, min(start, n - 1))                                                                                  #makes it so it stays within bounds
    traj_win = traj_prot[start:]                                                                                       #creates the analysis window

    time_ps = np.arange(n) * args.frame_ps                                                                             #frame by frame time axis in picoseconds
    time_ns = time_ps / 1000.0                                                                                         #changes it to nanoseconds
    time_ns_win = time_ns[start:]                                                                                      #then makes the same thing for the analysis window

    ref = traj_prot[0]                                                                                                 #gets first frame(0) as the reference structure.
    ca_idx = traj_prot.topology.select("name CA")                                                                      #finds all CA residues

    traj_prot.superpose(ref, atom_indices=ca_idx)                                                                      #get protein trajectory similarity to frame 0 using backbone
    traj_win.superpose(ref, atom_indices=ca_idx)                                                                       #same thing for analysis window

    rmsd_nm = md.rmsd(traj_prot, ref, atom_indices=ca_idx)                                                             #gets RMSD for every frame relative to the reference, in nm
    rg_nm = md.compute_rg(traj_prot)                                                                                   #gets Rg for every frame relative to the reference, in nm

    native_pairs, q_cutoff = native_contacts_pairs(traj_prot, ca_only=True, cutoff_nm=0.45)                            #gets the native-contact set from frame 0 and cutoff
    q = compute_q(traj_prot, native_pairs, q_cutoff)                                                                   #computes the fraction of native contacts

    ss_df = try_dssp(traj_win)                                                                                         #tries to get Q for secondary structure

    rmsf_nm = md.rmsf(traj_win, ref, atom_indices=ca_idx)                                                              #RMSF for analysis window relative to reference, in nm

    ca_atoms = [traj_prot.topology.atom(i) for i in ca_idx]                                                            #makes a list of the atoms in the backbone
    resseqs = np.array([a.residue.resSeq for a in ca_atoms], dtype=int)                                                #gets the pdb numbers in the sequence for those atoms

    rmsf_df = pd.DataFrame({"resSeq": resseqs, "rmsf_nm": rmsf_nm})                                                    #creates a residue by residue csv 

    disulfide_series = {}                                                                                              
    if args.disulfide:                                                                                                 #checks if disulfide pairs were requested
        for (r1, r2) in args.disulfide:                                                                                #loops over each disulfide
            a1 = _pick_atom(traj_prot.topology, r1, ["SG"])                                                            #get the index for one cystine
            a2 = _pick_atom(traj_prot.topology, r2, ["SG"])                                                            #get index of the second
            d_nm = md.compute_distances(traj_prot, np.array([[a1, a2]], dtype=int))[:, 0]                              #gets the distance between the atoms for every frame
            disulfide_series[f"S{r1}-S{r2}_nm"] = d_nm                                                                 #stores it

    triad_series = {}                                                                                                  
    if args.triad:                                                                                                     #if you want to check the triad
        ser_r, asp_r, his_r = args.triad                                                                               #splits the traid into its three rediues

        serO = _pick_atom(traj_prot.topology, ser_r, ["OG", "OG1"])                                                    #finds the oxygen on Ser or Thr
        hisNE2 = _pick_atom(traj_prot.topology, his_r, ["NE2"])                                                        #Ser-His distance
        hisND1 = _pick_atom(traj_prot.topology, his_r, ["ND1"])                                                        #His-Asp distance
        aspOD = _pick_atom(traj_prot.topology, asp_r, ["OD2", "OD1"])                                                  #finds one of the Asp carboxylate oxygens

        triad_series["SerOG-HisNE2_nm"] = md.compute_distances(traj_prot, np.array([[serO, hisNE2]]))[:, 0]            #Ser–His proxy distance over time
        triad_series["HisND1-AspOD_nm"] = md.compute_distances(traj_prot, np.array([[hisND1, aspOD]]))[:, 0]           #His-Asp proxy distance over time

    out_dir = run_dir / "analysis"                                                                                     #makes an analysis fold to store analysis outputs
    out_dir.mkdir(exist_ok=True)                                                                                       #checks that it doesnt already exist

    ts = pd.DataFrame({                                                                                                #starts building data frame for all metrics
        "time_ns": time_ns,                                                                                            #sets time axis in ns for every frame
        "rmsd_nm": rmsd_nm,                                                                                            #adds the rmsd for every frame
        "rg_nm": rg_nm,                                                                                                #adds the rg for every frame
        "Q_native": q,                                                                                                 #adds the native contacts for every frame
    })  
    for k, v in disulfide_series.items():                                                                              #loops through disulfide distances
        ts[k] = v                                                                                                      #adds each disulfide as a new collumn
    for k, v in triad_series.items():                                                                                  #loops through traid distances
        ts[k] = v                                                                                                      #adds triad as new collumn

    ts.to_csv(out_dir / "timeseries.csv", index=False)                                                                 #writes the dataframe to a csv file in the analysis
    rmsf_df.to_csv(out_dir / "rmsf_ca.csv", index=False)                                                               #makes a csv for the residue level RMSF

    if ss_df is not None:                                                                                              #if secondary structure worked
        ss_out = ss_df.copy()                                                                                          #make a copy of the data frame
        ss_out.insert(0, "time_ns", time_ns_win)                                                                       #add the time axis to the ss dataframe
        ss_out.to_csv(out_dir / "secstruct_window.csv", index=False)                                                   #write the dataframe to a new csv

    win_mask = np.zeros(n, dtype=bool)                                                                                 #create boolean mask for all frames- starts false
    win_mask[start:] = True                                                                                            #Makes the analysis window true

    summary = {                                                                                                        #makes a summary dictionary
        "run_dir": str(run_dir),                                                                                       #stores the directory of the run as text
        "n_frames_total": int(n),                                                                                      #stores frame count
        "analysis_start_frame": int(start),                                                                            #stores the begining of the analysis window
        "analysis_fraction": float(args.prod_fraction),                                                                #stores the fraction of the run that was analyzed

        "mean_rmsd_nm_window": float(rmsd_nm[win_mask].mean()),                                                        #stores mean RMSD
        "std_rmsd_nm_window": float(rmsd_nm[win_mask].std(ddof=1)),                                                    #stores standard deviation of RMSD

        "mean_rg_nm_window": float(rg_nm[win_mask].mean()),                                                            #stores mean Rg
        "std_rg_nm_window": float(rg_nm[win_mask].std(ddof=1)),                                                        #stores standard deviation of Rg
        "mean_Q_window": float(q[win_mask].mean()),                                                                    #stores mean native contacts
        "std_Q_window": float(q[win_mask].std(ddof=1)),                                                                #stores standard deviation of native contacts
    }

    for k, v in disulfide_series.items():                                                                              #for each disulfide
        summary[f"mean_{k}_window"] = float(v[win_mask].mean())                                                        #store mean disulfide distance
        summary[f"std_{k}_window"] = float(v[win_mask].std(ddof=1))                                                    #store standard deviation of disufide distance
    for k, v in triad_series.items():                                                                                  #for the triad
        summary[f"mean_{k}_window"] = float(v[win_mask].mean())                                                        #store mean triad distance
        summary[f"std_{k}_window"] = float(v[win_mask].std(ddof=1))                                                    #staore standard deviation of traid distances

    # RMSF summary
    summary["mean_RMSF_CA_nm_window"] = float(rmsf_nm.mean())                                                          #stores mean RMSF
    summary["median_RMSF_CA_nm_window"] = float(np.median(rmsf_nm))                                                    #stores median RMSF

    with open(out_dir / "summary.json", "w") as f:                                                                     #writes a summary into a json in the output folder
        json.dump(summary, f, indent=2)                                                                                

    print("Wrote:")                                                                                     
    print(" ", out_dir / "timeseries.csv")
    print(" ", out_dir / "rmsf_ca.csv")
    if ss_df is not None:
        print(" ", out_dir / "secstruct_window.csv")
    print(" ", out_dir / "summary.json")
    print(f'''Key window means:
    mean RMSD (nm): {summary["mean_rmsd_nm_window"]}
    mean Rg (nm): {summary["mean_rg_nm_window"]}
    mean Q: {summary["mean_Q_window"]}''')


if __name__ == "__main__":
    main()
