#!/usr/bin/env python3
"""
paper_figures_revamp.py

Upgrades 4 key "paper-quality" figures from your MD pipeline:
  1) Mean Q vs Temperature (error bars)
  2) Mean Cα RMSD vs Temperature in Å (error bars)
  3) Mean ΔRg (Å) relative to 30°C (error bars)
  4) Disulfide distance histogram overlay (30°C vs 80°C) for S82–S130 in the TRIPLE mutant
     (production window only, last prod_fraction of frames)

Requirements: pandas, numpy, matplotlib
"""

import argparse                                                                                                        #imports
import re                                                                                                           
from pathlib import Path                         

import numpy as np                                                                                                   
import pandas as pd                                                                                                    
import matplotlib.pyplot as plt                                                                                        


# -----------------------------
# Helpers
# -----------------------------
NM_TO_A = 10.0                                                                                                        #conversion factor: 1 nm = 10 Å

def nm_to_A(x):                                                                                                       #converts nanometer values to angstroms
    return np.asarray(x) * NM_TO_A                                                                                    #multiply input by 10 and return as numpy array

def parse_temp_C(x):
    """
    Robustly parse temperature in °C from strings like:
      'T30C_rep1', '30C', 'T80C', 'T70C_rep2', etc.
    """
    if pd.isna(x):                                                                                                     #if the value is NaN or None, cannot parse a temperature
        return None                                                                                                    #return None to indicate missing temperature
    s = str(x)                                                                                                        #convert input to string for regex matching
    m = re.search(r'(\d+)\s*C', s)                                                                                    #try to match a pattern like "30C" or "30 C"
    if m:                                                                                                              #if the pattern was found
        return int(m.group(1))                                                                                        #return the matched integer temperature value
    m = re.search(r'T(\d+)', s)                                                                                       #fallback: try to match a pattern like "T30" (without trailing C)
    if m:                                                                                                              #if the fallback pattern was found
        return int(m.group(1))                                                                                        #return the matched integer temperature value
    return None                                                                                                       #could not parse temperature from string, return None

def nice_name(protein):
    """Pretty names for legends."""
    p = str(protein)                                                                                                   #ensure protein identifier is a string

    mapping = {                                                                                                        #dictionary mapping internal protein folder names to display names for figure legends
        "FASTPETASE_WT": "FAST-PETase (WT)",
        "WTPETASE_WT": "WT PETase",
        "ISPETASE_WT": "IsPETase",
        "ThemoPETASE_WT": "ThermoPETase",
        "ThermoPETASE_WT": "ThermoPETase",
        "FASTPETASE_A82C_A130C": "FAST-PETase +DS (A82C–A130C)",
        "FASTPETASE_S54C_Y69C": "FAST-PETase +DS (S54C–Y69C)",
        "FASTPETASE_M156C_S166C": "FAST-PETase +DS (M156C–S166C)",
        "FASTPETASE_S54C_Y69C_A82C_A130C": "FAST-PETase +2DS (Y69–S54, A82–A130)",
        "FASTPETASE_S54C_Y69C_A82C_A130C_M156C_S166C": "FAST-PETase +3DS (triple)",
    }
    return mapping.get(p, p)                                                                                          #return mapped display name, or the original string if not found in mapping

def set_paper_style():
    """Matplotlib styling: clean, not clunky, paper-ready."""
    plt.rcParams.update({                                                                                              #apply all paper styling parameters to matplotlib's global settings
        "figure.dpi": 140,                                                                                            #screen render DPI for interactive display
        "savefig.dpi": 600,                                                                                           #output DPI when saving figures (publication quality)
        "font.size": 12,                                                                                              #default font size for all text elements
        "axes.titlesize": 16,                                                                                         #font size for axes titles
        "axes.labelsize": 13,                                                                                         #font size for x and y axis labels
        "legend.fontsize": 10,                                                                                        #font size for legend entries
        "xtick.labelsize": 11,                                                                                        #font size for x-axis tick labels
        "ytick.labelsize": 11,                                                                                        #font size for y-axis tick labels
        "axes.spines.top": False,                                                                                     #hide top border of plot area for cleaner look
        "axes.spines.right": False,                                                                                   #hide right border of plot area for cleaner look
        "axes.grid": True,                                                                                            #enable background grid lines
        "grid.alpha": 0.18,                                                                                           #grid lines are nearly transparent so they don't dominate
        "grid.linewidth": 0.8,                                                                                        #thin grid lines to keep the plot clean
        "lines.linewidth": 2.2,                                                                                       #default line width for data series
        "lines.markersize": 6.0,                                                                                      #default marker size for data points
    })

def ensure_dir(p: Path):                                                                                              #creates directory at path p, including any missing parent directories
    p.mkdir(parents=True, exist_ok=True)                                                                              #exist_ok=True prevents error if directory already exists

def find_col(df, candidates):
    """Return first matching column name from candidates list."""
    for c in candidates:                                                                                               #iterate through candidate column names in priority order
        if c in df.columns:                                                                                           #check if this candidate exists as a column in the dataframe
            return c                                                                                                   #return the first matching column name found
    return None                                                                                                       #none of the candidate column names exist in the dataframe

def find_timeseries_col(df, kind):
    """
    Try to locate key timeseries columns.
    - kind='Q' -> Q column
    - kind='rmsd_nm' -> rmsd in nm
    - kind='rg_nm' -> rg in nm
    """
    cols = df.columns.tolist()                                                                                        #get all column names as a list for searching
    if kind == "Q":                                                                                                   #looking for the native contact fraction column
        for c in ["Q", "q", "native_contacts", "native_contacts_Q", "Q_fraction"]:                                   #try known canonical Q column names
            if c in cols:                                                                                             #if this candidate column exists
                return c                                                                                              #return it immediately
        # fallback: any column that is exactly 'Q' ignoring case
        for c in cols:                                                                                                #scan all columns for a case-insensitive Q match
            if c.lower() == "q":                                                                                      #match any column named exactly "Q" or "q"
                return c                                                                                              #return the matching column name
        return None                                                                                                   #no Q column found in this dataframe

    if kind == "rmsd_nm":                                                                                             #looking for the RMSD column in nanometer units
        for c in ["rmsd_nm", "RMSD_nm", "ca_rmsd_nm", "rmsd_backbone_nm"]:                                           #try known canonical RMSD column names
            if c in cols:                                                                                             #if this candidate column exists
                return c                                                                                              #return it immediately
        # fallback regex
        for c in cols:                                                                                                #scan all columns for any that look like an RMSD in nm
            if "rmsd" in c.lower() and c.lower().endswith("_nm"):                                                     #match columns containing "rmsd" and ending with "_nm"
                return c                                                                                              #return the first matching column
        return None                                                                                                   #no RMSD column found in this dataframe

    if kind == "rg_nm":                                                                                               #looking for the radius of gyration column in nanometer units
        for c in ["rg_nm", "Rg_nm", "radius_of_gyration_nm"]:                                                        #try known canonical Rg column names
            if c in cols:                                                                                             #if this candidate column exists
                return c                                                                                              #return it immediately
        for c in cols:                                                                                                #scan all columns for any that look like an Rg in nm
            if ("rg" in c.lower() or "gyr" in c.lower()) and c.lower().endswith("_nm"):                              #match columns containing "rg" or "gyr" and ending with "_nm"
                return c                                                                                              #return the first matching column
        return None                                                                                                   #no Rg column found in this dataframe

    return None                                                                                                       #unknown kind argument, return None

def find_disulfide_col(df, res1=82, res2=130):
    """
    Find a disulfide distance column in timeseries.
    Looks for columns containing both residue numbers and 'nm' (or 'A').
    Examples you've used: 'S82-S130_nm', 'S82-S130_A', etc.
    """
    cols = df.columns.tolist()                                                                                        #get all column names as a list for searching
    r1 = str(res1)                                                                                                    #convert first residue number to string for substring matching
    r2 = str(res2)                                                                                                    #convert second residue number to string for substring matching

    # best: explicit S82-S130_nm / S82-S130_A
    for c in cols:                                                                                                    #scan all columns for the best matching disulfide distance column
        cl = c.lower()                                                                                                #lowercase the column name for case-insensitive comparison
        if r1 in cl and r2 in cl and ("nm" in cl or "_a" in cl or "angstrom" in cl):                                 #column must contain both residue numbers and a unit indicator
            # avoid summary/window columns if present
            if "window" not in cl and "mean" not in cl and "std" not in cl:                                           #skip aggregated columns, prefer raw timeseries values
                return c                                                                                              #return the best matching column name

    # fallback: any column with "S82" and "S130"
    for c in cols:                                                                                                    #looser fallback: any column mentioning both residue numbers
        cl = c.lower()                                                                                                #lowercase for case-insensitive matching
        if "s82" in cl and "s130" in cl:                                                                              #match columns referencing both S82 and S130
            return c                                                                                                  #return the fallback match

    return None                                                                                                       #no disulfide distance column found for these residues

def production_slice(df, prod_fraction=0.5):
    """
    Keep last prod_fraction of frames (production window) by row count.
    """
    n = len(df)                                                                                                       #total number of frames in the timeseries
    start = int(np.floor(n * (1.0 - prod_fraction)))                                                                  #index of first frame in the production window
    return df.iloc[start:].reset_index(drop=True)                                                                     #return the trailing production frames with reset row index


# -----------------------------
# Load + summarize master CSV
# -----------------------------
def load_master(master_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(master_csv)                                                                                      #read the master summary CSV into a dataframe

    # Standardize expected columns / create temp_C
    if "temp_C" not in df.columns:                                                                                    #check if temperature column already exists under canonical name
        temp_source = None                                                                                            #initialize search for an alternative temperature column
        for cand in ["temp_label", "run", "temp", "temperature", "tempC"]:                                            #try common alternative temperature column names in priority order
            if cand in df.columns:                                                                                    #if this candidate column exists in the dataframe
                temp_source = cand                                                                                    #record it as the temperature source
                break                                                                                                 #stop searching after the first match
        if temp_source is None:                                                                                       #if no recognizable temperature column was found
            raise ValueError("Could not find a temperature column in master CSV.")                                    #raise an error so the user knows what is missing
        df["temp_C"] = df[temp_source].apply(parse_temp_C)                                                            #parse temperature values from the source column into numeric °C

    # Must have 'protein'
    if "protein" not in df.columns:                                                                                   #protein column is required to group runs by variant
        raise ValueError("master CSV must include a 'protein' column.")                                               #raise an informative error if it is absent

    # Identify metric columns (window means)
    rmsd_col = find_col(df, ["mean_rmsd_nm_window", "mean_ca_rmsd_nm_window", "mean_rmsd_nm"])                        #find the RMSD window mean column using known naming conventions
    rg_col   = find_col(df, ["mean_rg_nm_window", "mean_rg_nm"])                                                      #find the Rg window mean column using known naming conventions
    q_col    = find_col(df, ["mean_Q_window", "mean_q_window", "mean_Q", "mean_q"])                                   #find the Q window mean column using known naming conventions

    if rmsd_col is None or rg_col is None or q_col is None:                                                           #all three metric columns must be present to proceed
        raise ValueError(
            "Missing required metric columns. Expected something like:\n"
            "  mean_rmsd_nm_window, mean_rg_nm_window, mean_Q_window\n"
            f"Found columns: {list(df.columns)[:30]} ..."
        )

    # store canonical names for downstream use
    df = df.copy()                                                                                                     #copy to avoid modifying the original dataframe in place
    df["_rmsd_nm"] = df[rmsd_col].astype(float)                                                                       #store RMSD values under a canonical internal column name
    df["_rg_nm"]   = df[rg_col].astype(float)                                                                         #store Rg values under a canonical internal column name
    df["_Q"]       = df[q_col].astype(float)                                                                          #store Q values under a canonical internal column name

    # If you have a core RMSF column and want it later, keep it too:
    rmsf_col = find_col(df, ["mean_RMSF_CA_nm_window", "mean_rmsf_ca_nm_window", "mean_rmsf_nm_window"])              #optionally find the RMSF window mean column if present
    if rmsf_col is not None:                                                                                          #only add the canonical RMSF column if a source was found
        df["_rmsf_nm"] = df[rmsf_col].astype(float)                                                                   #store RMSF values under a canonical internal column name

    # rep label helps delta-Rg pairing
    if "rep_label" not in df.columns:                                                                                 #rep_label is needed to pair replicates when computing ΔRg
        # try to infer from run_dir if present
        if "run_dir" in df.columns:                                                                                   #if the run directory path is available, extract rep label from it
            df["rep_label"] = df["run_dir"].astype(str).str.extract(r"(rep\d+)", expand=False)                        #extract "rep1", "rep2", etc. from the run directory string
        else:                                                                                                         #no run_dir column to extract from
            df["rep_label"] = "rep?"                                                                                  #assign a placeholder rep label so merges don't silently drop rows

    return df                                                                                                         #return the cleaned and canonicalized dataframe

def summarize_by_protein_temp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean ± SD across replicates at each (protein, temp_C)
    """
    g = df.groupby(["protein", "temp_C"], as_index=False)                                                             #group rows by protein variant and temperature
    out = g.agg(                                                                                                      #compute summary statistics for each group
        n=("protein", "count"),                                                                                       #number of replicates in each group
        rmsd_nm_mean=("_rmsd_nm", "mean"),                                                                            #mean RMSD across replicates in nm
        rmsd_nm_sd=("_rmsd_nm", "std"),                                                                               #standard deviation of RMSD across replicates
        rg_nm_mean=("_rg_nm", "mean"),                                                                                #mean Rg across replicates in nm
        rg_nm_sd=("_rg_nm", "std"),                                                                                   #standard deviation of Rg across replicates
        Q_mean=("_Q", "mean"),                                                                                        #mean Q across replicates
        Q_sd=("_Q", "std"),                                                                                           #standard deviation of Q across replicates
    )
    return out                                                                                                        #return summary dataframe with one row per (protein, temp_C)

def delta_rg_vs_temp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ΔRg per protein per temperature relative to 30°C,
    pairing by rep_label when possible (better than subtracting two means).
    Returns mean ± SD across replicate deltas.
    """
    base = df[df["temp_C"] == 30][["protein", "rep_label", "_rg_nm"]].rename(columns={"_rg_nm": "rg_nm_30"})         #extract 30°C Rg values per replicate as the reference baseline
    merged = df.merge(base, on=["protein", "rep_label"], how="inner")                                                 #join each row with its corresponding 30°C baseline by protein and replicate
    merged["dRg_nm"] = merged["_rg_nm"] - merged["rg_nm_30"]                                                         #compute per-replicate ΔRg relative to the 30°C value

    g = merged.groupby(["protein", "temp_C"], as_index=False)                                                         #group delta-Rg values by protein and temperature
    out = g.agg(                                                                                                      #compute summary statistics for each group
        n=("dRg_nm", "count"),                                                                                        #number of replicate deltas in each group
        dRg_nm_mean=("dRg_nm", "mean"),                                                                               #mean ΔRg across replicates in nm
        dRg_nm_sd=("dRg_nm", "std"),                                                                                  #standard deviation of ΔRg across replicates
    )
    return out                                                                                                        #return summary dataframe with ΔRg mean and SD per (protein, temp_C)


# -----------------------------
# Plotting
# -----------------------------
def plot_mean_Q_vs_T(summary, out_dir, highlight=None):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))                                                                        #create figure with landscape dimensions suited for publication

    proteins = sorted(summary["protein"].unique(), key=lambda x: nice_name(x))                                        #get sorted list of protein names, ordered by their display name

    q_min = []                                                                                                        #collect minimum Q values (with error) for axis scaling
    q_max = []                                                                                                        #collect maximum Q values (with error) for axis scaling

    for p in proteins:                                                                                                 #plot one line per protein variant
        sub = summary[summary["protein"] == p].sort_values("temp_C")                                                  #filter to this protein and sort by temperature for correct line order
        lw = 3.2 if highlight and p in highlight else 2.2                                                             #use a thicker line for highlighted proteins

        y = sub["Q_mean"].to_numpy()                                                                                  #mean Q values as numpy array for plotting
        yerr = sub["Q_sd"].fillna(0).to_numpy()                                                                       #standard deviation for error bars, treating NaN as 0

        q_min.append(np.min(y - yerr))                                                                                #track minimum Q minus error for dynamic y-axis lower bound
        q_max.append(np.max(y + yerr))                                                                                #track maximum Q plus error for dynamic y-axis upper bound

        ax.errorbar(                                                                                                   #plot mean Q with error bars at each temperature
            sub["temp_C"],
            y,
            yerr=yerr,
            marker="o",
            linewidth=lw,
            capsize=3,
            label=nice_name(p)
        )

    # Zoom to data range with a little padding
    ylo = max(0.0, min(q_min) - 0.015)                                                                                #lower y-axis limit: clamp to 0 with a small margin below data
    yhi = min(1.0, max(q_max) + 0.015)                                                                                #upper y-axis limit: clamp to 1 with a small margin above data
    ax.set_ylim(ylo, yhi)                                                                                             #apply the computed y-axis limits

    ax.set_title("Mean Q (native contacts) vs Temperature", pad=12)                                                   #set figure title with padding above the axes
    ax.set_xlabel("Temperature (°C)")                                                                                  #label the x-axis
    ax.set_ylabel("Q (fraction)")                                                                                     #label the y-axis

    ax.legend(                                                                                                        #add legend with clean white background
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
        fontsize=9
    )

    fig.tight_layout()                                                                                                 #adjust subplot parameters to fit all elements within the figure
    fig.savefig(out_dir / "FIG_Q_vs_T.png", bbox_inches="tight")                                                      #save raster PNG for presentations and quick viewing
    fig.savefig(out_dir / "FIG_Q_vs_T.pdf", bbox_inches="tight")                                                      #save vector PDF for publication submission
    plt.close(fig)                                                                                                     #close figure to free memory

def plot_mean_rmsd_vs_T(summary, out_dir, highlight=None):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))                                                                        #create figure with landscape dimensions suited for publication

    proteins = sorted(summary["protein"].unique(), key=lambda x: nice_name(x))                                        #get sorted list of protein names, ordered by their display name

    for p in proteins:                                                                                                 #plot one line per protein variant
        sub = summary[summary["protein"] == p].sort_values("temp_C")                                                  #filter to this protein and sort by temperature for correct line order
        lw = 3.2 if highlight and p in highlight else 2.2                                                             #use a thicker line for highlighted proteins

        ax.plot(                                                                                                       #plot mean Cα RMSD converted to Å vs temperature
            sub["temp_C"],
            nm_to_A(sub["rmsd_nm_mean"]),                                                                             #convert RMSD from nm to Å for display
            marker="o",
            linewidth=lw,
            label=nice_name(p)
        )

    ax.set_title("Mean Cα RMSD vs Temperature", pad=12)                                                               #set figure title with padding above the axes
    ax.set_xlabel("Temperature (°C)")                                                                                  #label the x-axis
    ax.set_ylabel("Cα RMSD (Å)")                                                                                      #label the y-axis with unit in angstroms

    ax.legend(                                                                                                        #add legend with clean white background
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
        fontsize=9
    )

    fig.tight_layout()                                                                                                 #adjust subplot parameters to fit all elements within the figure
    fig.savefig(out_dir / "FIG_RMSD_vs_T.png", bbox_inches="tight")                                                   #save raster PNG for presentations and quick viewing
    fig.savefig(out_dir / "FIG_RMSD_vs_T.pdf", bbox_inches="tight")                                                   #save vector PDF for publication submission
    plt.close(fig)                                                                                                     #close figure to free memory

def plot_disulfide_histogram(
    md_root,
    triple_protein,
    out_dir,
    res1=82,
    res2=130,
    temps=(30, 80),
    prod_fraction=0.5,
):
    """
    Histogram overlay (30C vs 80C) for S82–S130 disulfide distance in the triple mutant.
    Pools all reps found for those temps.
    """
    def collect_distances(tempC):                                                                                      #inner function to collect disulfide distances for a given temperature
        tpat = f"T{tempC}C_rep*"                                                                                      #glob pattern to match all replicate folders at this temperature
        dists_A = []                                                                                                   #list to accumulate distance arrays from all replicates

        for run_dir in sorted((md_root / triple_protein).glob(tpat)):                                                 #iterate over all replicate directories matching the temperature pattern
            ts = run_dir / "analysis" / "timeseries.csv"                                                              #expected path to the timeseries CSV for this replicate
            if not ts.exists():                                                                                        #skip replicates that have no timeseries file
                continue                                                                                               #move on to the next replicate directory
            df = pd.read_csv(ts)                                                                                       #load the timeseries data for this replicate
            df = production_slice(df, prod_fraction=prod_fraction)                                                     #trim to the production window, discarding equilibration frames

            col = find_disulfide_col(df, res1=res1, res2=res2)                                                        #find the column containing the S82–S130 distance
            if col is None:                                                                                            #raise an error if the disulfide column cannot be located
                raise ValueError(
                    f"Could not find disulfide column for {res1}-{res2} in {ts}. "
                    f"Available cols include: {list(df.columns)[:25]} ..."
                )

            vals = df[col].astype(float).values                                                                        #extract distance values as a float array
            if "nm" in col.lower():                                                                                    #if the column is in nanometers, convert to angstroms for display
                vals = nm_to_A(vals)                                                                                   #multiply by 10 to convert nm → Å
            dists_A.append(vals)                                                                                       #add this replicate's distances to the pooled list

        if not dists_A:                                                                                                #if no replicates were found, raise a clear error
            raise FileNotFoundError(f"No timeseries found for {triple_protein} at {tempC}C under {md_root}")
        return np.concatenate(dists_A)                                                                                 #pool all replicate distances into one array

    x30 = collect_distances(temps[0])                                                                                  #collect pooled S82–S130 distances at the lower temperature (30°C)
    x80 = collect_distances(temps[1])                                                                                  #collect pooled S82–S130 distances at the higher temperature (80°C)

    fig, ax = plt.subplots(figsize=(8.8, 5.2))                                                                        #create figure for the histogram overlay

    bins = np.linspace(min(x30.min(), x80.min()), max(x30.max(), x80.max()), 55)                                      #shared bin edges spanning the full range of both distributions

    # Frequency histograms
    ax.hist(x30, bins=bins, alpha=0.5, label=f"{temps[0]}°C")                                                         #plot semi-transparent histogram for the lower temperature
    ax.hist(x80, bins=bins, alpha=0.5, label=f"{temps[1]}°C")                                                         #plot semi-transparent histogram for the higher temperature, overlaid

    ax.set_title(f"S{res1}–S{res2} Disulfide Distance Distribution")                                                  #set figure title with the residue pair
    ax.set_xlabel("S–S distance (Å)")                                                                                  #label the x-axis with unit
    ax.set_ylabel("Frequency")                                                                                         #label the y-axis
    ax.legend(frameon=True)                                                                                            #add legend distinguishing the two temperatures

    lo = max(1.7, min(x30.min(), x80.min()) - 0.05)                                                                   #lower x-axis limit: clamp at 1.7 Å (below typical S–S bond length)
    hi = min(2.6, max(x30.max(), x80.max()) + 0.05)                                                                   #upper x-axis limit: clamp at 2.6 Å (above typical S–S bond length)
    ax.set_xlim(lo, hi)                                                                                                #apply the computed x-axis limits to focus on the disulfide range

    fig.tight_layout()                                                                                                 #adjust layout to prevent clipping
    fig.savefig(out_dir / f"FIG_hist_S{res1}_S{res2}_triple_{temps[0]}C_vs_{temps[1]}C.png", bbox_inches="tight")     #save raster PNG
    fig.savefig(out_dir / f"FIG_hist_S{res1}_S{res2}_triple_{temps[0]}C_vs_{temps[1]}C.pdf", bbox_inches="tight")     #save vector PDF
    plt.close(fig)                                                                                                     #close figure to free memory

def plot_delta_rg_vs_T(drg, out_dir, highlight=None):
    fig, ax = plt.subplots(figsize=(9.5, 5.4))                                                                        #create figure with landscape dimensions suited for publication

    proteins = sorted(drg["protein"].unique(), key=lambda x: nice_name(x))                                            #get sorted list of protein names, ordered by their display name

    for p in proteins:                                                                                                 #plot one line per protein variant
        sub = drg[drg["protein"] == p].sort_values("temp_C")                                                          #filter to this protein and sort by temperature for correct line order
        lw = 3.2 if highlight and p in highlight else 2.2                                                             #use a thicker line for highlighted proteins

        ax.plot(                                                                                                       #plot mean ΔRg converted to Å vs temperature
            sub["temp_C"],
            nm_to_A(sub["dRg_nm_mean"]),                                                                              #convert ΔRg from nm to Å for display
            marker="o",
            linewidth=lw,
            label=nice_name(p)
        )

    ax.axhline(0, linewidth=1.2, alpha=0.6)                                                                           #draw a horizontal reference line at ΔRg = 0 (baseline at 30°C)
    ax.set_title("Mean ΔRg relative to 30°C vs Temperature", pad=12)                                                  #set figure title with padding above the axes
    ax.set_xlabel("Temperature (°C)")                                                                                  #label the x-axis
    ax.set_ylabel("ΔRg relative to 30°C (Å)")                                                                         #label the y-axis with unit and reference temperature

    ax.legend(                                                                                                        #add legend with clean white background
        loc="upper left",
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="0.8",
        fontsize=9
    )

    fig.tight_layout()                                                                                                 #adjust layout to prevent clipping
    fig.savefig(out_dir / "FIG_dRg_vs_T.png", bbox_inches="tight")                                                    #save raster PNG for presentations and quick viewing
    fig.savefig(out_dir / "FIG_dRg_vs_T.pdf", bbox_inches="tight")                                                    #save vector PDF for publication submission
    plt.close(fig)                                                                                                     #close figure to free memory

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()                                                                                     #create argument parser for command-line invocation
    ap.add_argument("--md_root", required=True, help="Path to md/ directory (contains per-protein run folders).")     #required: root directory containing all per-protein MD run subdirectories
    ap.add_argument("--master_csv", required=True, help="Path to analysis_master_summary.csv")                        #required: path to the master summary CSV produced by the analysis pipeline
    ap.add_argument("--out_dir", default="paper_figures", help="Output directory for upgraded figures.")               #output directory for saved figures, defaults to "paper_figures"
    ap.add_argument("--prod_fraction", type=float, default=0.5, help="Last fraction of frames used as production window.") #fraction of trajectory frames to treat as production (discards early equilibration)
    ap.add_argument("--triple_protein", default="FASTPETASE_S54C_Y69C_A82C_A130C_M156C_S166C",
                    help="Folder name under md_root for triple mutant.")                                               #folder name of the triple disulfide mutant used for the S82–S130 histogram
    ap.add_argument("--highlight", default="FASTPETASE_S54C_Y69C_A82C_A130C_M156C_S166C",
                    help="Comma-separated protein names to emphasize (thicker lines).")                                #comma-separated list of protein names to plot with thicker lines for emphasis
    args = ap.parse_args()                                                                                             #parse all command-line arguments into args namespace

    md_root = Path(args.md_root).expanduser().resolve()                                                               #resolve MD root to an absolute path, expanding ~ if present
    master_csv = Path(args.master_csv).expanduser().resolve()                                                         #resolve master CSV to an absolute path, expanding ~ if present
    out_dir = Path(args.out_dir).expanduser().resolve()                                                               #resolve output directory to an absolute path, expanding ~ if present
    ensure_dir(out_dir)                                                                                               #create output directory if it does not already exist

    set_paper_style()                                                                                                  #apply global matplotlib styling for paper-quality figures

    df = load_master(master_csv)                                                                                       #load and canonicalize the master summary CSV
    summary = summarize_by_protein_temp(df)                                                                            #compute mean ± SD for RMSD, Rg, and Q by protein and temperature
    drg = delta_rg_vs_temp(df)                                                                                        #compute per-replicate ΔRg relative to 30°C and summarize

    highlight = [s.strip() for s in args.highlight.split(",") if s.strip()]                                           #parse highlight argument into a list of protein names, stripping whitespace

    plot_mean_Q_vs_T(summary, out_dir, highlight=highlight)                                                           #generate and save mean Q vs temperature figure
    plot_mean_rmsd_vs_T(summary, out_dir, highlight=highlight)                                                        #generate and save mean Cα RMSD vs temperature figure
    plot_delta_rg_vs_T(drg, out_dir, highlight=highlight)                                                             #generate and save mean ΔRg vs temperature figure

    # Histogram for S82–S130 in triple mutant (30 vs 80)
    plot_disulfide_histogram(                                                                                          #generate and save S82–S130 disulfide distance histogram overlay
        md_root=md_root,
        triple_protein=args.triple_protein,
        out_dir=out_dir,
        res1=82, res2=130,
        temps=(30, 80),
        prod_fraction=args.prod_fraction,
    )

    print(f"Done. Wrote upgraded figures to: {out_dir}")                                                              #confirm completion and print output location

if __name__ == "__main__":                                                                                            #only run if script is executed directly, not when imported as a module
    main()                                                                                                            #start the figure generation pipeline
