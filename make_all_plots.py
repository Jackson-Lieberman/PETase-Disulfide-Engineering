#!/usr/bin/env python3
"""
make_all_plots.py

One script that generates *all* your MD graphs:
  A) Per-run timeseries plots (RMSD/Rg/Q/disulfides vs time)
  B) 30°C vs 80°C overlays on same axes + rolling mean
  C) Summary-vs-temperature plots from analysis_master_summary.csv
  D) "Extra" defensible plots:
       1) Core RMSF vs T (trim flexible termini)
       2) Δ-metrics vs T relative to 30°C
       3) Disulfide distance histograms (30 vs 80)
       4) Loop-specific RMSF bars at 80°C (optional: provide loops.json)

Directory assumptions (matches your pipeline):
  md_root/
    PROTEIN/
      T30C_rep1/
        analysis/
          timeseries.csv
          rmsf_ca.csv
          secstruct_window.csv   (optional)
      T80C_rep1/
        analysis/...

Run examples:
  /opt/anaconda3/envs/petase/bin/python make_all_plots.py \\
    --md_root /Users/jacksonlieberman/Desktop/PETase/Mutations/md \\
    --master_csv /Users/jacksonlieberman/Desktop/PETase/Mutations/analysis_master_summary.csv \\
    --out_root plots

If you omit --master_csv, the script will try to find it in:
  1) <md_root>/analysis_master_summary.csv
  2) ./analysis_master_summary.csv
"""

from __future__ import annotations                                                                                     #imports

import argparse                                                                                             
import json                                                                                                    
import os                                                                        
import re                                                               
from pathlib import Path                             
from typing import Dict, List, Optional, Tuple       

import numpy as np                                                                                                
import pandas as pd                                                                                                   
import matplotlib.pyplot as plt                                                                                      

# -----------------------------
# Utility helpers
# -----------------------------
TEMP_RE = re.compile(r"^T(?P<temp>\d+)C_rep(?P<rep>\d+)$")                                                           #compiled regex to parse run directory names like "T30C_rep1" into temp and rep


def safe_mkdir(p: Path) -> None:                                                                                      #creates directory at path p, including any missing parent directories
    p.mkdir(parents=True, exist_ok=True)                                                                              #exist_ok=True prevents error if directory already exists


def nm_to_A(x) -> np.ndarray:                                                                                         #converts nanometer values to angstroms
    return np.asarray(x, dtype=float) * 10.0                                                                          #multiply by 10 and return as a float numpy array


def rolling_mean(series: pd.Series, window: int) -> pd.Series:                                                        #computes a centered rolling mean over a pandas Series
    return series.rolling(window=window, center=True, min_periods=1).mean()                                            #min_periods=1 avoids NaN at the edges of the series


def parse_tempC_from_label(label: str) -> int:                                                                        #parses temperature in °C from a label string like "30C"
    # "30C" -> 30
    return int(str(label).replace("C", "").strip())                                                                    #strip the "C" suffix and convert to integer


def find_run_dirs(md_root: Path) -> List[Tuple[str, int, int, Path]]:
    """
    Returns list of (protein, tempC, rep, run_dir)
    """
    runs = []                                                                                                          #accumulator list for all discovered run tuples
    for protein_dir in sorted([p for p in md_root.iterdir() if p.is_dir()]):                                          #iterate over each protein subdirectory in sorted order
        protein = protein_dir.name                                                                                     #use the directory name as the protein identifier
        for run_dir in sorted([d for d in protein_dir.iterdir() if d.is_dir()]):                                      #iterate over each run subdirectory (e.g. T30C_rep1) in sorted order
            m = TEMP_RE.match(run_dir.name)                                                                            #attempt to parse temperature and replicate from the directory name
            if not m:                                                                                                  #skip directories that don't match the expected naming convention
                continue
            tempC = int(m.group("temp"))                                                                               #extract temperature in °C from the regex match
            rep = int(m.group("rep"))                                                                                  #extract replicate number from the regex match
            runs.append((protein, tempC, rep, run_dir))                                                               #add the parsed run tuple to the list
    return runs                                                                                                        #return the full list of discovered runs


def find_analysis_file(run_dir: Path, filename: str) -> Optional[Path]:
    """
    Tries common locations for analysis files inside a run directory.
    """
    candidates = [                                                                                                     #ordered list of locations to check for the requested file
        run_dir / "analysis" / filename,                                                                              #primary location: run_dir/analysis/filename
        run_dir / filename,                                                                                            #fallback location: directly inside run_dir
    ]
    for c in candidates:                                                                                               #check each candidate path in order
        if c.exists():                                                                                                 #if this candidate file exists, return it immediately
            return c
    # fallback: search shallow
    hits = list(run_dir.glob(f"**/{filename}"))                                                                        #do a recursive search under run_dir for the filename
    return hits[0] if hits else None                                                                                   #return the first match, or None if nothing was found


def auto_disulfide_cols(timeseries_cols: List[str]) -> List[str]:
    """
    Heuristic: disulfide columns in your pipeline look like "S82-S130_nm"
    """
    exclude = {"rmsd_nm", "rg_nm"}                                                                                    #these nm columns are not disulfides; exclude them from the result
    dis = []                                                                                                           #list to collect identified disulfide distance column names
    for c in timeseries_cols:                                                                                          #check each column name in the timeseries
        if c in exclude:                                                                                               #skip RMSD and Rg columns
            continue
        if c.endswith("_nm") and ("-" in c) and (c.lower().startswith("s")):                                          #disulfide columns end in _nm, contain a dash (pair), and start with "s" (for serine/sulfur)
            dis.append(c)                                                                                              #add the column to the disulfide list
    return dis                                                                                                         #return list of identified disulfide distance column names


def nice_ylim(ymin: float, ymax: float, metric: str) -> Tuple[float, float]:
    if ymax <= ymin:                                                                                                   #degenerate case: all values are the same, use a tiny padding
        pad = 1e-6
    else:                                                                                                              #normal case: pad by 5% of the data range
        pad = 0.05 * (ymax - ymin)

    lo, hi = ymin - pad, ymax + pad                                                                                    #apply symmetric padding to lower and upper bounds

    if metric.lower().startswith("q"):                                                                                 #Q is a fraction bounded between 0 and 1
        lo = max(0.0, lo)                                                                                              #clamp lower limit to 0
        hi = min(1.0, hi)                                                                                              #clamp upper limit to 1

    return lo, hi                                                                                                      #return the final (lo, hi) y-axis limits


def analysis_window_timeseries(df: pd.DataFrame, prod_fraction: float) -> pd.DataFrame:
    """
    Keep last `prod_fraction` of frames based on time_ns.
    """
    if "time_ns" not in df.columns:                                                                                    #time_ns is required to trim the trajectory to the production window
        raise ValueError("timeseries.csv missing time_ns")
    tmax = float(df["time_ns"].max())                                                                                  #find the final simulation time in ns
    t0 = prod_fraction * tmax                                                                                          #compute the start time of the production window (e.g. 50% of total)
    return df[df["time_ns"] >= t0].copy()                                                                              #return only frames at or after the production window start time


# -----------------------------
# A) Per-run timeseries plots
# -----------------------------
def plot_timeseries_one(csv_path: Path, outdir: Path, prod_fraction: float, units: str) -> List[Path]:
    """
    Makes one png per metric column in timeseries.csv.
    Returns list of created image Paths.
    """
    df = pd.read_csv(csv_path)                                                                                         #load the timeseries CSV for this run
    if "time_ns" not in df.columns:                                                                                    #time_ns is required as the x-axis
        raise ValueError(f"{csv_path} missing time_ns")

    safe_mkdir(outdir)                                                                                                 #ensure the output directory exists

    t = df["time_ns"].to_numpy()                                                                                       #extract time values as a numpy array for plotting
    t_mark = prod_fraction * float(t.max())                                                                            #compute the time at which the production window begins
    created = []                                                                                                       #accumulator for paths of created PNG files

    # choose columns (prioritize these, then the rest)
    priority = ["rmsd_nm", "rg_nm", "Q_native"]                                                                       #always plot these key metrics first if they exist
    dis_cols = auto_disulfide_cols(list(df.columns))                                                                   #detect any disulfide distance columns present in this run
    cols = [c for c in priority if c in df.columns] + [c for c in dis_cols if c in df.columns]                        #combine priority and disulfide columns, skipping any that are absent

    # also include any secondary structure columns if present
    sec_candidates = ["helix_frac", "sheet_frac", "coil_frac"]                                                        #secondary structure fraction columns to include if all three are present
    if all(c in df.columns for c in sec_candidates):                                                                   #only add secondary structure if the full set of three columns exists
        cols += sec_candidates                                                                                         #append helix, sheet, and coil fraction columns to the plot list

    # optionally include anything else numeric (but not too spammy)
    # Uncomment if you want everything:
    # for c in df.columns:
    #     if c not in cols and c != "time_ns":
    #         if pd.api.types.is_numeric_dtype(df[c]):
    #             cols.append(c)

    for c in cols:                                                                                                     #generate one figure per selected metric column
        y = df[c].astype(float)                                                                                        #extract metric values as floats

        if c.endswith("_nm") and units.lower().startswith("a"):                                                        #convert nm columns to Å if the user requested angstrom units
            y_plot = y * 10.0                                                                                          #multiply by 10 to convert nm to Å
            ylab = f"{c} (Å)"                                                                                          #update y-axis label to reflect angstrom units
        else:                                                                                                          #no conversion needed for this column
            y_plot = y                                                                                                  #use values as-is
            ylab = c                                                                                                    #use the column name as the y-axis label

        plt.figure(figsize=(10, 5))                                                                                    #create a wide figure suitable for timeseries data
        plt.plot(t, y_plot)                                                                                            #plot the metric vs time
        plt.axvline(t_mark, linestyle="--", linewidth=1.0)                                                            #draw a dashed vertical line marking the start of the production window
        plt.xlabel("Time (ns)")                                                                                        #label the x-axis
        plt.ylabel(ylab)                                                                                               #label the y-axis with metric name and unit
        plt.title(f"{c} vs time")                                                                                      #set figure title
        plt.tight_layout()                                                                                             #adjust layout to prevent clipping

        out_png = outdir / f"{c}_vs_time.png"                                                                          #construct output path using the column name
        plt.savefig(out_png, dpi=300)                                                                                  #save the figure at 300 DPI
        plt.close()                                                                                                    #close figure to free memory
        created.append(out_png)                                                                                        #record the created file path

    return created                                                                                                     #return list of all PNG paths created for this run


def make_all_timeseries(md_root: Path, out_root: Path, prod_fraction: float, units: str) -> List[Path]:
    created = []                                                                                                       #accumulator for all created PNG paths across all runs
    for protein, tempC, rep, run_dir in find_run_dirs(md_root):                                                       #iterate over every discovered run
        ts = find_analysis_file(run_dir, "timeseries.csv")                                                             #locate the timeseries CSV for this run
        if not ts:                                                                                                     #skip runs that have no timeseries file
            continue
        outdir = out_root / "timeseries" / protein / f"T{tempC}C_rep{rep}"                                            #construct per-run output directory path
        created += plot_timeseries_one(ts, outdir, prod_fraction=prod_fraction, units=units)                           #generate and collect timeseries plots for this run
    return created                                                                                                     #return list of all created timeseries PNG paths


# -----------------------------
# B) 30°C vs 80°C overlays (per protein)
# -----------------------------
OVERLAY_SETS: Dict[str, List[str]] = {                                                                                #defines which metric columns to include in each named overlay panel set
    # one panel
    "Q": ["Q_native"],                                                                                                 #single-panel overlay showing only native contact fraction
    # two panels
    "Q+RMSD": ["Q_native", "rmsd_nm"],                                                                                #two-panel overlay showing Q and RMSD
    # three panels
    "Q+RMSD+Rg": ["Q_native", "rmsd_nm", "rg_nm"],                                                                    #three-panel overlay showing Q, RMSD, and radius of gyration
}


def _metric_pretty(metric: str) -> Tuple[str, str]:
    """Return (pretty_label, unit_string)."""
    m = metric.strip()                                                                                                 #strip whitespace from the metric name
    if m == "Q_native":                                                                                               #map Q_native to its display name and unit
        return "Q (native contacts)", "fraction"
    if m == "rmsd_nm":                                                                                                 #map rmsd_nm to its display name and unit
        return "Cα RMSD", "Å"
    if m == "rg_nm":                                                                                                   #map rg_nm to its display name and unit
        return "Radius of gyration (Rg)", "Å"
    # fallback
    if m.endswith("_nm"):                                                                                              #for any other _nm column, strip the suffix and use Å as unit
        return m.replace("_nm", ""), "Å"
    return m, ""                                                                                                       #for unrecognized metrics, use the column name as-is with no unit


def _metric_to_plot_units(metric: str, y: pd.Series) -> pd.Series:
    """Convert series into plotting units (Å for *_nm)."""
    if metric.endswith("_nm"):                                                                                         #convert nm columns to Å for display
        return y.astype(float) * 10.0                                                                                  #multiply by 10 to convert nm to Å
    return y.astype(float)                                                                                             #return non-nm columns as float without conversion


def compute_global_overlay_ylims(md_root: Path, rep: int, prod_fraction: float) -> Dict[str, Tuple[float, float]]:
    """Compute global y-limits (shared across proteins) for overlay metrics."""
    vals: Dict[str, List[float]] = {"Q_native": [], "rmsd_nm": [], "rg_nm": []}                                       #accumulate all production-window values for each metric across every protein and temperature

    proteins = [p.name for p in md_root.iterdir() if p.is_dir()]                                                      #get all protein directory names
    for protein in sorted(proteins):                                                                                   #iterate over proteins in sorted order
        for tC in (30, 80):                                                                                            #only compute limits from the 30°C and 80°C runs
            run = md_root / protein / f"T{tC}C_rep{rep}"                                                              #construct path to the specified replicate run at this temperature
            ts_path = find_analysis_file(run, "timeseries.csv")                                                        #locate the timeseries CSV for this run
            if not ts_path:                                                                                            #skip if no timeseries file found
                continue
            df = pd.read_csv(ts_path)                                                                                  #load the timeseries data
            df = analysis_window_timeseries(df, prod_fraction)                                                         #trim to the production window
            for m in list(vals.keys()):                                                                                #collect values for each tracked metric
                if m not in df.columns:                                                                                #skip if this metric is not present in this run
                    continue
                y = _metric_to_plot_units(m, df[m])                                                                   #convert to plot units (Å for nm columns)
                vals[m].extend([float(v) for v in y.to_numpy() if np.isfinite(v)])                                    #append finite values to the global list

    ylims: Dict[str, Tuple[float, float]] = {}                                                                        #output dict mapping metric name to (lo, hi) y-axis limits
    for m, arr in vals.items():                                                                                        #compute y-limits for each metric
        if not arr:                                                                                                    #skip metrics with no data
            continue
        mn = float(min(arr))                                                                                           #global minimum value for this metric
        mx = float(max(arr))                                                                                           #global maximum value for this metric
        if m.startswith("Q"):                                                                                          #Q is a fraction, always use [0, 1] scale for comparability
            ylims[m] = (0.0, 1.0)
        else:                                                                                                          #for other metrics, use padded data range
            ylims[m] = nice_ylim(mn, mx, m)
    return ylims                                                                                                       #return dict of global y-limits for all metrics


def overlay_30_80_for_protein(
    md_root: Path,
    out_root: Path,
    protein: str,
    rep: int,
    window: int,
    prod_fraction: float,
    overlay_sets: List[str],
    ylims: Dict[str, Tuple[float, float]],
) -> List[Path]:
    """Create 30°C vs 80°C overlays for requested metric sets (production window only)."""
    created: List[Path] = []                                                                                           #accumulator for created PNG paths

    run30 = md_root / protein / f"T30C_rep{rep}"                                                                       #path to the 30°C replicate run directory
    run80 = md_root / protein / f"T80C_rep{rep}"                                                                       #path to the 80°C replicate run directory

    ts30_path = find_analysis_file(run30, "timeseries.csv")                                                            #locate 30°C timeseries CSV
    ts80_path = find_analysis_file(run80, "timeseries.csv")                                                            #locate 80°C timeseries CSV
    if not ts30_path or not ts80_path:                                                                                 #both temperatures must have timeseries data to create an overlay
        return created

    df30 = analysis_window_timeseries(pd.read_csv(ts30_path), prod_fraction)                                           #load and trim 30°C timeseries to production window
    df80 = analysis_window_timeseries(pd.read_csv(ts80_path), prod_fraction)                                           #load and trim 80°C timeseries to production window

    # time axis
    t30 = df30["time_ns"].astype(float)                                                                                #30°C time axis in ns
    t80 = df80["time_ns"].astype(float)                                                                                #80°C time axis in ns

    def _plot_one_set(set_name: str, metrics: List[str]) -> None:                                                      #inner function to create one multi-panel overlay figure for a named metric set
        # filter to metrics that exist
        metrics = [m for m in metrics if (m in df30.columns and m in df80.columns)]                                    #only include metrics present in both temperature runs
        if not metrics:                                                                                                 #skip this set if no valid metrics remain
            return

        n = len(metrics)                                                                                               #number of subplot panels (one per metric)
        fig, axes = plt.subplots(n, 1, figsize=(10, 3.2 * n), sharex=True)                                            #create stacked subplots sharing the x-axis
        if n == 1:                                                                                                     #plt.subplots returns a single Axes when n=1, wrap it in a list for uniform handling
            axes = [axes]

        # consistent styling
        raw_alpha = 0.15                                                                                               #transparency for raw (unsmoothed) trace lines
        raw_lw = 0.6                                                                                                   #line width for raw traces
        mean_lw = 1.0                                                                                                  #line width for rolling mean traces

        handles = None                                                                                                 #legend handles, collected from the first subplot
        labels = None                                                                                                  #legend labels, collected from the first subplot

        for ax, metric in zip(axes, metrics):                                                                          #populate each subplot with one metric
            y30 = _metric_to_plot_units(metric, df30[metric])                                                          #convert 30°C metric values to plot units
            y80 = _metric_to_plot_units(metric, df80[metric])                                                          #convert 80°C metric values to plot units

            y30_s = rolling_mean(y30, window)                                                                          #compute rolling mean of 30°C trace
            y80_s = rolling_mean(y80, window)                                                                          #compute rolling mean of 80°C trace

            ax.plot(t30, y30, alpha=raw_alpha, linewidth=raw_lw, label="30°C raw")                                    #plot semi-transparent 30°C raw trace
            ax.plot(t80, y80, alpha=raw_alpha, linewidth=raw_lw, label="80°C raw")                                    #plot semi-transparent 80°C raw trace
            ax.plot(t30, y30_s, linewidth=mean_lw, label=f"30°C mean ({window} frames)")                              #plot 30°C rolling mean on top of raw trace
            ax.plot(t80, y80_s, linewidth=mean_lw, label=f"80°C mean ({window} frames)")                              #plot 80°C rolling mean on top of raw trace

            if metric in ylims:                                                                                        #use the precomputed global y-limits if available for this metric
                ax.set_ylim(*ylims[metric])
            else:                                                                                                      #fall back to data-driven limits if no global limits were computed
                ax.set_ylim(*nice_ylim(float(min(y30.min(), y80.min())), float(max(y30.max(), y80.max())), metric))

            pretty, unit = _metric_pretty(metric)                                                                      #get display name and unit string for this metric
            if unit:                                                                                                   #include unit in the y-axis label if one exists
                ax.set_ylabel(f"{pretty} ({unit})")
            else:                                                                                                      #omit unit from label if there is none
                ax.set_ylabel(pretty)

            if handles is None:                                                                                        #capture legend handles and labels from the first subplot only
                handles, labels = ax.get_legend_handles_labels()

        axes[-1].set_xlabel("Time (ns)")                                                                               #add x-axis label only to the bottom subplot
        fig.suptitle(f"{protein}: 30°C vs 80°C (production window)")                                                  #overall figure title showing protein and temperature comparison
        if handles and labels:                                                                                         #add a shared legend below the figure if handles were collected
            fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
            fig.subplots_adjust(bottom=0.12)                                                                           #make room for the bottom legend
        fig.tight_layout(rect=[0, 0.06, 1, 0.95])                                                                     #adjust layout leaving space for suptitle and bottom legend

        out_png = out_root / "overlays_30_vs_80" / protein / f"{set_name.replace('+','_')}_30C_vs_80C_rep{rep}.png"   #construct output path using protein and set name
        safe_mkdir(out_png.parent)                                                                                     #ensure the parent directory exists
        fig.savefig(out_png, dpi=300)                                                                                  #save the figure at 300 DPI
        plt.close(fig)                                                                                                 #close figure to free memory
        created.append(out_png)                                                                                        #record the created file path

    # make requested sets
    for s in overlay_sets:                                                                                             #generate one figure for each requested overlay set
        if s not in OVERLAY_SETS:                                                                                      #skip unknown set names
            continue
        _plot_one_set(s, OVERLAY_SETS[s])                                                                              #generate the overlay figure for this set

    return created                                                                                                     #return list of all PNG paths created for this protein


def make_all_overlays(
    md_root: Path,
    out_root: Path,
    rep: int,
    window: int,
    prod_fraction: float,
    overlay_sets: List[str],
) -> List[Path]:
    created: List[Path] = []                                                                                           #accumulator for all created overlay PNG paths

    # Compute once so every protein shares identical y-limits.
    ylims = compute_global_overlay_ylims(md_root, rep=rep, prod_fraction=prod_fraction)                                #compute shared y-axis limits from all proteins so figures are directly comparable

    proteins = [p.name for p in md_root.iterdir() if p.is_dir()]                                                      #get all protein directory names
    for protein in sorted(proteins):                                                                                   #iterate over proteins in sorted order
        created += overlay_30_80_for_protein(                                                                          #generate overlays for this protein and collect created paths
            md_root,
            out_root,
            protein,
            rep=rep,
            window=window,
            prod_fraction=prod_fraction,
            overlay_sets=overlay_sets,
            ylims=ylims,
        )
    return created                                                                                                     #return list of all created overlay PNG paths

# -----------------------------
# C) Summary vs Temperature plots
# -----------------------------
def find_master_csv(md_root: Path, master_csv_arg: Optional[str]) -> Path:
    if master_csv_arg:                                                                                                 #if the user explicitly provided a path, use it
        p = Path(master_csv_arg).expanduser().resolve()                                                               #resolve to absolute path, expanding ~ if present
        if not p.exists():                                                                                             #raise a clear error if the specified file does not exist
            raise FileNotFoundError(f"--master_csv not found: {p}")
        return p                                                                                                       #return the verified user-specified path

    c1 = md_root / "analysis_master_summary.csv"                                                                       #first fallback: look for the CSV inside the md_root directory
    c2 = Path("analysis_master_summary.csv").resolve()                                                                 #second fallback: look for the CSV in the current working directory
    for c in [c1, c2]:                                                                                                 #check each fallback location in order
        if c.exists():                                                                                                 #return the first fallback that exists
            return c

    raise FileNotFoundError(                                                                                           #if no CSV was found anywhere, raise a descriptive error
        "Could not find analysis_master_summary.csv. Provide --master_csv or place it in md_root/ or current folder."
    )


def summary_vs_temperature(master_csv: Path, out_root: Path, units: str = "A") -> List[Path]:
    """
    Plots mean RMSD, mean Rg, mean Q, mean RMSF vs temperature per protein using master summary.
    Uses std columns if present.
    """
    df = pd.read_csv(master_csv)                                                                                       #load the master summary CSV
    if not {"protein", "temp_label"}.issubset(df.columns):                                                            #require protein and temp_label columns to proceed
        raise ValueError("master summary missing required columns: protein, temp_label")

    df["tempC"] = df["temp_label"].apply(parse_tempC_from_label)                                                      #parse numeric temperature from the label string (e.g. "30C" → 30)

    # aggregate reps (mean of means, std across reps)
    metrics = [                                                                                                        #list of (mean_col, std_col, display_title) tuples for each metric to plot
        ("mean_rmsd_nm_window", "std_rmsd_nm_window", "Cα RMSD"),
        ("mean_rg_nm_window", "std_rg_nm_window", "Rg"),
        ("mean_Q_window", "std_Q_window", "Q (native contacts)"),
        ("mean_RMSF_CA_nm_window", None, "Mean Cα RMSF"),
    ]

    created = []                                                                                                       #accumulator for created PNG paths
    outdir = out_root / "summary_vs_T"                                                                                 #output directory for summary vs temperature figures
    safe_mkdir(outdir)                                                                                                 #create the output directory if it doesn't exist

    # long-form table for plotting
    agg_rows = []                                                                                                      #list to collect per-metric aggregated dataframes
    for (mcol, scol, _) in metrics:                                                                                    #aggregate each metric by protein and temperature
        if mcol not in df.columns:                                                                                     #skip metrics that are absent from the master CSV
            continue
        grp = df.groupby(["protein", "tempC"])[mcol].agg(["mean", "std"]).reset_index()                               #compute mean and std of the metric across replicates
        grp["metric"] = mcol                                                                                           #tag each row with the metric column name for downstream filtering
        grp.rename(columns={"mean": "mean_val", "std": "std_val"}, inplace=True)                                      #rename to canonical column names
        agg_rows.append(grp)                                                                                           #add this metric's aggregated data to the list

    if not agg_rows:                                                                                                   #if no metrics were found, return without creating any figures
        return created

    tab = pd.concat(agg_rows, ignore_index=True)                                                                      #combine all metrics into a single long-form table
    tab.to_csv(outdir / "summary_vs_T_table.csv", index=False)                                                        #save the aggregated table for reference

    for (mcol, _, title) in metrics:                                                                                   #generate one figure per metric
        sub = tab[tab["metric"] == mcol].copy()                                                                        #filter the long-form table to this metric
        if sub.empty:                                                                                                   #skip if no data for this metric
            continue

        # unit conversion
        factor = 10.0 if (mcol.endswith("_nm_window") and units.lower().startswith("a")) else 1.0                     #convert nm to Å if user requested angstrom units
        unit_str = "Å" if (mcol.endswith("_nm_window") and units.lower().startswith("a")) else ""                     #unit label string for the y-axis
        ylab = f"{title} ({unit_str})".strip()                                                                         #construct y-axis label, stripping trailing space if no unit

        plt.figure(figsize=(10, 6))                                                                                    #create a figure for this metric vs temperature
        for protein in sorted(sub["protein"].unique()):                                                                #plot one line per protein variant
            s2 = sub[sub["protein"] == protein].sort_values("tempC")                                                   #filter and sort this protein's rows by temperature
            x = s2["tempC"].to_numpy()                                                                                 #temperature values for x-axis
            y = (s2["mean_val"] * factor).to_numpy()                                                                   #metric values converted to the requested units
            plt.plot(x, y, marker="o", label=protein)                                                                 #plot mean metric vs temperature for this protein

        plt.xlabel("Temperature (°C)")                                                                                 #label the x-axis
        plt.ylabel(ylab)                                                                                               #label the y-axis with metric name and unit
        plt.title(f"{title} vs Temperature")                                                                           #set the figure title
        plt.legend(fontsize=8)                                                                                         #add legend with small font to fit many protein names
        plt.tight_layout()                                                                                             #adjust layout to prevent clipping

        out_png = outdir / f"{title.replace(' ', '_')}_vs_T.png"                                                      #construct output filename from the metric title
        plt.savefig(out_png, dpi=300)                                                                                  #save the figure at 300 DPI
        plt.close()                                                                                                    #close figure to free memory
        created.append(out_png)                                                                                        #record the created file path

    return created                                                                                                     #return list of all created summary PNG paths


# -----------------------------
# D) Extra defensible plots (4)
# -----------------------------
def core_rmsf_vs_T(md_root: Path, out_root: Path, temps: List[int], core_exclude: int = 20) -> List[Path]:
    """
    Mean RMSF(core) vs temperature, where "core" is defined by trimming ±core_exclude residues.
    Uses rmsf_ca.csv per run. Aggregates reps by mean.
    """
    created = []                                                                                                       #accumulator for created PNG paths
    rows = []                                                                                                          #list to collect per-run core RMSF values

    for protein, tempC, rep, run_dir in find_run_dirs(md_root):                                                       #iterate over every discovered run
        if tempC not in temps:                                                                                         #skip temperatures not in the requested list
            continue
        rmsf_path = find_analysis_file(run_dir, "rmsf_ca.csv")                                                        #locate the per-residue Cα RMSF CSV for this run
        if not rmsf_path:                                                                                              #skip runs that have no RMSF file
            continue
        df = pd.read_csv(rmsf_path)                                                                                    #load the RMSF data for this run
        if not {"resSeq", "rmsf_nm"}.issubset(df.columns):                                                            #require residue sequence number and RMSF columns
            continue

        res = df["resSeq"].astype(int).to_numpy()                                                                      #residue sequence numbers as integer array
        y = df["rmsf_nm"].astype(float).to_numpy()                                                                     #RMSF values in nm as float array

        rmin, rmax = int(res.min()), int(res.max())                                                                    #first and last residue numbers in this chain
        lo, hi = rmin + core_exclude, rmax - core_exclude                                                              #define core region by trimming ±core_exclude residues from each terminus
        core_mask = (res >= lo) & (res <= hi)                                                                          #boolean mask selecting only core residues
        if not np.any(core_mask):                                                                                      #skip if no residues remain in the core after trimming
            continue

        rows.append({                                                                                                  #record RMSF statistics for this run
            "protein": protein,
            "tempC": tempC,
            "rep": rep,
            "mean_rmsf_all_nm": float(np.mean(y)),                                                                    #mean RMSF over all residues
            "mean_rmsf_core_nm": float(np.mean(y[core_mask])),                                                        #mean RMSF over core residues only
        })

    if not rows:                                                                                                       #return early if no valid RMSF data was found
        return created

    tab = pd.DataFrame(rows)                                                                                           #convert collected rows to a dataframe
    outdir = out_root / "extras" / "core_rmsf_vs_T"                                                                   #output directory for core RMSF figures
    safe_mkdir(outdir)                                                                                                 #create output directory if needed
    tab.to_csv(outdir / "core_rmsf_raw.csv", index=False)                                                             #save the raw per-run RMSF table for reference

    # aggregate reps
    agg = tab.groupby(["protein", "tempC"], as_index=False).agg({                                                     #average RMSF values across replicates at each (protein, temperature)
        "mean_rmsf_all_nm": "mean",
        "mean_rmsf_core_nm": "mean",
    })

    plt.figure(figsize=(10, 6))                                                                                        #create figure for core RMSF vs temperature
    for protein in sorted(agg["protein"].unique()):                                                                    #plot one line per protein variant
        s = agg[agg["protein"] == protein].sort_values("tempC")                                                        #filter and sort this protein's rows by temperature
        plt.plot(s["tempC"], nm_to_A(s["mean_rmsf_core_nm"]), marker="o", label=protein)                              #plot core RMSF in Å vs temperature

    plt.xlabel("Temperature (°C)")                                                                                     #label the x-axis
    plt.ylabel(f"Core mean Cα RMSF (Å)  (trim ±{core_exclude} res)")                                                  #label the y-axis indicating the core definition
    plt.title("Core RMSF vs Temperature")                                                                              #set figure title
    plt.legend(fontsize=8)                                                                                             #add legend with small font
    plt.tight_layout()                                                                                                 #adjust layout to prevent clipping

    out_png = outdir / "core_RMSF_vs_T.png"                                                                           #output path for the core RMSF figure
    plt.savefig(out_png, dpi=300)                                                                                      #save at 300 DPI
    plt.close()                                                                                                        #close figure to free memory
    created.append(out_png)                                                                                            #record the created file path
    return created                                                                                                     #return list of created PNG paths


def delta_metrics_vs_T(master_csv: Path, out_root: Path) -> List[Path]:
    """
    ΔQ, ΔRMSD, ΔRg, ΔRMSF relative to 30°C baseline (per protein).
    Uses analysis_master_summary.csv
    """
    created = []                                                                                                       #accumulator for created PNG paths
    df = pd.read_csv(master_csv)                                                                                       #load the master summary CSV
    needed = {"protein", "temp_label", "mean_rmsd_nm_window", "mean_rg_nm_window", "mean_Q_window", "mean_RMSF_CA_nm_window"}
    missing = needed - set(df.columns)                                                                                 #check which required columns are absent
    if missing:                                                                                                        #raise an error listing missing columns
        raise ValueError(f"master summary missing columns for delta plots: {missing}")

    df["tempC"] = df["temp_label"].apply(parse_tempC_from_label)                                                      #parse numeric temperature from the label string

    # mean across reps per protein/temp
    g = df.groupby(["protein", "tempC"], as_index=False).agg({                                                        #average each metric across replicates at each (protein, temperature)
        "mean_rmsd_nm_window": "mean",
        "mean_rg_nm_window": "mean",
        "mean_Q_window": "mean",
        "mean_RMSF_CA_nm_window": "mean",
    })

    base = g[g["tempC"] == 30].set_index("protein")                                                                   #extract 30°C rows indexed by protein as the reference baseline
    if base.empty:                                                                                                     #raise a clear error if no 30°C data exists to use as baseline
        raise ValueError("No 30°C rows in master summary — cannot compute deltas.")

    rows = []                                                                                                          #list to collect per-protein per-temperature delta values
    for _, r in g.iterrows():                                                                                          #iterate over each (protein, temperature) row
        p = r["protein"]                                                                                               #protein name for this row
        if p not in base.index:                                                                                        #skip if no 30°C baseline exists for this protein
            continue
        b = base.loc[p]                                                                                                #retrieve the 30°C baseline values for this protein
        rows.append({                                                                                                  #compute delta values relative to the 30°C baseline
            "protein": p,
            "tempC": int(r["tempC"]),
            "dQ": float(r["mean_Q_window"] - b["mean_Q_window"]),                                                     #change in Q relative to 30°C
            "dRMSD_A": float(nm_to_A([r["mean_rmsd_nm_window"] - b["mean_rmsd_nm_window"]])[0]),                      #change in RMSD in Å relative to 30°C
            "dRg_A": float(nm_to_A([r["mean_rg_nm_window"] - b["mean_rg_nm_window"]])[0]),                            #change in Rg in Å relative to 30°C
            "dRMSF_A": float(nm_to_A([r["mean_RMSF_CA_nm_window"] - b["mean_RMSF_CA_nm_window"]])[0]),                #change in core RMSF in Å relative to 30°C
        })

    tab = pd.DataFrame(rows).sort_values(["protein", "tempC"])                                                        #build delta table sorted by protein then temperature
    outdir = out_root / "extras" / "delta_vs_T"                                                                       #output directory for delta metric figures
    safe_mkdir(outdir)                                                                                                 #create output directory if needed
    tab.to_csv(outdir / "delta_table.csv", index=False)                                                               #save the delta table for reference

    def plot(col: str, ylabel: str, fname: str):                                                                       #inner function to generate one delta metric figure
        plt.figure(figsize=(10, 6))                                                                                    #create figure
        for protein in sorted(tab["protein"].unique()):                                                                #plot one line per protein variant
            s = tab[tab["protein"] == protein].sort_values("tempC")                                                    #filter and sort by temperature
            plt.plot(s["tempC"], s[col], marker="o", label=protein)                                                   #plot delta metric vs temperature
        plt.axhline(0.0, linestyle="--", linewidth=1.0)                                                               #draw a dashed reference line at delta = 0 (the 30°C baseline)
        plt.xlabel("Temperature (°C)")                                                                                 #label the x-axis
        plt.ylabel(ylabel)                                                                                             #label the y-axis
        plt.title(f"{ylabel} relative to 30°C")                                                                       #set figure title
        plt.legend(fontsize=8)                                                                                         #add legend with small font
        plt.tight_layout()                                                                                             #adjust layout to prevent clipping
        out_png = outdir / fname                                                                                       #construct output path
        plt.savefig(out_png, dpi=300)                                                                                  #save at 300 DPI
        plt.close()                                                                                                    #close figure to free memory
        return out_png                                                                                                 #return the created file path

    created.append(plot("dQ", "ΔQ (native contacts)", "dQ_vs_T.png"))                                                 #generate ΔQ vs temperature figure
    created.append(plot("dRMSD_A", "ΔRMSD (Å)", "dRMSD_vs_T.png"))                                                    #generate ΔRMSD vs temperature figure
    created.append(plot("dRg_A", "ΔRg (Å)", "dRg_vs_T.png"))                                                          #generate ΔRg vs temperature figure
    created.append(plot("dRMSF_A", "ΔMean RMSF (Å)", "dRMSF_vs_T.png"))                                               #generate ΔRMSF vs temperature figure
    return created                                                                                                     #return list of all created delta figure paths


def disulfide_histograms(md_root: Path, out_root: Path, prod_fraction: float = 0.5, rep: int = 1) -> List[Path]:
    """
    For each protein that has disulfide distance columns, build histograms for 30C vs 80C.
    Uses timeseries.csv analysis window.
    """
    created = []                                                                                                       #accumulator for created PNG paths
    outdir = out_root / "extras" / "disulfide_hists"                                                                   #output directory for disulfide histogram figures
    safe_mkdir(outdir)                                                                                                 #create output directory if needed

    proteins = [p.name for p in md_root.iterdir() if p.is_dir()]                                                      #get all protein directory names
    for protein in sorted(proteins):                                                                                   #iterate over proteins in sorted order
        run30 = md_root / protein / f"T30C_rep{rep}"                                                                   #path to the 30°C replicate run directory
        run80 = md_root / protein / f"T80C_rep{rep}"                                                                   #path to the 80°C replicate run directory
        ts30_path = find_analysis_file(run30, "timeseries.csv")                                                        #locate 30°C timeseries CSV
        ts80_path = find_analysis_file(run80, "timeseries.csv")                                                        #locate 80°C timeseries CSV
        if not ts30_path or not ts80_path:                                                                             #skip proteins missing either temperature's timeseries
            continue

        ts30 = analysis_window_timeseries(pd.read_csv(ts30_path), prod_fraction)                                       #load and trim 30°C timeseries to production window
        ts80 = analysis_window_timeseries(pd.read_csv(ts80_path), prod_fraction)                                       #load and trim 80°C timeseries to production window

        dis_cols = [c for c in auto_disulfide_cols(list(ts30.columns)) if c in ts80.columns]                           #find disulfide columns present in both temperature runs
        if not dis_cols:                                                                                               #skip proteins with no disulfide distance columns
            continue

        p_out = outdir / protein                                                                                       #per-protein output subdirectory
        safe_mkdir(p_out)                                                                                              #create per-protein output directory if needed

        for col in dis_cols:                                                                                           #generate one histogram per disulfide pair
            y30 = nm_to_A(ts30[col].astype(float).to_numpy())                                                         #30°C disulfide distances converted to Å
            y80 = nm_to_A(ts80[col].astype(float).to_numpy())                                                         #80°C disulfide distances converted to Å

            xmin = float(min(y30.min(), y80.min()))                                                                    #x-axis lower bound spanning both temperature distributions
            xmax = float(max(y30.max(), y80.max()))                                                                    #x-axis upper bound spanning both temperature distributions

            plt.figure(figsize=(10, 6))                                                                                #create figure for this disulfide histogram
            plt.hist(y30, bins=40, alpha=0.5, label="30°C")                                                           #plot semi-transparent 30°C histogram
            plt.hist(y80, bins=40, alpha=0.5, label="80°C")                                                           #plot semi-transparent 80°C histogram overlaid
            plt.xlim(xmin, xmax)                                                                                       #set x-axis limits to span both distributions
            plt.xlabel("SG–SG distance (Å)")                                                                          #label the x-axis
            plt.ylabel("Counts")                                                                                       #label the y-axis
            plt.title(f"{protein}: {col} histogram (analysis window)")                                                 #set figure title with protein and column name
            plt.legend()                                                                                               #add legend distinguishing temperatures
            plt.tight_layout()                                                                                         #adjust layout to prevent clipping

            out_png = p_out / f"{col}_hist_30C_vs_80C.png"                                                            #construct output path using the disulfide column name
            plt.savefig(out_png, dpi=300)                                                                              #save at 300 DPI
            plt.close()                                                                                                #close figure to free memory
            created.append(out_png)                                                                                    #record the created file path

    return created                                                                                                     #return list of all created disulfide histogram paths


def loop_rmsf_bars(md_root: Path, out_root: Path, loops: Dict[str, List[int]], tempC: int = 80) -> List[Path]:
    """
    Loop RMSF bars at a single temperature (default 80C).
    loops: dict like {"LoopA":[60,75], "LoopB":[120,140]}
    """
    created = []                                                                                                       #accumulator for created PNG paths
    rows = []                                                                                                          #list to collect per-run per-loop RMSF values

    # collect per run
    for protein, tC, rep, run_dir in find_run_dirs(md_root):                                                          #iterate over every discovered run
        if tC != tempC:                                                                                                #only process runs at the requested temperature
            continue
        rmsf_path = find_analysis_file(run_dir, "rmsf_ca.csv")                                                        #locate the per-residue Cα RMSF CSV for this run
        if not rmsf_path:                                                                                              #skip runs that have no RMSF file
            continue
        df = pd.read_csv(rmsf_path)                                                                                    #load the RMSF data for this run
        if not {"resSeq", "rmsf_nm"}.issubset(df.columns):                                                            #require residue sequence number and RMSF columns
            continue

        res = df["resSeq"].astype(int).to_numpy()                                                                      #residue sequence numbers as integer array
        y = df["rmsf_nm"].astype(float).to_numpy()                                                                     #RMSF values in nm as float array

        for loop_name, (lo, hi) in loops.items():                                                                      #compute mean RMSF for each user-defined loop region
            mask = (res >= lo) & (res <= hi)                                                                           #boolean mask selecting residues within this loop's range
            if not np.any(mask):                                                                                       #skip if no residues fall within this loop range for this protein
                continue
            rows.append({                                                                                              #record mean loop RMSF for this run
                "protein": protein,
                "loop": loop_name,
                "mean_rmsf_A": float(nm_to_A([np.mean(y[mask])])[0]),                                                 #mean RMSF over the loop residues, converted to Å
            })

    if not rows:                                                                                                       #return early if no loop RMSF data was collected
        # If you didn't provide loops that match your numbering, this will simply skip.
        return created

    tab = pd.DataFrame(rows)                                                                                           #convert collected rows to a dataframe
    outdir = out_root / "extras" / "loop_rmsf"                                                                        #output directory for loop RMSF bar charts
    safe_mkdir(outdir)                                                                                                 #create output directory if needed
    tab.to_csv(outdir / f"loop_rmsf_T{tempC}C_raw.csv", index=False)                                                  #save the raw per-run loop RMSF table for reference

    # plot one bar chart per loop (proteins on x)
    for loop_name in sorted(tab["loop"].unique()):                                                                     #generate one bar chart per loop region
        sub = tab[tab["loop"] == loop_name].groupby("protein", as_index=False)["mean_rmsf_A"].mean()                  #average RMSF across replicates for each protein within this loop
        sub = sub.sort_values("protein")                                                                               #sort proteins alphabetically for consistent bar ordering

        plt.figure(figsize=(10, 5))                                                                                    #create figure for this loop's bar chart
        plt.bar(sub["protein"], sub["mean_rmsf_A"])                                                                    #plot mean loop RMSF as bars with proteins on x-axis
        plt.xticks(rotation=45, ha="right")                                                                           #rotate x-axis labels to prevent overlap
        plt.xlabel("Protein")                                                                                          #label the x-axis
        plt.ylabel("Loop mean Cα RMSF (Å)")                                                                           #label the y-axis
        plt.title(f"{loop_name} — RMSF at {tempC}°C")                                                                 #set figure title with loop name and temperature
        plt.tight_layout()                                                                                             #adjust layout to prevent clipping

        out_png = outdir / f"{loop_name}_RMSF_T{tempC}C.png"                                                          #construct output path using loop name and temperature
        plt.savefig(out_png, dpi=300)                                                                                  #save at 300 DPI
        plt.close()                                                                                                    #close figure to free memory
        created.append(out_png)                                                                                        #record the created file path

    return created                                                                                                     #return list of all created loop RMSF bar chart paths


# -----------------------------
# Orchestrator (single entrypoint)
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="One-stop script to generate all MD plots.")                             #create argument parser with a description
    ap.add_argument("--md_root", required=True, help="Path to Mutations/md folder")                                   #required: root directory containing all per-protein MD run subdirectories
    ap.add_argument("--master_csv", default="", help="Path to analysis_master_summary.csv (optional)")                #optional path to master summary CSV; auto-discovered if omitted
    ap.add_argument("--out_root", default="plots_all", help="Output folder root")                                     #root output directory for all generated figures, defaults to "plots_all"
    ap.add_argument("--make", default="all",
                    choices=["all", "timeseries", "overlays", "summary", "extras"],
                    help="What to generate")                                                                           #select which plot sections to generate; "all" runs everything
    ap.add_argument("--prod_fraction", type=float, default=0.5, help="Analysis window fraction to mark/use (default 0.5)") #fraction of trajectory to treat as production (trims early equilibration)
    ap.add_argument("--units", default="A", choices=["A", "nm"], help="Units for nm metrics (A or nm)")               #display units for distance/RMSF metrics: angstroms or nanometers
    ap.add_argument("--overlay_rep", type=int, default=1, help="Replicate number for overlays/hists (default 1)")     #which replicate number to use for 30 vs 80 overlays and histograms
    ap.add_argument("--overlay_window", type=int, default=100, help="Rolling mean window (frames) for overlays")      #number of frames to use in the rolling mean for overlay smoothing
    ap.add_argument("--overlay_sets", default="Q,Q+RMSD,Q+RMSD+Rg",
                    help="Comma-separated overlay sets to make: Q, Q+RMSD, Q+RMSD+Rg (default all three)")            #which multi-panel overlay combinations to generate
    ap.add_argument("--temps", default="30,40,50,60,70,80", help="Temps (C) for core RMSF vs T (comma list)")         #comma-separated list of temperatures to include in the core RMSF vs T plot
    ap.add_argument("--core_exclude", type=int, default=20, help="Trim ±N residues for 'core' RMSF")                  #number of terminal residues to exclude when computing core RMSF
    ap.add_argument("--loops_json", default="", help="Optional loops.json with {loopName:[lo,hi],...} for loop bars") #optional JSON file defining named loop regions for the loop RMSF bar charts
    args = ap.parse_args()                                                                                             #parse all command-line arguments into args namespace

    md_root = Path(args.md_root).expanduser().resolve()                                                               #resolve MD root to an absolute path, expanding ~ if present
    if not md_root.exists():                                                                                           #raise a clear error if the MD root directory does not exist
        raise FileNotFoundError(f"md_root not found: {md_root}")

    out_root = Path(args.out_root).expanduser().resolve()                                                             #resolve output root to an absolute path, expanding ~ if present
    safe_mkdir(out_root)                                                                                               #create output root directory if it doesn't exist

    created_all: List[Path] = []                                                                                       #accumulator for all created PNG paths across every section

    # Discover master csv only if needed
    master_csv: Optional[Path] = None                                                                                  #initialize master CSV path; only loaded if summary or extras sections are requested
    if args.make in ("all", "summary", "extras"):                                                                      #only locate the master CSV when it will actually be used
        master_csv = find_master_csv(md_root, args.master_csv if args.master_csv else None)                           #find the master CSV at the user-specified path or auto-discover it

    temps = [int(x.strip()) for x in args.temps.split(",") if x.strip()]                                              #parse the comma-separated temperature list into a list of integers

    loops = None                                                                                                       #initialize loops dict; only populated if a loops JSON was provided
    if args.loops_json:                                                                                                #only load the loops JSON if the argument was provided
        loops_path = Path(args.loops_json).expanduser().resolve()                                                      #resolve loops JSON path to an absolute path
        if not loops_path.exists():                                                                                    #raise a clear error if the loops JSON file doesn't exist
            raise FileNotFoundError(f"loops_json not found: {loops_path}")
        loops = json.loads(loops_path.read_text())                                                                     #parse the loops JSON into a dict mapping loop name to [lo, hi] residue range

    # Run selected steps
    if args.make in ("all", "timeseries"):                                                                             #section A: per-run timeseries plots
        print("\n[1/4] Making per-run timeseries plots...")
        created_all += make_all_timeseries(md_root, out_root, prod_fraction=args.prod_fraction, units=args.units)      #generate all per-run timeseries figures

    if args.make in ("all", "overlays"):                                                                               #section B: 30 vs 80°C overlay figures
        print("\n[2/4] Making 30°C vs 80°C overlays...")
        created_all += make_all_overlays(md_root, out_root, rep=args.overlay_rep, window=args.overlay_window, prod_fraction=args.prod_fraction, overlay_sets=[s.strip() for s in args.overlay_sets.split(",") if s.strip()])  #generate all overlay figures with shared y-limits

    if args.make in ("all", "summary") and master_csv:                                                                #section C: summary vs temperature plots (requires master CSV)
        print("\n[3/4] Making summary-vs-temperature plots...")
        created_all += summary_vs_temperature(master_csv, out_root, units=args.units)                                  #generate RMSD/Rg/Q/RMSF vs temperature summary figures

    if args.make in ("all", "extras") and master_csv:                                                                  #section D: extra defensible plots (requires master CSV)
        print("\n[4/4] Making extra plots (core RMSF, deltas, hists, loops)...")
        created_all += core_rmsf_vs_T(md_root, out_root, temps=temps, core_exclude=args.core_exclude)                 #D1: core RMSF vs temperature
        created_all += delta_metrics_vs_T(master_csv, out_root)                                                       #D2: delta metrics vs temperature relative to 30°C
        created_all += disulfide_histograms(md_root, out_root, prod_fraction=args.prod_fraction, rep=args.overlay_rep) #D3: disulfide distance histograms for 30 vs 80°C
        if loops:                                                                                                      #D4: loop RMSF bars (only if loops JSON was provided)
            created_all += loop_rmsf_bars(md_root, out_root, loops=loops, tempC=80)
        else:                                                                                                          #notify the user that loop bars were skipped
            print("  (Loop RMSF bars skipped — provide --loops_json if you want them)")

    # Write a manifest of created images
    manifest = out_root / "plots_manifest.csv"                                                                         #path to the output manifest CSV listing all created PNG files
    img_rows = [{"path": str(p)} for p in created_all if str(p).lower().endswith(".png")]                             #collect paths of all created PNG files
    pd.DataFrame(img_rows).to_csv(manifest, index=False)                                                              #write the manifest CSV with one row per created image
    print(f"\nDone. Created {len(img_rows)} images.")                                                                  #report total number of images created
    print(f"Manifest: {manifest}")                                                                                     #report the path to the manifest file


if __name__ == "__main__":                                                                                             #only run if script is executed directly, not when imported as a module
    main()                                                                                                             #start the plot generation pipeline
