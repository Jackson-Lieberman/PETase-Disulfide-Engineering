#!/usr/bin/env python3
from __future__ import annotations                                                                                     #imports

import argparse
from pathlib import Path
from datetime import datetime

from openmm import unit, Platform
import openmm as mm
from openmm.app import (
    PDBFile,
    ForceField,
    Modeller,
    Simulation,
    DCDReporter,
    StateDataReporter,
    CheckpointReporter,
    HBonds,
    PME,
)


def count_disulfides(topology) -> int:
    """Counts SG–SG bonds (disulfides) present in the topology."""
    sg_atoms = set()                                                                                                   #creates set for disulfide atoms
    for atom in topology.atoms():                                                                                      #for every atom
        if atom.name == "SG":                                                                                          #check if the atom is a sulfur
            sg_atoms.add(atom)                                                                                         #if it is then add the atom to the set

    ss = 0                                                                                                             #initialize disulfide bond counter
    for a1, a2 in topology.bonds():                                                                                    #loop through all bonds
        if (a1 in sg_atoms) and (a2 in sg_atoms):                                                                      #if both atoms in the bond are sulfur, then we have a disulfide
            ss += 1                                                                                                    #add one to ss variable
    return ss                                                                                                          #return total count of disulfide bonds


def pick_platform(requested: str | None) -> Platform:
    if requested:                                                                                                      #if the user requested a specific platform, try to use it (will raise if not available)
        return Platform.getPlatformByName(requested)                                                                   #return the the requested platform

    # Default: try common fast options first
    for name in ["CUDA", "OpenCL", "CPU"]:                                                                             #if platform not specified, then try to use each platform in order
        try:                                                                                                           #tries to get that platform
            return Platform.getPlatformByName(name)                                                                    #if successful, return it
        except Exception:                                                                                              #if not
            pass                                                                                                       #just go to the next
    return Platform.getPlatformByName("CPU")                                                                           #if all else fails, use CPU


def next_cont_suffix(out_dir: Path) -> str:
    """
    Choose a non-overwriting suffix for continuation outputs:
    traj_cont1.dcd, log_cont1.csv, ...
    """
    for i in range(1, 1000):                                                                                           #for suffix numbers 1-999
        if not (out_dir / f"traj_cont{i}.dcd").exists() and not (out_dir / f"log_cont{i}.csv").exists():               #check whether both the trajectory continuation file and log continuation file for that suffix do not already exist.
            return f"_cont{i}"                                                                                         #if it is a new suffix, return it
    return "_cont999"                                                                                                  #if all suffixes up to 999 already exist, return _cont999


def main():
    parser = argparse.ArgumentParser(description="OpenMM MD runner with checkpoint resume.")                           #creates argument parser with a description
    parser.add_argument("--pdb", type=str, required=True, help="Path to *_solv.pdb")                                   #required: path to the input solvated PDB file
    parser.add_argument("--out", type=str, required=True, help="Output directory for this run")                        #required: path to output directory
    parser.add_argument("--temp", type=float, default=300.0, help="Temperature (K)")                                   #simulation temperature in Kelvin, default 300 K
    parser.add_argument("--ns", type=float, default=1.0, help="Production length (ns)")                                #production run length in nanoseconds, default 1 ns
    parser.add_argument("--eq_ps", type=float, default=200.0, help="Equilibration length (ps), NPT")                   #equilibration length in picoseconds, default 200 ps
    parser.add_argument("--timestep_fs", type=float, default=2.0, help="Timestep (fs)")                                #integration timestep in femtoseconds, default 2 fs
    parser.add_argument("--friction", type=float, default=1.0, help="Langevin friction (1/ps)")                        #Langevin thermostat friction coefficient in 1/ps
    parser.add_argument("--pressure_atm", type=float, default=1.0, help="Pressure (atm)")                              #target pressure in atmospheres for NPT barostat
    parser.add_argument("--report_ps", type=float, default=1.0, help="Report interval (ps)")                           #how often to write log/trajectory output, in picoseconds
    parser.add_argument("--checkpoint_ps", type=float, default=10.0, help="Checkpoint interval (ps)")                  #how often to save a checkpoint file, in picoseconds
    parser.add_argument("--seed", type=int, default=1, help="Random seed (fresh starts only)")                         #random seed used for velocity initialization on fresh runs
    parser.add_argument("--platform", type=str, default=None, help="CPU, OpenCL, CUDA (optional)")                     #optionally override the compute platform
    parser.add_argument(
        "--traj_during_eq",
        action="store_true",
        help="If set, writes trajectory during equilibration too (default: only production).",
    )                                                                                                                  #flag to also record trajectory frames during equilibration
    args = parser.parse_args()                                                                                         #parse all command-line arguments into args namespace

    pdb_path = Path(args.pdb).resolve()                                                                                #resolve PDB file path to an absolute path
    out_dir = Path(args.out).resolve()                                                                                 #resolve output directory to an absolute path
    out_dir.mkdir(parents=True, exist_ok=True)                                                                         #create output directory (and any parents) if it doesn't exist

    # If already finished, do nothing (nice for re-running batch scripts)
    final_pdb_path = out_dir / "final.pdb"                                                                             #path where final coordinates will be saved at end of run
    if final_pdb_path.exists() and final_pdb_path.stat().st_size > 0:                                                  #if final.pdb already exists and is non-empty, the run already completed
        print(f"SKIP (already finished): {out_dir}")                                                                   #notify user that this run is being skipped
        return                                                                                                         #exit without re-running the simulation

    # Units
    T = args.temp * unit.kelvin                                                                                        #temperature as an OpenMM quantity with kelvin units
    P = args.pressure_atm * unit.atmosphere                                                                            #pressure as an OpenMM quantity with atmosphere units
    dt = args.timestep_fs * unit.femtoseconds                                                                          #timestep as an OpenMM quantity with femtosecond units
    friction = args.friction / unit.picosecond                                                                         #friction coefficient as an OpenMM quantity with 1/picosecond units

    # Step counts (define BEFORE reporters)
    eq_steps = int((args.eq_ps * unit.picoseconds) / dt)                                                               #convert equilibration time to integer number of timesteps
    prod_steps = int((args.ns * unit.nanoseconds) / dt)                                                                #convert production time to integer number of timesteps
    total_steps = eq_steps + prod_steps                                                                                #total steps across both equilibration and production

    report_steps = max(1, int((args.report_ps * unit.picoseconds) / dt))                                               #steps between each log/trajectory output (floor at 1)
    chk_steps = max(1, int((args.checkpoint_ps * unit.picoseconds) / dt))                                              #steps between each checkpoint save (floor at 1)

    # Paths
    chk_path = out_dir / "checkpoint.chk"                                                                              #path to the checkpoint file used for resuming

    # Resume?
    resume = chk_path.exists() and chk_path.stat().st_size > 0                                                         #true if a valid non-empty checkpoint file exists from a prior run
    suffix = next_cont_suffix(out_dir) if resume else ""                                                               #generate a continuation suffix for output files if resuming, otherwise empty

    traj_path = out_dir / f"traj{suffix}.dcd"                                                                          #path to trajectory DCD output file
    log_path = out_dir / f"log{suffix}.csv"                                                                            #path to thermodynamic log CSV output file

    # --- Load solvated coordinates/topology
    pdb = PDBFile(str(pdb_path))                                                                                       #load the solvated PDB file into OpenMM
    ss_count = count_disulfides(pdb.topology)                                                                          #count number of disulfide bonds present in the topology

    # --- Build system (must match your solvation forcefield)
    forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")                                                  #load AMBER14 forcefield with TIP3P-FB water model
    modeller = Modeller(pdb.topology, pdb.positions)                                                                   #create modeller object from the PDB topology and positions

    # Ensure consistent protonation/H atoms (adds missing H only)
    modeller.addHydrogens(forcefield, pH=7.0)                                                                          #add any missing hydrogen atoms at physiological pH 7.0

    system = forcefield.createSystem(                                                                                  #create the OpenMM system with the specified force field settings
        modeller.topology,
        nonbondedMethod=PME,                                                                                           #particle mesh ewald for accurate long-range electrostatics
        nonbondedCutoff=1.0 * unit.nanometer,                                                                          #1 nm real-space cutoff for nonbonded interactions
        constraints=HBonds,                                                                                            #constrain all bonds involving hydrogen atoms
        rigidWater=True,                                                                                               #treat water molecules as rigid bodies (no internal DOF)
        ewaldErrorTolerance=1e-4,                                                                                      #target relative error tolerance for PME calculation
    )

    # NPT (pressure coupling)
    system.addForce(mm.MonteCarloBarostat(P, T, 25))                                                                   #add Monte Carlo barostat for NPT ensemble, attempting pressure change every 25 steps

    # Integrator
    integrator = mm.LangevinMiddleIntegrator(T, friction, dt)                                                          #create Langevin middle integrator for constant-temperature dynamics
    integrator.setConstraintTolerance(1e-5)                                                                            #set numerical tolerance for satisfying bond constraints

    # Platform
    platform = pick_platform(args.platform)                                                                            #select the best available (or user-specified) compute platform

    # Simulation
    simulation = Simulation(modeller.topology, system, integrator, platform)                                           #assemble simulation object from topology, system, integrator, and platform
    simulation.context.setPositions(modeller.positions)                                                                #set initial atomic positions in the simulation context

    # --- Header
    print("\n" + "=" * 78)
    print("MD RUN")
    print("PDB:", pdb_path.name)
    print("OUT:", out_dir)
    print(f"T = {args.temp:.2f} K | P = {args.pressure_atm:.2f} atm | dt = {args.timestep_fs:.2f} fs")
    print(f"Equil = {args.eq_ps:.2f} ps ({eq_steps} steps) | Prod = {args.ns:.3f} ns ({prod_steps} steps)")
    print(f"Report every {args.report_ps:.2f} ps ({report_steps} steps)")
    print(f"Checkpoint every {args.checkpoint_ps:.2f} ps ({chk_steps} steps)")
    print("Platform:", platform.getName())
    print("SG–SG disulfides in topology:", ss_count)
    print("Resume:", resume)
    if resume:
        print("Checkpoint:", chk_path)
        print("Continuation suffix:", suffix)
    print("=" * 78 + "\n")

    simulation.reporters.append(                                                                                       #add state data reporter to log thermodynamic quantities to CSV
        StateDataReporter(
            str(log_path),                                                                                             #write output to the log CSV file
            report_steps,                                                                                              #write a row every report_steps steps
            step=True,                                                                                                 #log current step number
            time=True,                                                                                                 #log simulation time in ps
            potentialEnergy=True,                                                                                      #log potential energy in kJ/mol
            kineticEnergy=True,                                                                                        #log kinetic energy in kJ/mol
            totalEnergy=True,                                                                                          #log total energy in kJ/mol
            temperature=True,                                                                                          #log instantaneous temperature in K
            density=True,                                                                                              #log system density in g/mL
            volume=True,                                                                                               #log periodic box volume in nm^3
            progress=True,                                                                                             #log percent completion of simulation
            remainingTime=True,                                                                                        #log estimated time remaining
            speed=True,                                                                                                #log simulation speed in ns/day
            totalSteps=total_steps,                                                                                    #total steps needed to compute progress and remaining time
            separator=",",                                                                                             #use comma delimiter for CSV format
        )
    )
    simulation.reporters.append(CheckpointReporter(str(chk_path), chk_steps))                                          #add checkpoint reporter to save full simulation state every chk_steps steps

    # Trajectory reporting strategy:
    # - Fresh runs: by default only record production (saves space), unless --traj_during_eq set.
    # - Resume runs: always record to a new traj_cont*.dcd so you capture the continuation.
    if resume or args.traj_during_eq:                                                                                  #attach DCD reporter now if resuming or user requested trajectory during equilibration
        simulation.reporters.append(DCDReporter(str(traj_path), report_steps))                                         #write atomic coordinates to DCD file every report_steps steps

    # --- Start / Resume
    if resume:                                                                                                         #if restarting from an existing checkpoint
        print(f"Loading checkpoint: {chk_path}")
        simulation.loadCheckpoint(str(chk_path))                                                                       #restore positions, velocities, and box vectors from the checkpoint file
    else:                                                                                                              #if this is a fresh run with no prior checkpoint
        print("Minimizing...")
        simulation.minimizeEnergy(maxIterations=2000)                                                                  #energy minimize to remove atomic clashes before starting dynamics
        print("Initializing velocities...")
        simulation.context.setVelocitiesToTemperature(T, args.seed)                                                    #assign random velocities drawn from Maxwell-Boltzmann distribution at temperature T

    # Determine where we are
    cur = simulation.currentStep                                                                                       #get current step count (0 for fresh run, or resumed step for checkpoint)
    print(f"Current step: {cur} / {total_steps}")

    # Continue equilibration if needed
    if cur < eq_steps:                                                                                                 #check if equilibration phase still has steps remaining
        rem_eq = eq_steps - cur                                                                                        #calculate number of remaining equilibration steps
        # If we DIDN'T attach DCDReporter yet (fresh run and not traj during eq), keep it that way.
        print(f"Equilibrating (NPT) remaining: {rem_eq} steps ({rem_eq * dt})")
        simulation.step(rem_eq)                                                                                        #run the remaining equilibration steps under NPT conditions

    # Attach trajectory reporter for production if we delayed it (fresh run, no traj during eq)
    if (not resume) and (not args.traj_during_eq):                                                                     #for fresh runs without equilibration trajectory, start recording now at production
        # We are now at the start of production, so start trajectory output here.
        simulation.reporters.append(DCDReporter(str(traj_path), report_steps))                                         #attach DCD reporter at the start of production phase

    # Continue production if needed
    cur = simulation.currentStep                                                                                       #get current step count after equilibration is complete
    if cur < total_steps:                                                                                              #check if production phase still has steps remaining
        rem = total_steps - cur                                                                                        #calculate number of remaining production steps
        print(f"Production remaining: {rem} steps ({rem * dt})")
        simulation.step(rem)                                                                                           #run the remaining production steps

    # Save final coordinates
    state = simulation.context.getState(getPositions=True)                                                             #retrieve final simulation state including atomic positions
    with open(final_pdb_path, "w") as f:                                                                               #open the final PDB file for writing
        PDBFile.writeFile(simulation.topology, state.getPositions(), f, keepIds=True)                                  #write final structure to PDB, preserving original atom and residue IDs

    print("\nDone.")
    print("Wrote:", traj_path)
    print("Wrote:", log_path)
    print("Wrote:", chk_path)
    print("Wrote:", final_pdb_path)


if __name__ == "__main__":                                                                                             #only run if script is executed directly, not when imported as a module
    main()                                                                                                             #start the simulation
