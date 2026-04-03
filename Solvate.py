from pathlib import Path                                                                                               #imports
from openmm.app import PDBFile, Modeller, ForceField                                                                 
from openmm.unit import nanometer, molar              

PREP_DIR = Path("02_structures_prepped")                                                                               #sets the path to find the prepped structures
OUT_DIR  = Path("03_structures_solvated")                                                                              #sets the path that the solvated structures will be in
OUT_DIR.mkdir(parents=True, exist_ok=True)                                                                             #makes the folder for the solvated structures

forcefield = ForceField("amber14-all.xml", "amber14/tip3pfb.xml")                                                      #sets force fields: AMBER14 all-atom parameters with TIP3P-FB water model

IONIC_STRENGTH = 0.15 * molar                                                                                          #physiological salt concentration (150 mM NaCl) added to the solvent box

# Default padding for most systems
DEFAULT_PADDING_NM = .75                                                                                               #minimum distance in nm between the protein and the periodic box edge

for pdb_path in sorted(PREP_DIR.glob("*_prepped.pdb")):                                                               #iterate over all prepped PDB files in sorted order
    padding_nm = 0.75                                                                                                  #set water box padding to 0.75 nm for this structure

    print(f"\n=== Solvating {pdb_path.name} (padding={padding_nm} nm) ===")                                           #print progress header for this structure
    pdb = PDBFile(str(pdb_path))                                                                                       #load the prepped PDB file into OpenMM
    modeller = Modeller(pdb.topology, pdb.positions)                                                                   #create a modeller object from the PDB topology and atomic positions

    modeller.addSolvent(                                                                                               #add explicit water and ions to create a periodic solvated system
        forcefield,
        padding=padding_nm * nanometer,                                                                                #set the minimum padding between protein and box edge in nm
        ionicStrength=IONIC_STRENGTH,                                                                                  #add NaCl ions to reach 150 mM ionic strength
    )

    out_path = OUT_DIR / pdb_path.name.replace("_prepped.pdb", "_solv.pdb")                                           #construct output path by replacing "_prepped.pdb" suffix with "_solv.pdb"
    with open(out_path, "w") as f:                                                                                     #open the output PDB file for writing
        PDBFile.writeFile(modeller.topology, modeller.positions, f, keepIds=True)                                      #write the solvated structure to PDB, preserving original atom and residue IDs

    n_atoms = sum(1 for _ in modeller.topology.atoms())                                                                #count total atoms in the solvated system (protein + water + ions)
    print(f"Saved -> {out_path} ({n_atoms} atoms)")                                                                    #print the output path and total atom count as a confirmation
