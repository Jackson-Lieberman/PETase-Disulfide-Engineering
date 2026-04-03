from pathlib import Path                                                                                               #imports
from pdbfixer import PDBFixer                   
from openmm.app import PDBFile                  


BASE = Path(__file__).resolve().parent                                                                                 #absolute path to the directory containing this script
RAW_DIR = BASE / "01_structures_raw"                                                                                   #input directory containing raw PDB files downloaded from AlphaFold or PDB
OUT_DIR = BASE / "02_structures_prepped"                                                                               #output directory where cleaned and hydrogenated PDB files will be saved
OUT_DIR.mkdir(parents=True, exist_ok=True)                                                                             #create the output directory (and any parents) if it doesn't already exist

print("BASE =", BASE)                                                                                                  #print the resolved base directory for verification
print("RAW_DIR =", RAW_DIR)                                                                                            #print the raw input directory path for verification
print("RAW_DIR exists?", RAW_DIR.exists())                                                                             #confirm the raw directory exists before attempting to process files
print("Found PDB files:", len(list(RAW_DIR.glob("*.pdb"))))                                                           #report how many PDB files were found in the raw directory


PH = 7.0                                                                                                               #protonation pH: histidine and other titratable residues are set at this pH
KEEP_WATER = False                                                                                                     #discard crystallographic or model water molecules during cleaning

for pdb_path in sorted(RAW_DIR.glob("*.pdb")):                                                                        #iterate over all raw PDB files in sorted order
    print(f"\n=== Prepping {pdb_path.name} ===")                                                                      #print progress header for this structure

    fixer = PDBFixer(filename=str(pdb_path))                                                                           #load the raw PDB file into PDBFixer for repair

    # Remove waters/ligands unless you explicitly want them
    fixer.removeHeterogens(keepWater=KEEP_WATER)                                                                       #strip all HETATM records (ligands, ions, cofactors); keep water only if KEEP_WATER is True

    # Fix missing heavy atoms
    fixer.findMissingResidues()                                                                                        #identify any residues that are entirely absent from the structure
    fixer.findMissingAtoms()                                                                                           #identify any heavy atoms missing within existing residues
    fixer.addMissingAtoms()                                                                                            #add coordinates for all missing residues and heavy atoms using template geometry

    # Add hydrogens
    fixer.addMissingHydrogens(pH=PH)                                                                                   #add hydrogen atoms with protonation states appropriate for pH 7.0

    # Detect + apply disulfide bonds (native + engineered, if SG–SG is close)
    #fixer.findDisulfideBonds()
    #fixer.applyDisulfideBonds()

    out_file = OUT_DIR / f"{pdb_path.stem}_prepped.pdb"                                                               #construct output path by appending "_prepped.pdb" to the original filename stem
    with open(out_file, "w") as f:                                                                                     #open the output PDB file for writing
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)                                            #write the cleaned structure to PDB, preserving original chain, residue, and atom IDs

    print(f"Saved -> {out_file}")                                                                                      #confirm the file was saved and print the output path
