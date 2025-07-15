# Neural Iterative Selection Expansion (NISE) using LASErMPNN and Boltz-1x/Boltz-2x

### Check out the paper [here](https://www.biorxiv.org/content/10.1101/2025.04.22.649862v1)!

Jointly optimize the sequence and structure of a protein-ligand binding pose with iterative selection-expansion.

### Installing NISE Environment

To run NISE, install the conda environment for LASErMPNN and a separate conda environment which contains Boltz-1x or Boltz-2x:

1) Install [LASErMPNN](https://github.com/polizzilab/LASErMPNN) conda environment

2) Install ProDy from source into lasermpnn conda environment. 
There is currently an issue with the conda installable ProDy and distance-based selections which are used within NISE. 
This can be resolved for now by installing ProDy from source.

Just follow this set of commands inside the NISE project directory after `git clone`-ing the project and installing the lasermpnn environment.
```bash
git submodule update --init --recursive

conda activate lasermpnn

git clone git@github.com:prody/ProDy.git
cd ProDy
python setup.py build_ext --inplace --force
pip install -Ue .

cd ..
tar -xvf hetdict.tar.gz
```

3) Activate your conda environment containing Boltz-1x or Boltz-2x and run `which boltz` to get the path to the executable you call when running `boltz predict` commands. 
You will need to update this path in `run_nise_boltz1x.py` or `run_nise_boltz2x.py` respectively.


### Generating input poses:

While we would recommend following the protocol using COMBS to generate initial poses as outlined in our paper, decent starting poses may be generated using the workflow outlined [here using CARPdock,](https://github.com/benf549/CARPdock) though this has not yet been experimentally validated.
Initializations from [BoltzDesign1](https://github.com/yehlincho/BoltzDesign1) will almost certainly work as well, but this remains untested.

### Running NISE:

1) Create a PDB file containing your PROTONATED input ligand with CONECT records encoding bonds:
If you have a non-protonated ligand/are missing conect records, run `protonate_and_add_conect_records.py {input_path}.pdb {smiles_string} {output_path}.pdb`.
WARNING: This will rename the ligand atoms, ligand chain, and resnum.


2) [Optional] If you want to protonate using reduce (keeps added ligand hydrogen names consistent with input, a bit more finicky than the alternative RDKit), Inject your ligand into REDUCE hetdict by running `inject_ligand_into_hetdict.py {output_path}.pdb`


3) Create an input directory with a subfolder called input_backbones. Ex: `./debug/input_backbones/`.


4) Update the params dictionary at the bottom of `./run_nise_boltz1x.py` with the path to your new input dir ex: (`input_dir = Path('./debug/')`).

6) Update burial and RMSD atom sets and smiles string in `./run_nise_boltz1x.py`

7) Update `boltz1x_executable_path` at bottom of `./run_nise_boltz1x.py`

8) If you want to constrain the number of alanine and glycine residues predicted on the surface of the protein in secondary-structured regions, run `identify_surface_residues.ipynb` and update 'budget_residue_sele_string' in the params dictionary at the bottom of the run_nise_boltz script.

To test out an example run:

```bash

# Protonated smiles string from ChemDraw.
./protonate_and_add_conect_records.py ./example_pdbs/16_pose26_en_-5p044_no_CG_top1_of_1_n4_00374_looped_master_6_gly_0001_trim_H_98.pdb "CC[C@]1(O)C2=C(C(N3CC4=C5[C@@H]([NH3+])CCC6=C5C(N=C4C3=C2)=CC(F)=C6C)=O)COC1=O" ./example_pdbs/test_input_protonated_conect.pdb

mkdir -p ./debug/input_backbones/

cp ./example_pdbs/test_input_protonated_conect.pdb ./debug/input_backbones/

./run_nise_boltz1x.py
```
