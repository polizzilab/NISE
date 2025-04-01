# Neural Iterative Selection Expansion (NISE)

Jointly optimize the sequence and structure of a protein-ligand binding pose with iterative selection-expansion.

### Installing NISE Environment

The NISE conda environment contains dependencies for both LASErMPNN and Boltz-1:

1) Install conda environment

2) Install ProDy from source. 
There is currently an issue with the conda installable ProDy and distance-based selections which are used within NISE. 
This can be resolved for now by installing ProDy from source.

3) Install [REDUCE](git@github.com:rlabduke/reduce.git) to use with our provided hetdict and hetdict modification code.


Just follow this set of commands inside the NISE project directory after `git clone`-ing the project.
```bash
conda env create -f conda_env.yml
conda activate nise

git clone git@github.com:prody/ProDy.git
cd ProDy
python setup.py build_ext --inplace --force
pip install -Ue .

tar -xvf hetdict.tar.gz
```

### Running NISE:

1) Create a PDB file containing your PROTONATED input ligand with CONECT records encoding bonds:
If you have a non-protonated ligand/are missing conect records, run `protonate_and_add_conect_records.py {input_path}.pdb {smiles_string} {output_path}.pdb`.
WARNING: This will rename the ligand atoms, ligand chain, and resnum.


2) Inject your ligand into REDUCE hetdict by running `inject_ligand_into_hetdict.py {output_path}.pdb`


3) Create an input directory with a subfolder called input_backbones. Ex: `./debug/input_backbones/`.


4) Update the params dictionary at the bottom of `./run_nise.py` with the path to your new input dir ex: (`input_dir = Path('./debug/input_backbones/'`).


5) Update burial and RMSD atom sets and smiles string in `./run_nise.py`

To test out an example run:

```bash

./protonate_and_add_conect_records.py ./example_pdbs/16_pose26_en_-5p044_no_CG_top1_of_1_n4_00374_looped_master_6_gly_0001_trim_H_98.pdb "CC[C@]1(O)C2=C(C(N3CC4=C5[C@@H]([NH3+])CCC6=C5C(N=C4C3=C2)=CC(F)=C6C)=O)COC1=O" ./example_pdbs/test_input_protonated_conect.pdb

./inject_ligand_into_hetdict.py ./example_pdbs/test_input_protonated_conect.pdb

mkdir -p ./debug/input_backbones/

cp ./example_pdbs/test_input_protonated_conect.pdb ./debug/input_backbones/

./run_nise.py
```


### TODO:

- [ ] checkout boltz-ext repo to improve ligand conformers.
