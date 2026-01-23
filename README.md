[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/polizzilab/NISE/blob/main/NISE_LASErMPNN.ipynb)

# Neural Iterative Selection Expansion (NISE) using LASErMPNN/LigandMPNN with Boltz-1x/Boltz-2x

![A NISE Trajectory demonstraing optimization of P(bind) and predicted affinity from Boltz-2](./images/boltz2_animation.gif)

### Introduced in the paper [Zero-shot design of drug-binding proteins via neural selection-expansion](https://www.biorxiv.org/content/10.1101/2025.04.22.649862v1)!

Check out our [interactive Google Colab notebook](https://colab.research.google.com/github/polizzilab/NISE/blob/main/NISE_LASErMPNN.ipynb) to get acquainted with the protocol. Novel protein-ligand poses can be generated with [CARPdock](https://github.com/benf549/CARPdock) which we provide a separate [interactive Google Colab notebook for running](https://colab.research.google.com/github/benf549/CARPdock/blob/main/run_CARPdock.ipynb).


Jointly optimize the sequence and structure of a protein-ligand binding pose with iterative selection-expansion.

### Installing NISE Environment

To run NISE, install the dependencies (LASErMPNN and Boltz-2) using either of the methods below:

##### 1. (Recommended) Create a Virtual Environment (venv) inside this repository.

1) Install `uv` if your system does not already have it installed. [See here for instructions on how to do this](https://docs.astral.sh/uv/getting-started/installation/).

2) Add `uv` to your path. For example, edit your `~/.bashrc` to contain the line `export PATH="~/.local/bin/uv:$PATH"`.

3) Run the setup.sh with the command `bash setup.sh`

This will install a new python environment containing the dependencies for LASErMPNN located at `./.venv/bin/python` (the setup.sh script will print out the install location so you can verify this.)


##### 2. Using separately installed LASErMPNN and Boltz conda environments.

You may wish to not install Boltz-2 again if it is already installed on your HPC. You can separately install LASErMPNN in its own conda environment setup the NISE repo following the instructions below.

1) Install [LASErMPNN](https://github.com/polizzilab/LASErMPNN) conda environment following the instructions in the README.md at the linked repository.

2) Follow this set of commands inside the NISE project directory after `git clone`-ing the project and installing the lasermpnn environment.

```bash
git submodule update --init --recursive

tar -xvf hetdict.tar.gz

conda activate lasermpnn

cd ./LigandMPNN
bash get_model_params.sh "./model_params"
```

3) Activate your conda environment containing Boltz-1x or Boltz-2x and run `which boltz` to get the path to the executable you call when running `boltz predict` commands. 
You will need to update this path in `run_nise_boltz1x.py` or `run_nise_boltz2x.py` respectively.

### Running NISE with LigandMPNN 

You may wish to run NISE trajectories using LigandMPNN in place of LASErMPNN. 

To do this, install LigandMPNN into a separate python environment (its dependencies conflict with LASErMPNN) and update `./run_nise_boltz2x_ligandmpnn.py` with the path to your LigandMPNN python executable.
With the ligandmpnn python environment activated, run `which python` to get the path to your LigandMPNN python executable and update the `ligandmpnn_python` parameter at the bottom of `./run_nise_boltz2x_ligandmpnn.py`.

### Generating input poses:

We recommend generating NISE input poses using the workflow outlined [here using CARPdock,](https://github.com/benf549/CARPdock). 
CARPdock is likely the fastest way to get a good starting point and has been experimentally validated on some (currently) unpublished test targets to generate binders with high experimentally determined affinities.
Initializations from RFDiffusion2/3, BoltzDesign1 or BoltzGen will almost certainly work as well, but how best to leverage these tools for ligand binder design remains untested. Generating de novo fold topologies will likely decrease experimental success rates as well.

### Running NISE:

1) Create a PDB file containing your PROTONATED input ligand with CONECT records encoding bonds (unless using NISE with LigandMPNN, then protonation is not necessary).:
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

# Use protonated smiles string from ChemDraw or OpenBabel prediction
./.venv/bin/python ./protonate_and_add_conect_records.py ./example_pdbs/02_apex_NISE_input-pose_00-seq_0980_model_0_rank_01.pdb "COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N" ./example_pdbs/test_input_protonated_conect.pdb

mkdir -p ./debug/input_backbones/

cp ./example_pdbs/test_input_protonated_conect.pdb ./debug/input_backbones/

./run_nise_boltz2x.py
```


For a LigandMPNN example run:

```bash
conda activate lasermpnn

mkdir -p ./debug/input_backbones/
cp ./example_pdbs/02_apex_NISE_input-pose_00-seq_0980_model_0_rank_01.pdb ./debug/input_backbones/

./run_nise_boltz2x_ligandmpnn.py
```

### For a complete example design campaign (generated an 80 pM Apixaban binder), check out the contents of [example_design_campaign](./example_design_campaign/README.md)
