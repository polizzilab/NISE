[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/polizzilab/NISE/blob/main/NISE_LASErMPNN.ipynb)

# Neural Iterative Selection Expansion (NISE) using LASErMPNN/LigandMPNN with Boltz-1x/Boltz-2x

![A NISE Trajectory demonstraing optimization of P(bind) and predicted affinity from Boltz-2](./images/boltz2_animation.gif)

### Introduced in the paper [Zero-shot design of drug-binding proteins via neural iterative selection-expansion](https://www.nature.com/articles/s41586-026-10670-w)!

___
Check out our [interactive Google Colab notebook](https://colab.research.google.com/github/polizzilab/NISE/blob/main/NISE_LASErMPNN.ipynb) to get acquainted with the protocol. Novel protein-ligand poses can be generated with [CARPdock](https://github.com/benf549/CARPdock) which we provide a separate [interactive Google Colab notebook for running](https://colab.research.google.com/github/benf549/CARPdock/blob/main/run_CARPdock.ipynb).

___

Jointly optimize the sequence and structure of a protein-ligand binding pose with iterative selection-expansion.

### Installing NISE Environment

To run NISE, install the dependencies (LASErMPNN and Boltz-2) using one of the methods below:

##### 1. (Recommended) Create a Virtual Environment (venv) inside this repository.

1) Install `uv` if your system does not already have it installed. [See here for instructions on how to do this](https://docs.astral.sh/uv/getting-started/installation/).

2) Add `uv` to your path. For example, edit your `~/.bashrc` to contain the line `export PATH="~/.local/bin/uv:$PATH"`.

3) Run the setup.sh with the command `bash setup.sh`

This will install a new python environment containing the dependencies for LASErMPNN located at `./.venv/bin/python` (the setup.sh script will print out the install location so you can verify this.)


##### 2. (Also recommended on HPC) Use the prebuilt Singularity/Apptainer container.

Download the prebuilt image: [`nise.sif`](https://huggingface.co/benf549/NISE/blob/main/nise.sif)

Run the baked CLI app with `--nv` so the NVIDIA driver is forwarded:

```bash
apptainer run --cleanenv --nv --app nise-boltz2x-cli nise.sif \
    --input-pdb ./example_pdbs/02_apex_NISE_input-pose_00-seq_0980_model_0_rank_01.pdb \
    --smiles "COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N" \
    --output-dir ./debug
```

Advanced users who want to build or customize the container should see
[`APPTAINER.md`](./APPTAINER.md).

##### 3. Using separately installed LASErMPNN and Boltz conda environments.

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

You can run NISE either through the CLI wrapper or by copying and editing the
`run_nise_boltz2x.py` script for a specific design campaign. The former is easier to run while the latter is easier to tweak and extend the design protocol with your own modifications.

##### 1. Running NISE with the CLI wrapper.

The CLI wrapper creates the input directory structure for you, protonates the
ligand / adds CONECT records by default, and then launches the same
`run_nise_boltz2x.py` design workflow without changing the design logic:

```bash
./run_nise_boltz2x_cli.py \
    --input-pdb ./example_pdbs/02_apex_NISE_input-pose_00-seq_0980_model_0_rank_01.pdb \
    --smiles "COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N" \
    --output-dir ./debug
```

This will create `./debug/input_backbones/`, write a prepared
`*_protonated_conect.pdb` there, generate the helper ligand files, and run NISE
with the same defaults currently hard-coded in `run_nise_boltz2x.py`.

If your input PDB is already protonated and has the CONECT records you want to
preserve, add `--no-prepare-input`.

Important: the default CLI preparation step uses `protonate_and_add_conect_records.py`,
which renames ligand atoms sequentially by element. If you pass
`--ligand-rmsd-mask-atoms`, `--ligand-atoms-enforce-buried`, or
`--ligand-atoms-enforce-exposed` with the original input atom names, the CLI
will map those names onto the prepared PDB before running NISE.

The CLI also writes `nise_cli_args.json` in the output directory, recording the
raw command-line arguments, parsed arguments, any remapped ligand atom names, and
the final parameters passed into NISE.

Useful CLI options:

- `--ligand-rmsd-mask-atoms C1 C2 C3`
- `--ligand-atoms-enforce-buried C4 C5`
- `--ligand-atoms-enforce-exposed O1`
- `--objective-function ligand_plddt_and_pbind`
- `--boltz-devices cuda:0 cuda:1`
- `--fixed-identity-residue-indices "resnum 10 12 15"`
- `--budget-residue-sele-string "resnum 10 12 15"`
- `--no-prepare-input` if your input PDB is already protonated and has the CONECT records you want to preserve
- `--dry-run` to just prepare the directory and print the resolved config

To test out an example run:

```bash
# Native Python / venv workflow
./run_nise_boltz2x_cli.py \
    --input-pdb ./example_pdbs/02_apex_NISE_input-pose_00-seq_0980_model_0_rank_01.pdb \
    --smiles "COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N" \
    --output-dir ./debug

# Apptainer workflow using the baked CLI app
apptainer run --cleanenv --nv --app nise-boltz2x-cli nise.sif \
    --input-pdb ./example_pdbs/02_apex_NISE_input-pose_00-seq_0980_model_0_rank_01.pdb \
    --smiles "COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N" \
    --output-dir ./debug
```

##### 2. Running NISE by copying `run_nise_boltz2x.py`.

Use this workflow when you want to edit the campaign parameters directly in a
script instead of passing CLI options.

1) Copy the template script for your campaign:

```bash
cp run_nise_boltz2x.py my_run_nise_boltz2x.py
```

If you move the copied script outside the NISE repository, update
`NISE_DIRECTORY_PATH` at the top of the copied file.

2) Prepare a PDB file containing your PROTONATED input ligand with CONECT
records encoding bonds. If you have a non-protonated ligand or are missing
CONECT records, run:

```bash
protonate_and_add_conect_records.py {input_path}.pdb {smiles_string} {output_path}.pdb
```

WARNING: This will rename the ligand atoms, ligand chain, and resnum.

3) [Optional] If you want to protonate using reduce (keeps added ligand
hydrogen names consistent with input, but is a bit more finicky than the
alternative RDKit), inject your ligand into REDUCE hetdict by running:

```bash
inject_ligand_into_hetdict.py {output_path}.pdb
```

4) Create the input directory and copy your prepared PDB into it:

```bash
mkdir -p ./debug/input_backbones/
cp {output_path}.pdb ./debug/input_backbones/
```

5) Edit the `params` dictionary at the bottom of your copied script. At minimum,
set `input_dir`, `ligand_smiles`, `boltz_inference_devices`, and any objective
or residue constraints you want to use. If you want to constrain the number of
alanine and glycine residues predicted on the surface of the protein in
secondary-structured regions, run `identify_surface_residues.ipynb` and set the
resulting selection string as `budget_residue_sele_string` inside
`laser_sampling_params`.

6) Run your copied script:

```bash
./.venv/bin/python my_run_nise_boltz2x.py
```


For a LigandMPNN example run:

```bash
conda activate lasermpnn

mkdir -p ./debug/input_backbones/
cp ./example_pdbs/02_apex_NISE_input-pose_00-seq_0980_model_0_rank_01.pdb ./debug/input_backbones/

./run_nise_boltz2x_ligandmpnn.py
```

### Advanced container documentation

For details on building or customizing the Apptainer/Singularity container, see
[`APPTAINER.md`](./APPTAINER.md).

### For a complete example design campaign (generated a $K_d = 80 \pm 40$ picomolar binder to apixaban), check out the contents of [example_design_campaign](./example_design_campaign/)
