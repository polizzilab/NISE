# Neural Iterative Selection Expansion (NISE)

Jointly optimize the sequence and structure of a protein-ligand binding pose with iterative selection-expansion.

### Installing NISE Environment

The NISE conda environment contains dependencies for both LASErMPNN and Boltz-1:

1) Install conda environment

2) Install ProDy from source. 
There appears to be an issue with the conda installable ProDy and distance-based selections which are used within NISE. 
This can be resolved for now by installing ProDy from source.

Just follow this set of commands:
```bash
conda env create -f conda_env.yml
conda activate nise

git clone git@github.com:prody/ProDy.git
cd ProDy
python setup.py build_ext --inplace --force
pip install -Ue .
```

### TODO:

- [ ] Upload hetdict somewhere.
- [ ] Add reduce hetdict injection code.
- [ ] 
