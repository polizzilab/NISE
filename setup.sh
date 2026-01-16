#!/usr/bin/env bash

# Installs LASErMPNN and LigandMPNN repositories inside the NISE directory.
git submodule update --init --recursive

# Untar the REDUCE hetdict.
tar -xvf hetdict.tar.gz

# Install LigandMPNN weights
cd ./LigandMPNN
bash get_model_params.sh "./model_params"
cd ..

# Creates a virtual environment in the ./.venv/ folder containing LASErMPNN and Boltz-2x dependencies.
python3 setup.py

