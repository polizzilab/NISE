#!/usr/bin/env python

# This script sets up a virtual environment with dependencies for LASErMPNN and Boltz-2 using uv. It will crash if uv is not installed.

import os
import re
import subprocess
import sys

class SetupException(Exception):
    pass

def has_uv() -> bool:
    try:
        subprocess.run("uv --version", check=True, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False

if not has_uv():
    raise SetupException('You do not have uv installed on this system. Please install uv or add an existing installation to your path and rerun this script. See https://docs.astral.sh/uv/getting-started/installation/')

venv_dir = ".venv"
venv_py = os.path.join(venv_dir, "bin", "python")

# 1) Create virtualenv if needed
if not os.path.isdir(venv_dir):
    subprocess.check_call(["uv", "venv", venv_dir])

print("Venv python:", venv_py)

# 2) Install base stack into the venv with uv, including pip itself
env = os.environ.copy()
env["UV_PYTHON"] = venv_py

subprocess.check_call(
    [
        "uv", "pip", "install",
        "pip",                # put pip into the venv
        "torch==2.8.0",
        "numpy<2.0.0",
        "scipy",
        "pandas",
        "scikit-learn",
        "h5py",
        "pytest",
        "prody",
        "matplotlib",
        "rdkit-to-params",
        "seaborn",
        "jupyter",
        "plotly",
        "pykeops",
        "logomaker",
        "wandb",
        "tqdm",
        "rdkit",
        "py3Dmol",
        "openbabel-wheel",
        "boltz[cuda]==2.2.1",
        "freesasa",
    ],
    env=env,
)

# 3) Inside the venv: detect torch CUDA + install matching PyG wheels with pip -f
pyg_install_code = r"""
import re
import sys
import subprocess
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Torch CUDA runtime version:", torch.version.cuda)

m = re.match(r"^(\d+\.\d+\.\d+)(\+cu\d+)?", torch.__version__)
if not m:
    raise SystemExit(f"Unexpected torch version: {torch.__version__}")

base_version, cuda_suffix = m.groups()
cuda_suffix = cuda_suffix or ""

if cuda_suffix:
    pyg_url = f"https://data.pyg.org/whl/torch-{base_version}{cuda_suffix}.html"
else:
    pyg_url = f"https://data.pyg.org/whl/torch-{base_version}+cpu.html"

print("Using PyG wheel index:", pyg_url)

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "torch_scatter",
    "torch_cluster",
    "-f", pyg_url,
])

print("All installs completed in venv:", sys.executable)
"""

subprocess.check_call([venv_py, "-c", pyg_install_code])

print("Done. A virtual environment containing LASErMPNN and Boltz-2 dependencies is now installed at:", venv_py)
