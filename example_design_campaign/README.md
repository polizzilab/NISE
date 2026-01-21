### This directory contains a full worked example of a design campaign:

Which starts from initial docking through filtering and ranking final designs.

Unzip the contents of `all_apex_design_scripts.zip` to get started.

Read through the scripts in order of index (00/01/02 etc...), many of the scripts are configured for infrastructure specific to the Polizzi lab e.g. (for example Rosetta installation, number of GPUs to split model inference over, etc...)

The directory structure of the contents of `all_apex_design_scripts.zip` are as follows:

```
all_apx_design_scripts
├── 00_carpdock
│   ├── 00_carpdock_screen_apx_ntf2_2.py
│   ├── 00_carpdock_screen_apx_ntf2.py
│   ├── 01_get_paths_txt.ipynb
│   ├── 02_get_boltz_inputs.ipynb
│   ├── 03_analyze_boltz.ipynb
│   ├── 04_run_1round_nise.ipynb
│   ├── calc_symmetry_aware_rmsd.py
│   ├── carp_dock.py
│   └── utils
│       └── burial_calc.py
├── 01_nise
│   ├── 00_run_nise_boltz_2x_pose00.py
│   ├── pose_00
│   │   ├── 01_rank_top_by_bun_pose_newrank.ipynb
│   │   ├── bunsalyze
│   │   │   ├── LICENSE
│   │   │   ├── pyproject.toml
│   │   │   └── README.md
│   │   ├── input_backbones
│   │   │   └── seq_0980_model_0_rank_01.pdb
│   │   ├── rosetta
│   │   │   ├── rosetta_script_fixbb_relax_apo.xml
│   │   │   ├── rosetta_script_fixbb_relax_sidechains.xml
│   │   │   └── run_rosetta_emin.py
│   │   └── top_designs_by_bbuns
│   └── utility_scripts
│       ├── burial_calc.py
│       └── calc_symmetry_aware_rmsd.py
├── 02_filter_top_designs
│   ├── 00_submit_alphafold2.ipynb
│   ├── 01_check_rf3.ipynb
│   ├── 02_bunsalyze.ipynb
│   ├── bbunsalyze
│   ├── calc_symmetry_aware_rmsd.py
│   └── submit_rf3.py
```
