#!/usr/bin/env python
"""
CLI wrapper around run_nise_boltz2x.py.

Parses all NISE / LASErMPNN / Boltz-2x hyperparameters from the command line
and prepares the on-disk input directories before kicking off the design run.
The design methodology itself is untouched: this script imports `main`,
`DesignCampaign`, the helper functions, and the module-level constants from
`run_nise_boltz2x` and just builds a `params` dict for them.

Minimal invocation:

    ./run_nise_boltz2x_cli.py \\
        --input-pdb ./example_pdbs/my_input_pose.pdb \\
        --smiles "COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N" \\
        --output-dir ./debug

The script will automatically create `<output-dir>/input_backbones/` and, by
default, protonate the input ligand + add CONECT records there before running.

Run with `-h` for the full list of options.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Resolve NISE_DIRECTORY_PATH so we can locate the baked submodules / venv
# regardless of where this script is invoked from.
_THIS_FILE = Path(os.path.abspath(__file__))
NISE_DIRECTORY_PATH = _THIS_FILE.parent
sys.path.insert(0, str(NISE_DIRECTORY_PATH))

# -----------------------------------------------------------------------------
# Defaults (mirror the params dict at the bottom of run_nise_boltz2x.py).
# -----------------------------------------------------------------------------
DEFAULT_MODEL_CHECKPOINT = (
    NISE_DIRECTORY_PATH / "LASErMPNN" / "model_weights" / "laser_weights_0p1A_nothing_heldout.pt"
)
DEFAULT_BOLTZ_EXECUTABLE = NISE_DIRECTORY_PATH / ".venv" / "bin" / "boltz"
DEFAULT_REDUCE_HETDICT = NISE_DIRECTORY_PATH / "reduce_wwPDB_het_dict_two_letter_bug_fixed.txt"

OBJECTIVE_FUNCTIONS = [
    "ligand_plddt",
    "iptm",
    "ligand_plddt_and_iptm",
    "pbind",
    "ligand_plddt_and_pbind",
    "iptm_and_pbind",
    "iptm_consensus",
    "ligand_plddt_consensus",
    "pbind_and_protenix_iptm",
]


# -----------------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------------
def _existing_file(p: str) -> Path:
    path = Path(p).expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"Not a regular file: {path}")
    return path


def _path_or_none(p: str) -> Path | None:
    if p is None or p == "" or p.lower() == "none":
        return None
    return Path(p).expanduser().resolve()


def _str_set(s: str) -> set:
    """Comma-or-whitespace-separated atom names -> set."""
    if s is None or s == "":
        return set()
    # Accept both "C1,C2,C3" and "C1 C2 C3".
    parts = [tok.strip() for tok in s.replace(",", " ").split() if tok.strip()]
    return set(parts)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_nise_boltz2x_cli.py",
        description=(
            "CLI for NISE with LASErMPNN + Boltz-2x. Creates the on-disk input "
            "tree under --output-dir and runs the design campaign. All design "
            "logic is inherited unchanged from run_nise_boltz2x.main()."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- I/O -----------------------------------------------------------------
    io_grp = p.add_argument_group("inputs / outputs (required)")
    io_grp.add_argument(
        "--input-pdb", type=_existing_file, required=True,
        help="Input pose PDB. By default the CLI protonates the ligand and adds "
             "CONECT records before placing the prepared structure in "
             "<output-dir>/input_backbones/.",
    )
    io_grp.add_argument(
        "--smiles", type=str, required=True,
        help="Ligand SMILES (must match the protonation state of --input-pdb).",
    )
    io_grp.add_argument(
        "--output-dir", type=Path, required=True,
        help="Root working directory for this campaign (will be created).",
    )
    io_grp.add_argument(
        "--ligand-3lc", type=str, default="GG2",
        help="Three-letter code assigned to the ligand in output PDBs. "
             "Should match the CCD code if --use-reduce-protonation is set.",
    )
    io_grp.add_argument(
        "--link", action="store_true",
        help="Symlink --input-pdb into input_backbones/ instead of copying. "
             "Only applies with --no-prepare-input.",
    )
    io_grp.add_argument(
        "--overwrite-input-backbones", action="store_true",
        help="If input_backbones/ already contains PDBs, wipe them first.",
    )
    io_grp.add_argument(
        "--prepare-input", action=argparse.BooleanOptionalAction, default=True,
        help="If enabled, protonate the ligand and add CONECT records using "
             "protonate_and_add_conect_records.py before running. Disable this "
             "if --input-pdb is already prepared the way NISE expects.",
    )

    # ---- NISE loop -----------------------------------------------------------
    nise_grp = p.add_argument_group("NISE loop")
    nise_grp.add_argument("--num-iterations", type=int, default=35)
    nise_grp.add_argument("--num-top-backbones-per-round", type=int, default=3)
    nise_grp.add_argument("--sequences-sampled-per-backbone", type=int, default=64)
    nise_grp.add_argument("--sequences-sampled-at-once", type=int, default=30)
    nise_grp.add_argument(
        "--objective-function", choices=OBJECTIVE_FUNCTIONS, default="ligand_plddt",
    )

    protenix_grp = p.add_argument_group("Protenix structural-confidence rescore (optional)")
    protenix_grp.add_argument(
        "--use-protenix-rescore", action=argparse.BooleanOptionalAction, default=False,
        help="Fold each design again with Protenix as an independent structural cross-check "
             "(anti reward-hacking). Required by the *_consensus / pbind_and_protenix_iptm objectives.",
    )
    protenix_grp.add_argument(
        "--protenix-executable", default="protenix",
        help="Protenix CLI, invoked as `protenix pred -i ... -o ... -n <model>`.",
    )
    protenix_grp.add_argument("--protenix-model-name", default="protenix_base_default_v1.0.0")
    protenix_grp.add_argument(
        "--protenix-use-msa", action=argparse.BooleanOptionalAction, default=False,
        help="Fold Protenix single-sequence (default) to match NISE's Boltz inputs (msa: empty).",
    )

    nise_grp.add_argument(
        "--drop-rmsd-mask-atoms-from-ligand-plddt-calc",
        action=argparse.BooleanOptionalAction, default=True,
    )
    nise_grp.add_argument(
        "--keep-input-backbone-in-queue",
        action=argparse.BooleanOptionalAction, default=False,
    )
    nise_grp.add_argument(
        "--keep-best-generator-backbone",
        action=argparse.BooleanOptionalAction, default=True,
    )
    nise_grp.add_argument(
        "--rmsd-use-chirality",
        action=argparse.BooleanOptionalAction, default=False,
    )
    nise_grp.add_argument("--self-consistency-ligand-rmsd-threshold", type=float, default=2.5)
    nise_grp.add_argument("--self-consistency-protein-rmsd-threshold", type=float, default=2.5)
    nise_grp.add_argument(
        "--align-on-binding-site",
        action=argparse.BooleanOptionalAction, default=False,
        help="If set, protein RMSD becomes binding-site RMSD (sidechain atoms "
             "within 5A of the ligand in the LASErMPNN output structure).",
    )
    nise_grp.add_argument(
        "--fixed-identity-residue-indices", type=str, default=None,
        help="Optional ProDy selection string, e.g. 'resindex 0 2 3' or "
             "'resnum 1 3 5'. If set, those residues are held fixed during "
             "LASErMPNN sampling.",
    )
    nise_grp.add_argument("--burial-mask-alpha-hull-alpha", type=float, default=9.0)

    # ---- Ligand atom sets ----------------------------------------------------
    lig_grp = p.add_argument_group("ligand atom selections (PDB atom names)")
    lig_grp.add_argument(
        "--ligand-rmsd-mask-atoms", type=_str_set, default=set(),
        help="Space/comma-separated atom names to IGNORE in RMSD calculation.",
    )
    lig_grp.add_argument(
        "--ligand-atoms-enforce-buried", type=_str_set, default=set(),
        help="Atoms that must remain buried inside the convex hull when "
             "selecting new backbones.",
    )
    lig_grp.add_argument(
        "--ligand-atoms-enforce-exposed", type=_str_set, default=set(),
        help="Atoms that must remain exposed relative to the convex hull when "
             "selecting new backbones.",
    )

    # ---- LASErMPNN -----------------------------------------------------------
    laser_grp = p.add_argument_group("LASErMPNN sampling")
    laser_grp.add_argument(
        "--model-checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT,
    )
    laser_grp.add_argument("--laser-inference-device", type=str, default=None,
        help="Torch device string for LASErMPNN. Defaults to --boltz-devices[0].",
    )
    laser_grp.add_argument(
        "--laser-inference-dropout",
        action=argparse.BooleanOptionalAction, default=True,
    )
    laser_grp.add_argument("--sequence-temp", type=float, default=0.5)
    laser_grp.add_argument("--first-shell-sequence-temp", type=float, default=0.7)
    laser_grp.add_argument("--chi-temp", type=float, default=1e-6)
    laser_grp.add_argument("--seq-min-p", type=float, default=0.0)
    laser_grp.add_argument("--chi-min-p", type=float, default=0.0)
    laser_grp.add_argument(
        "--disabled-residues", type=str, nargs="+", default=["X", "C"],
        help="One-letter codes of residues LASErMPNN is not allowed to sample.",
    )
    laser_grp.add_argument(
        "--constrain-ala-gly-sampling-to-exposed-non-secondary-structure",
        action=argparse.BooleanOptionalAction, default=True,
    )
    laser_grp.add_argument("--budget-residue-sele-string", type=str, default=None)
    laser_grp.add_argument("--ala-budget", type=int, default=4)
    laser_grp.add_argument("--gly-budget", type=int, default=0)
    laser_grp.add_argument(
        "--disable-charged-fs",
        action=argparse.BooleanOptionalAction, default=True,
        help="Disables sampling D,E,K,R for buried first-shell ligand residues.",
    )

    # ---- Boltz ---------------------------------------------------------------
    boltz_grp = p.add_argument_group("Boltz-2x / Boltz-1x")
    boltz_grp.add_argument(
        "--boltz-executable", type=Path, default=DEFAULT_BOLTZ_EXECUTABLE,
    )
    boltz_grp.add_argument(
        "--boltz-devices", type=str, nargs="+", default=["cuda:0"],
        help="Torch-style device strings, e.g. --boltz-devices cuda:0 cuda:1.",
    )
    boltz_grp.add_argument("--boltz2-sampling-steps", type=int, default=200)
    boltz_grp.add_argument(
        "--boltz2-cache-directory", type=_path_or_none, default=None,
        help="Optional path to a pre-downloaded Boltz weights cache.",
    )
    boltz_grp.add_argument(
        "--boltz2-predict-affinity",
        action=argparse.BooleanOptionalAction, default=None,
        help="Whether to have Boltz-2x predict affinity. Defaults to True iff "
             "the objective function contains 'pbind'.",
    )
    boltz_grp.add_argument(
        "--use-boltz-conformer-potentials",
        action=argparse.BooleanOptionalAction, default=True,
    )
    boltz_grp.add_argument(
        "--use-boltz-1x", action="store_true",
        help="Run Boltz-1x instead of Boltz-2x (via --model boltz1). Incompatible "
             "with affinity prediction.",
    )
    boltz_grp.add_argument(
        "--boltz2-disable-kernels", action="store_true",
        help="Disable cuEquivariance kernels.",
    )
    boltz_grp.add_argument(
        "--boltz2-disable-nccl-p2p", action="store_true",
        help="Set NCCL_P2P_DISABLE=1; fixes NCCL hangs on some hardware.",
    )

    # ---- REDUCE protonation --------------------------------------------------
    reduce_grp = p.add_argument_group("REDUCE protonation (optional)")
    reduce_grp.add_argument(
        "--use-reduce-protonation", action="store_true",
        help="Use REDUCE to protonate Boltz outputs. Otherwise RDKit is used.",
    )
    reduce_grp.add_argument(
        "--reduce-executable", type=_path_or_none, default=None,
        help="Path to the `reduce` binary.",
    )
    reduce_grp.add_argument(
        "--reduce-hetdict", type=_path_or_none,
        default=(DEFAULT_REDUCE_HETDICT if DEFAULT_REDUCE_HETDICT.exists() else None),
        help="Path to the REDUCE hetdict file (usually created by "
             "inject_ligand_into_hetdict.py).",
    )

    # ---- WandB / misc --------------------------------------------------------
    misc = p.add_argument_group("misc")
    misc.add_argument("--debug", action="store_true")
    misc.add_argument("--use-wandb", action="store_true")
    misc.add_argument("--wandb-project", type=str, default="design-campaigns")
    misc.add_argument("--wandb-entity", type=str, default=None)
    misc.add_argument(
        "--dry-run", action="store_true",
        help="Prepare the input_backbones/ directory and print the resolved "
             "params dict, but do not launch the design campaign.",
    )

    return p


# -----------------------------------------------------------------------------
# Input-directory preparation
# -----------------------------------------------------------------------------
def prepare_input_directory(
    output_dir: Path,
    input_pdb: Path,
    smiles: str,
    prepare_input: bool,
    link: bool,
    overwrite: bool,
) -> tuple[Path, Path]:
    """
    Create <output_dir>/input_backbones/ and place a prepared input PDB inside it.

    Returns the resolved output_dir path and prepared backbone path.
    """
    output_dir = output_dir.expanduser().resolve()
    backbones_dir = output_dir / "input_backbones"

    output_dir.mkdir(parents=True, exist_ok=True)
    backbones_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for old in backbones_dir.glob("*.pdb"):
            old.unlink()

    if prepare_input:
        dest = backbones_dir / f"{input_pdb.stem}_protonated_conect.pdb"
    else:
        dest = backbones_dir / input_pdb.name

    if dest.exists() or dest.is_symlink():
        if dest.resolve() == input_pdb.resolve():
            print(f"[prepare] {dest} already points to {input_pdb}; skipping.")
            return output_dir, dest
        if not overwrite:
            raise FileExistsError(
                f"{dest} already exists. Pass --overwrite-input-backbones to "
                f"replace it, or remove it manually."
            )
        dest.unlink()

    if prepare_input:
        # Import only when needed so argparse help still works outside the full env.
        from protonate_and_add_conect_records import main as protonate_input_pdb

        protonate_input_pdb(str(input_pdb), str(dest), smiles)
        print(f"[prepare] protonated {input_pdb}  ->  {dest}")
    elif link:
        os.symlink(input_pdb, dest)
        print(f"[prepare] symlink  {dest}  ->  {input_pdb}")
    else:
        shutil.copy2(input_pdb, dest)
        print(f"[prepare] copied   {input_pdb}  ->  {dest}")

    return output_dir, dest


def remap_ligand_atom_sets_after_prepare(
    args: argparse.Namespace,
    prepared_backbone_path: Path,
) -> None:
    atom_name_constraints = (
        args.ligand_rmsd_mask_atoms
        | args.ligand_atoms_enforce_buried
        | args.ligand_atoms_enforce_exposed
    )
    if not args.prepare_input or not atom_name_constraints:
        return

    from utility_scripts.calc_symmetry_aware_rmsd import main as calc_rmsd

    _, _, name_mapping = calc_rmsd(args.input_pdb, prepared_backbone_path, args.smiles)
    if not name_mapping:
        raise SystemExit(
            "ERROR: failed to map ligand atom names from the input PDB to the "
            "prepared PDB. Try preparing the PDB manually and rerun with "
            "--no-prepare-input."
        )

    missing_atom_names = sorted(atom_name_constraints - set(name_mapping))
    if missing_atom_names:
        missing = ", ".join(missing_atom_names)
        raise SystemExit(
            "ERROR: ligand atom names were not found in the input PDB mapping: "
            f"{missing}"
        )

    args.ligand_rmsd_mask_atoms = {
        name_mapping[name] for name in args.ligand_rmsd_mask_atoms
    }
    args.ligand_atoms_enforce_buried = {
        name_mapping[name] for name in args.ligand_atoms_enforce_buried
    }
    args.ligand_atoms_enforce_exposed = {
        name_mapping[name] for name in args.ligand_atoms_enforce_exposed
    }

    print("[prepare] remapped ligand atom-name options onto prepared PDB names:")
    for original_name in sorted(atom_name_constraints):
        print(f"[prepare]   {original_name} -> {name_mapping[original_name]}")


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, argparse.Namespace):
        return _json_safe(vars(value))
    return value


def write_cli_args_manifest(
    output_dir: Path,
    raw_argv: list[str],
    original_args: dict,
    resolved_args: argparse.Namespace,
    prepared_backbone_path: Path,
    params: dict,
) -> Path:
    manifest_path = output_dir / "nise_cli_args.json"
    manifest = {
        "command": [Path(sys.argv[0]).name, *raw_argv],
        "raw_argv": raw_argv,
        "parsed_args_original": original_args,
        "parsed_args_resolved": _json_safe(resolved_args),
        "prepared_backbone_path": str(prepared_backbone_path),
        "params": _json_safe(params),
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[run] wrote CLI argument manifest = {manifest_path}")
    return manifest_path


# -----------------------------------------------------------------------------
# Build the params dict that run_nise_boltz2x.main() consumes
# -----------------------------------------------------------------------------
def build_params(args: argparse.Namespace, input_dir: Path) -> dict:
    laser_sampling_params = {
        "sequence_temp": args.sequence_temp,
        "first_shell_sequence_temp": args.first_shell_sequence_temp,
        "chi_temp": args.chi_temp,
        "seq_min_p": args.seq_min_p,
        "chi_min_p": args.chi_min_p,
        "disable_pbar": True,
        "disabled_residues_list": list(args.disabled_residues),
        "constrain_ala_gly_sampling_to_exposed_non_secondary_structure":
            args.constrain_ala_gly_sampling_to_exposed_non_secondary_structure,
        "budget_residue_sele_string": args.budget_residue_sele_string,
        "ala_budget": args.ala_budget,
        "gly_budget": args.gly_budget,
        "disable_charged_fs": args.disable_charged_fs,
    }

    # Affinity prediction default: True iff 'pbind' in the objective.
    if args.boltz2_predict_affinity is None:
        predict_affinity = "pbind" in args.objective_function
    else:
        predict_affinity = args.boltz2_predict_affinity

    laser_device = args.laser_inference_device or args.boltz_devices[0]

    params: dict = dict(
        debug=args.debug,
        use_wandb=args.use_wandb,

        input_dir=input_dir,

        ligand_3lc=args.ligand_3lc,
        ligand_rmsd_mask_atoms=args.ligand_rmsd_mask_atoms,
        ligand_atoms_enforce_buried=args.ligand_atoms_enforce_buried,
        ligand_atoms_enforce_exposed=args.ligand_atoms_enforce_exposed,
        laser_sampling_params=laser_sampling_params,
        ligand_smiles=args.smiles,

        objective_function=args.objective_function,
        drop_rmsd_mask_atoms_from_ligand_plddt_calc=args.drop_rmsd_mask_atoms_from_ligand_plddt_calc,
        keep_input_backbone_in_queue=args.keep_input_backbone_in_queue,
        keep_best_generator_backbone=args.keep_best_generator_backbone,
        rmsd_use_chirality=args.rmsd_use_chirality,
        self_consistency_ligand_rmsd_threshold=args.self_consistency_ligand_rmsd_threshold,
        self_consistency_protein_rmsd_threshold=args.self_consistency_protein_rmsd_threshold,

        align_on_binding_site=args.align_on_binding_site,
        fixed_identity_residue_indices=args.fixed_identity_residue_indices,

        use_reduce_protonation=args.use_reduce_protonation,
        reduce_hetdict_path=args.reduce_hetdict,
        reduce_executable_path=args.reduce_executable,

        model_checkpoint=Path(args.model_checkpoint).expanduser().resolve(),

        num_iterations=args.num_iterations,
        num_top_backbones_per_round=args.num_top_backbones_per_round,
        sequences_sampled_at_once=args.sequences_sampled_at_once,
        sequences_sampled_per_backbone=args.sequences_sampled_per_backbone,

        boltz2x_executable_path=str(Path(args.boltz_executable).expanduser().resolve()),
        boltz2_cache_directory=args.boltz2_cache_directory,
        boltz2_sampling_steps=args.boltz2_sampling_steps,
        boltz_inference_devices=list(args.boltz_devices),
        use_boltz_conformer_potentials=args.use_boltz_conformer_potentials,
        boltz2_predict_affinity=predict_affinity,
        use_boltz_1x=args.use_boltz_1x,
        boltz2_disable_kernels=args.boltz2_disable_kernels,
        boltz2_disable_nccl_p2p=args.boltz2_disable_nccl_p2p,

        burial_mask_alpha_hull_alpha=args.burial_mask_alpha_hull_alpha,

        laser_inference_device=laser_device,
        laser_inference_dropout=args.laser_inference_dropout,

        use_protenix_rescore=args.use_protenix_rescore,
        protenix_executable_path=args.protenix_executable,
        protenix_model_name=args.protenix_model_name,
        protenix_use_msa=args.protenix_use_msa,
    )
    return params


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
def validate_args_before_prepare(args: argparse.Namespace) -> None:
    if args.prepare_input and args.link:
        raise SystemExit(
            "ERROR: --link cannot be used with the default input preparation step. "
            "Use --no-prepare-input if you want to link an already-prepared PDB."
        )


def validate(args: argparse.Namespace, params: dict) -> None:
    if args.use_boltz_1x and params["boltz2_predict_affinity"]:
        raise SystemExit("ERROR: --use-boltz-1x is incompatible with affinity prediction.")

    if "pbind" in args.objective_function and not params["boltz2_predict_affinity"]:
        raise SystemExit(
            f"ERROR: objective '{args.objective_function}' requires affinity "
            f"prediction. Remove --no-boltz2-predict-affinity or pick a different objective."
        )

    ckpt = Path(params["model_checkpoint"])
    if not ckpt.exists():
        message = (
            f"LASErMPNN checkpoint not found at {ckpt}. "
            f"Pass --model-checkpoint."
        )
        if args.dry_run:
            print(f"WARNING: {message}")
        else:
            raise SystemExit(f"ERROR: {message}")

    boltz_exe = Path(params["boltz2x_executable_path"])
    if not boltz_exe.exists():
        message = (
            f"boltz executable not found at {boltz_exe}. "
            f"Pass --boltz-executable."
        )
        if args.dry_run:
            print(f"WARNING: {message}")
        else:
            raise SystemExit(f"ERROR: {message}")

    if args.use_reduce_protonation:
        if args.reduce_executable is None or not Path(args.reduce_executable).exists():
            raise SystemExit(
                "ERROR: --use-reduce-protonation requires --reduce-executable pointing at "
                "a working `reduce` binary."
            )
        if args.reduce_hetdict is None or not Path(args.reduce_hetdict).exists():
            raise SystemExit(
                "ERROR: --use-reduce-protonation requires --reduce-hetdict pointing at a "
                "REDUCE hetdict (see inject_ligand_into_hetdict.py)."
            )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    original_args = _json_safe(args)
    validate_args_before_prepare(args)

    input_dir, prepared_backbone_path = prepare_input_directory(
        args.output_dir,
        args.input_pdb,
        smiles=args.smiles,
        prepare_input=args.prepare_input,
        link=args.link,
        overwrite=args.overwrite_input_backbones,
    )
    remap_ligand_atom_sets_after_prepare(args, prepared_backbone_path)

    params = build_params(args, input_dir)
    validate(args, params)
    write_cli_args_manifest(input_dir, raw_argv, original_args, args, prepared_backbone_path, params)

    print(f"[run] NISE_DIRECTORY_PATH = {NISE_DIRECTORY_PATH}")
    print(f"[run] input_dir          = {input_dir}")
    print(f"[run] prepared backbone = {prepared_backbone_path}")
    print(f"[run] model_checkpoint   = {params['model_checkpoint']}")
    print(f"[run] boltz executable   = {params['boltz2x_executable_path']}")
    print(f"[run] devices            = {params['boltz_inference_devices']}")
    print(f"[run] objective          = {params['objective_function']}  "
          f"(predict_affinity={params['boltz2_predict_affinity']})")

    if args.dry_run:
        import pprint
        print("\n[dry-run] resolved params dict:")
        pprint.pprint(params, width=120, compact=False)
        return 0

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=params)

    # Import the unmodified design-campaign logic only when we are actually
    # ready to run, so that `-h` / `--dry-run` work even outside the full env.
    import run_nise_boltz2x as nise
    nise.main(**params)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
