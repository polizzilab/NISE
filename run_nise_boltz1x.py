#!/usr/bin/env python3
import io
import os
import sys
import time
import subprocess
from pathlib import Path
import prody as pr
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit_to_params import Params
import pandas as pd
from collections import defaultdict
import shutil

import warnings
warnings.filterwarnings("ignore")

from typing import *

# NOTE: If you move the run_nise*.py script, adjust the NISE_DIRECTORY_PATH
NISE_DIRECTORY_PATH = str(Path(os.path.abspath(__file__)).parent)
LASER_PATH = str(Path(NISE_DIRECTORY_PATH) / 'LASErMPNN')
sys.path.append(NISE_DIRECTORY_PATH)

import wandb
import torch
import numpy as np
import plotly
import plotly.express as px
from LASErMPNN.run_inference import load_model_from_parameter_dict # type: ignore
from LASErMPNN.run_batch_inference import _run_inference, output_protein_structure, output_ligand_structure # type: ignore

from utility_scripts.burial_calc import compute_fast_ligand_burial_mask
from utility_scripts.calc_symmetry_aware_rmsd import _main as calc_rmsd


def get_boltz_fasta_boilerplate(sequence, smiles):
    boltz_fasta = f">A|protein|empty\n{sequence}\n>B|smiles\n{smiles}\n"
    return boltz_fasta


def check_input(dir_path: Path, model_weights_path: Path):
    if not dir_path.exists():
        raise FileNotFoundError(f"Path {dir_path} does not exist.")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path {dir_path} is not a directory.")
    
    num_files = len([x for x in dir_path.iterdir() if x.is_file() and x.suffix == '.pdb'])
    if num_files == 0:
        raise FileNotFoundError(f"No PDB files found in {dir_path}")

    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights file {model_weights_path} does not exist.")


def handle_directory_creation(input_dir, model_checkpoint):
    input_backbones_path = input_dir / 'input_backbones'
    check_input(input_backbones_path, model_checkpoint)

    sampling_dataframe_path = input_dir / 'sampling_dataframes'
    sampled_backbones_path = input_dir / 'sampled_backbones'
    sampling_dataframe_path.mkdir(exist_ok=True)
    sampled_backbones_path.mkdir(exist_ok=True)

    sdf_path = input_dir / 'lig_from_input_pdb.sdf'
    params_path = input_dir / 'lig_from_input_pdb.params'

    return input_backbones_path, sampling_dataframe_path, sampled_backbones_path, sdf_path, params_path


def construct_helper_files(sdf_path, params_path, backbone_path, ligand_smiles):

    input_protein = pr.parsePDB(str(backbone_path))
    assert isinstance(input_protein, pr.AtomGroup), f"Error loading backbone {backbone_path}"

    ligand = input_protein.select('not protein').copy()
    assert ligand.numResidues() == 1, f"Error selecting ligand from backbone {backbone_path}"

    # Create a pdb string for the ligand.
    pdb_stream = io.StringIO()
    pr.writePDBStream(pdb_stream, ligand)
    ligand_string = pdb_stream.getvalue()

    lignames_set = set(ligand.getResnames())

    # Write ligand to PDB file.
    pdb_path = Path(str(sdf_path.resolve()).rsplit('.')[0] + '.pdb')
    with pdb_path.open('w') as f:
        f.write(ligand_string)
    
    pdb_mol = Chem.MolFromPDBBlock(ligand_string)
    smi_mol = Chem.MolFromSmiles(ligand_smiles)

    pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
    pdb_mol = AllChem.AddHs(pdb_mol, addCoords=True)
    AllChem.ComputeGasteigerCharges(pdb_mol)
    Chem.MolToMolFile(pdb_mol, str(sdf_path.resolve()))

    Chem.MolToPDBFile(pdb_mol, 'test.pdb')

    B = pr.parsePDB('test.pdb')
    B.setChids(['B' for _ in range(len(ligand.getResnames()))])
    B.setResnums([1 for _ in range(len(ligand.getResnames()))])
    A = input_protein.select('protein').copy() + B
    # pr.writePDB('test2.pdb', A)

    ligname = lignames_set.pop()
    p = Params.from_mol(pdb_mol, name=ligname)
    p.dump(params_path) # type: ignore

    with open(Path(params_path).parent / '.gitignore', 'w') as f:
        f.write('*')


class DesignCampaign:
    def __init__(self, 
        model_checkpoint, input_dir, ligand_rmsd_mask_atoms, ligand_atoms_enforce_buried, ligand_atoms_enforce_exposed, laser_inference_device, debug, ligand_3lc,
        rmsd_use_chirality, self_consistency_ligand_rmsd_threshold, self_consistency_protein_rmsd_threshold,
        laser_inference_dropout, num_iterations, num_top_backbones_per_round, laser_sampling_params, sequences_sampled_per_backbone, 
        sequences_sampled_at_once, boltz_inference_devices, ligand_smiles, 
        boltz1x_executable_path, use_reduce_protonation, keep_input_backbone_in_queue, keep_best_generator_backbone, boltz1x_disable_nccl_p2p, 
        burial_mask_alpha_hull_alpha, **kwargs
    ):
        self.debug = debug
        self.ligand_3lc = ligand_3lc
        self.boltz_inference_devices = boltz_inference_devices
        self.boltz1x_executable_path = boltz1x_executable_path
        self.use_reduce_protonation = use_reduce_protonation

        self.rmsd_use_chirality = rmsd_use_chirality
        self.self_consistency_ligand_rmsd_threshold = self_consistency_ligand_rmsd_threshold
        self.self_consistency_protein_rmsd_threshold = self_consistency_protein_rmsd_threshold
        self.keep_input_backbone_in_queue = keep_input_backbone_in_queue 
        self.keep_best_generator_backbone = keep_best_generator_backbone

        self.boltz1x_disable_nccl_p2p = boltz1x_disable_nccl_p2p 

        self.num_iterations = num_iterations
        self.sequences_sampled_per_backbone = sequences_sampled_per_backbone
        self.sequences_sampled_at_once = sequences_sampled_at_once
        self.top_k = num_top_backbones_per_round

        self.ligand_rmsd_mask_atoms = ligand_rmsd_mask_atoms
        self.ligand_atoms_enforce_buried = ligand_atoms_enforce_buried
        self.ligand_atoms_enforce_exposed = ligand_atoms_enforce_exposed
        self.burial_mask_alpha_hull_alpha = burial_mask_alpha_hull_alpha
        self.ligand_smiles = ligand_smiles

        self.laser_sampling_params = laser_sampling_params
        self.laser_inference_dropout = laser_inference_dropout
        
        self.sampling_metadata = defaultdict(float)
        self.sampling_metadata['min_protein_rmsd'] = float('inf')
        self.sampling_metadata['min_ligand_rmsd'] = float('inf')

        self.input_backbones_path, self.sampling_dataframe_path, self.sampled_backbones_path, self.sdf_path, self.params_path = handle_directory_creation(input_dir, model_checkpoint)
        self.model, self.model_params = load_model_from_parameter_dict(model_checkpoint, torch.device(laser_inference_device))

        # Set model to eval mode, enable inference dropout if specified.
        self.model.eval()
        if self.laser_inference_dropout:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        # Set path and priority for input backbones.
        if self.keep_input_backbone_in_queue:
            self.backbone_queue = [(x, torch.inf) for x in self.input_backbones_path.iterdir() if x.is_file() and x.suffix == '.pdb']
        else:
            self.backbone_queue = [(x, 0) for x in self.input_backbones_path.iterdir() if x.is_file() and x.suffix == '.pdb']

        self.backbone_to_best_generation = defaultdict(float)

        if len(self.backbone_queue) > 1:
            raise NotImplementedError(f"More than one input backbone not currently supported.")

        construct_helper_files(self.sdf_path, self.params_path, self.backbone_queue[0][0], ligand_smiles)

        input_prot = pr.parsePDB(str(self.backbone_queue[0][0]))
        input_prot_lig_heavy = input_prot.select('(not protein) and not element H')

        self.ligand_rmsd_mask_atoms = ligand_rmsd_mask_atoms

        assert self.sdf_path.exists(), f"Error creating SDF file {self.sdf_path}"
        assert self.params_path.exists(), f"Error creating params file {self.params_path}"
    
    def sample_sequences(self, backbone_path: str) -> List[pr.AtomGroup]:
        sampled_proteins = []
        sampled_sequences = []
        num_sampled = 0
        while num_sampled < self.sequences_sampled_per_backbone:
            remaining_samples = self.sequences_sampled_per_backbone - num_sampled
            num_seq_to_sample = min(remaining_samples, self.sequences_sampled_at_once)

            sampling_output, full_atom_coords, nh_coords, sampled_probs, batch_data, protein_complex_data = _run_inference(self.model, self.model_params, Path(backbone_path), num_seq_to_sample, **self.laser_sampling_params)
            protein_complex_data = protein_complex_data[0]
            for idx in range(num_seq_to_sample):
                curr_batch_mask = batch_data.batch_indices == idx
                out_prot = output_protein_structure(full_atom_coords[curr_batch_mask], sampling_output.sampled_sequence_indices[curr_batch_mask], protein_complex_data.residue_identifiers, nh_coords[curr_batch_mask], sampled_probs[curr_batch_mask])
                out_lig = output_ligand_structure(protein_complex_data.ligand_info)
                out_complex = out_prot + out_lig
                sampled_proteins.append(out_complex)
                sampled_sequences.append(out_prot.ca.getSequence())

            num_sampled += num_seq_to_sample
        
        return sampled_proteins, sampled_sequences

    def identify_backbone_candidates(self, sorted_designs_boltz: Sequence[Path], sorted_designs_laser: Sequence[Path], reduce_executable_path, reduce_hetdict_path):

        ligand_rmsds = []
        protein_rmsds = []
        ligand_plddts = []
        ligand_is_buried = []

        assert len(sorted_designs_laser) == len(sorted_designs_boltz), f"Error: different number of designs in" # {boltz_output_subdir} and {laser_output_subdir}."
        smi_mol = Chem.MolFromSmiles(self.ligand_smiles)

        for laser, boltz in zip(sorted_designs_laser, sorted_designs_boltz):
            laser_prot = pr.parsePDB(str(laser))
            boltz_string_str = open(boltz, 'r').read()
            boltz_string_io = io.StringIO(boltz_string_str)
            boltz_prot = pr.parsePDBStream(boltz_string_io)
            boltz_bfacs = boltz_prot.select('chid B and not element H').getBetas().mean() / 100

            try:
                protein_rmsd, ligand_rmsd, laser_to_boltz_name_mapping = calc_rmsd(laser_prot, boltz_prot, self.ligand_smiles, self.rmsd_use_chirality, self.ligand_rmsd_mask_atoms)
                if len(laser_to_boltz_name_mapping) == 0:
                    raise ValueError('No atoms in common between laser and boltz structures.')
                boltz_to_laser_name_mapping = {v: k for k, v in laser_to_boltz_name_mapping.items()}
            except:
                protein_rmsd = np.nan
                ligand_rmsd = np.nan
                laser_to_boltz_name_mapping = {}

            ligand_rmsds.append(ligand_rmsd)
            protein_rmsds.append(protein_rmsd)
            ligand_plddts.append(boltz_bfacs)

            if len(laser_to_boltz_name_mapping) == 0:
                print(f'{boltz}: Failed to map names between laser and boltz structures.')
                ligand_is_buried.append(False)
                continue

            # Remap the boltz structure ligand atoms with the name mapping.
            boltz_prot_only = boltz_prot.select('protein')
            boltz_lig_only = boltz_prot.select('(not protein) and not element H')
            boltz_lig_only.setNames([boltz_to_laser_name_mapping[x] for x in boltz_lig_only.getNames()])
            boltz_lig_only.setResnames([self.ligand_3lc for _ in range(len(boltz_lig_only.getResnames()))])
            boltz_coords = boltz_lig_only.getCoords()

            atoms_enforced_buried_mask = np.array([x in self.ligand_atoms_enforce_buried for x in boltz_lig_only.getNames()])
            atoms_enforced_exposed_mask = np.array([x in self.ligand_atoms_enforce_exposed for x in boltz_lig_only.getNames()])

            pdb_output_path = str(boltz)
            pr.writePDB(pdb_output_path, boltz_prot_only + boltz_lig_only)

            # Check ligand burial in boltz structure.
            all_buried_mask = compute_fast_ligand_burial_mask(boltz_prot.ca.getCoords(), boltz_coords[atoms_enforced_buried_mask], num_rays=5, alpha=self.burial_mask_alpha_hull_alpha)
            none_buried_mask = compute_fast_ligand_burial_mask(boltz_prot.ca.getCoords(), boltz_coords[atoms_enforced_exposed_mask], num_rays=5, alpha=self.burial_mask_alpha_hull_alpha)

            if (all_buried_mask.all().item() and (not none_buried_mask.any().item())) or self.debug:
                ligand_is_buried.append(True)
                if (ligand_rmsd < self.self_consistency_ligand_rmsd_threshold and protein_rmsd < self.self_consistency_protein_rmsd_threshold) or self.debug:

                    if self.use_reduce_protonation:
                        subprocess.run(
                            f'{reduce_executable_path} -DB {reduce_hetdict_path} -DROP_HYDROGENS_ON_ATOM_RECORDS -BUILD {pdb_output_path} > {pdb_output_path}_', 
                            shell=True, check=False, 
                            stdout=subprocess.DEVNULL if not self.debug else subprocess.PIPE, 
                            stderr=subprocess.DEVNULL if not self.debug else subprocess.PIPE
                        )
                        shutil.move(f'{pdb_output_path}_', pdb_output_path)
                    else:
                        ligand_string = io.StringIO()
                        pr.writePDBStream(ligand_string, boltz_lig_only.copy())
                        pdb_mol = Chem.MolFromPDBBlock(ligand_string.getvalue())
                        pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
                        pdb_mol = AllChem.AddHs(pdb_mol, addCoords=True)

                        ligand_prody = pr.parsePDBStream(io.StringIO(Chem.MolToPDBBlock(pdb_mol)))
                        ligand_prody.setResnames(self.ligand_3lc)
                        ligand_prody.setChids('B')
                        pr.writePDB(pdb_output_path, boltz_prot_only.copy() + ligand_prody)

                    self.backbone_queue.append((pdb_output_path, boltz_bfacs))
            else:
                ligand_is_buried.append(False)
        
        self.backbone_queue = sorted(self.backbone_queue, key=lambda x: float(x[1]), reverse=True)[:self.top_k]

        if self.keep_best_generator_backbone:
            top_backbone = max(self.backbone_to_best_generation.items(), key=lambda x: x[1])
            print(top_backbone)
            if not (top_backbone[0] in [x[0] for x in self.backbone_queue]):
                self.backbone_queue = self.backbone_queue[:-1] + [top_backbone]

        return sorted_designs_laser, sorted_designs_boltz, ligand_rmsds, protein_rmsds, ligand_plddts, ligand_is_buried

    
    def log(self, use_wandb: bool, dataframe: pd.DataFrame):

        self.sampling_metadata['min_protein_rmsd'] = min(self.sampling_metadata['min_protein_rmsd'], dataframe['protein_rmsds'].dropna().min())
        self.sampling_metadata['min_ligand_rmsd'] = min(self.sampling_metadata['min_ligand_rmsd'], dataframe['ligand_rmsds'].dropna().min())

        logs = dict(self.sampling_metadata)

        logs['mean_sampled_ligand_plddt'] = dataframe['ligand_plddts'].mean()
        logs['mean_sampled_protein_rmsd'] = dataframe['protein_rmsds'].dropna().mean()
        logs['mean_sampled_ligand_rmsd'] = dataframe['ligand_rmsds'].dropna().mean()
        logs['num_sequences_sampled'] = len(dataframe)

        logs['mean_laser_score'] = dataframe['laser_nll'].mean()
        logs['mean_laser_bs_score'] = dataframe['laser_bs_nll'].mean()

        try:
            if self.keep_input_backbone_in_queue:
                logs['max_sampled_backbone_priority'] = max(self.backbone_queue[1:], key=lambda x: x[1])[1]
            else:
                logs['max_sampled_backbone_priority'] = max(self.backbone_queue, key=lambda x: x[1])[1]
        except:
            logs['max_sampled_backbone_priority'] = 0

        print(logs)
        if use_wandb:
            scatter_fig = px.scatter(dataframe, x='ligand_rmsds', y='ligand_plddts', color='protein_rmsds', hover_data=['protein_rmsds', 'sequences'], range_color=[0.0, 2.5], range_y=[0.0, 1.0], range_x=[0.0, 5.0])
            logs['ligand_RMSD_vs_pLDDT_scatter'] = wandb.Html(plotly.io.to_html(scatter_fig)) # type: ignore
            wandb.log(logs)
    

def predict_complex_structures(boltz_inputs_dir, boltz1x_executable_path, boltz_inference_devices, boltz_output_dir, disable_nccl_p2p, debug):
    device_ints = [x.split(':')[-1] for x in boltz_inference_devices]
    command = f'{boltz1x_executable_path} predict {boltz_inputs_dir} --devices {len(device_ints)} --out_dir {boltz_output_dir} --output_format pdb --override'
    command = f'CUDA_VISIBLE_DEVICES={",".join(device_ints)} {command}'

    if disable_nccl_p2p:
        command = f'NCCL_P2P_DISABLE=1 {command}'

    try:
        # Boltz sometimes completes with a nonzero exit code despite completing successfully. 
        # If not all expected files were generated NISE will crash at the log step.
        subprocess.run(command, shell=True, check=False, stdout=subprocess.DEVNULL if not debug else None, stderr=subprocess.DEVNULL if not debug else None)
    except:
        pass


def compute_laser_scores(protein_sequences_list: Sequence[pr.AtomGroup]) -> Tuple[List[float], List[float]]:
    full_sequence_scores = []
    binding_site_scores = []

    for protein in protein_sequences_list:
        laser_score = (-1 * np.log10(protein.ca.getBetas())).mean()
        laser_score_bs = (-1 * np.log10(protein.select('(same residue as ((protein and not element H) within 5.0 of ((not protein) and not element H))) and name CA').getBetas())).mean()

        full_sequence_scores.append(laser_score)
        binding_site_scores.append(laser_score_bs)

    return full_sequence_scores, binding_site_scores


def main(use_wandb, reduce_executable_path, reduce_hetdict_path, **kwargs):

    design_campaign = DesignCampaign(**kwargs)

    for iidx in range(design_campaign.num_iterations):

        # Run laser on all backbone queue inputs.
        all_sampled_proteins = []
        backbone_sample_indices = []
        sampled_backbone_path = []
        all_sampled_sequences = []
        for bidx, (backbone_path, score) in enumerate(design_campaign.backbone_queue):
            sampled_proteins, sampled_sequences = design_campaign.sample_sequences(backbone_path)
            all_sampled_proteins.extend(sampled_proteins)
            backbone_sample_indices.extend([bidx] * len(sampled_proteins))
            sampled_backbone_path.extend([design_campaign.backbone_queue[bidx][0]] * len(sampled_proteins))
            all_sampled_sequences.extend(sampled_sequences)

        # Compute laser confidences.
        laser_nll, laser_bs_nll = compute_laser_scores(all_sampled_proteins)
        assert len(all_sampled_proteins) == len(backbone_sample_indices), f'Error in laser sampling: {len(all_sampled_proteins)} != {len(backbone_sample_indices)}'
        assert len(all_sampled_proteins) == len(laser_nll), f'Error in laser sampling: {len(all_sampled_proteins)} != {len(laser_nll)}'
        assert len(all_sampled_proteins) == len(laser_bs_nll), f'Error in laser sampling: {len(all_sampled_proteins)} != {len(laser_bs_nll)}'

        # Make subdirectories to write outputs to disk.
        sampling_subdir = design_campaign.sampled_backbones_path / f'iter_{iidx}'
        laser_output_subdir = sampling_subdir / 'laser_outputs'
        boltz_input_dir = sampling_subdir / 'boltz_inputs' 
        laser_output_subdir.mkdir(exist_ok=True, parents=True)
        boltz_input_dir.mkdir(exist_ok=True)

        # Write all the boltz input directories.
        all_boltz_input_path_names = []
        sampled_sequence_chunks = np.array_split(all_sampled_sequences, len(design_campaign.boltz_inference_devices)) 
        for idx, chunk_sequences in enumerate(sampled_sequence_chunks):
            for seq_idx, seq in enumerate(chunk_sequences):
                boltz_input_output_path = boltz_input_dir / f'chunk_{idx}_seq_{seq_idx}.fasta'
                all_boltz_input_path_names.append(boltz_input_output_path.stem)
                with boltz_input_output_path.open('w') as f:
                    f.write(get_boltz_fasta_boilerplate(seq, design_campaign.ligand_smiles))

        all_boltz_model_paths = [(sampling_subdir / 'boltz_results_boltz_inputs' / 'predictions' / x / f'{x}_model_0.pdb') for x in all_boltz_input_path_names]
        all_laser_output_paths = []
        for laser_output_structure, boltz_output_path in zip(all_sampled_proteins, all_boltz_model_paths):
            laser_output_path = laser_output_subdir / f'laser_{boltz_output_path.parent.stem}.pdb'
            pr.writePDB(str(laser_output_path), laser_output_structure)
            all_laser_output_paths.append(laser_output_path)

        curr_tries = 0
        max_tries = 10
        while curr_tries < max_tries and not all([x.exists() for x in all_boltz_model_paths]):
            if curr_tries != 0:
                print('Not all boltz predictions were completed or file system not updated.. retrying...')
                time.sleep(30)
            
            predict_complex_structures(boltz_input_dir, design_campaign.boltz1x_executable_path, design_campaign.boltz_inference_devices, sampling_subdir, design_campaign.boltz1x_disable_nccl_p2p, design_campaign.debug)
            curr_tries += 1

        assert all([x.exists() for x in all_boltz_model_paths]), f"Error: not all boltz predictions were written to disk."

        # Identify any new backbone candidates.
        sorted_designs_laser, sorted_designs_rosetta, ligand_rmsds, protein_rmsds, ligand_plddts, ligand_is_buried = design_campaign.identify_backbone_candidates(all_boltz_model_paths, all_laser_output_paths, reduce_executable_path, reduce_hetdict_path)

        with open(sampling_subdir / 'backbone_queue.txt', 'w') as f:
            f.write('\n'.join([f"{x[1]}\t{x[0]}" for x in design_campaign.backbone_queue]))

        iidx_data = {
            'sampled_backbone_path': sampled_backbone_path,
            'laser_nll': laser_nll,
            'laser_bs_nll': laser_bs_nll,
            'ligand_rmsds': ligand_rmsds,
            'protein_rmsds': protein_rmsds,
            'ligand_plddts': ligand_plddts,
            'ligand_is_buried': ligand_is_buried,
            'laser_paths': sorted_designs_laser,
            'rosetta_paths': sorted_designs_rosetta,
            'sequences': all_sampled_sequences,
            'curr_idx': [iidx] * len(laser_nll),
        }

        # Write data to disk.
        iidx_dataframe = pd.DataFrame(iidx_data)
        iidx_dataframe.to_pickle(design_campaign.sampling_dataframe_path / f'iter_{iidx}_data.pkl')

        design_campaign.log(use_wandb, iidx_dataframe)


if __name__ == "__main__":

    laser_sampling_params = {
        'sequence_temp': 0.5, 'first_shell_sequence_temp': 0.5, 
        'chi_temp': 1e-6, 'seq_min_p': 0.0, 'chi_min_p': 0.0,
        'disable_pbar': True, 'disabled_residues_list': ['X', 'C'], # Disables cysteine sampling by default.
        # ==================================================================================================== 
        # If constrain_ala_gly_sampling_to_exposed_non_secondary_structure is True (recommended), 
        # the ala_budget and gly_budget parameters are 
        # used to constrain the sampling of ALA and GLY residues to exposed non-secondary structured residues.
        # You can override the specific residues with the budget_residue_sele_string parameter (not recommended).
        # If constrain_ala_gly_sampling_to_exposed_non_secondary_structure is False and budget_residue_sele_string is None,
        # no constraints are applied to the sampling of ALA and GLY residues.
        # The reason we suggest doing this is to constrain the generated sequences to the manifold of sequences that are likely to exist in nature 
        # and not exploiting a propensity of structure prediction networks to predict structure in alanine rich sequences
        # ==================================================================================================== 
        'constrain_ala_gly_sampling_to_exposed_non_secondary_structure': True,
        'budget_residue_sele_string': None, 
        'ala_budget': 4, 'gly_budget': 0, # May sample up to 4 Ala and 0 Gly over the selected region.
        'disable_charged_fs': True,
    }

    params = dict(
        debug = (debug := True),
        use_wandb = (use_wandb := (True and not debug)),

        input_dir = Path('./debug/').resolve(),

        ligand_3lc = 'EXA', # Should match CCD if using reduce.
        ligand_rmsd_mask_atoms = {'C20', 'C21'}, # Atoms to IGNORE in RMSD calculation. 
        ligand_atoms_enforce_buried = {'O3', 'O2', 'N3', 'F1'}, # Atoms to enforce remain buried inside convex hull when selecting new backbones.
        ligand_atoms_enforce_exposed = {'N2'}, # Atoms to enforce remain exposed relative to the convex hull when selecting new backbones. I would suggest only using this for linker regions attached to your ligand or clearly exposed charged polar groups.
        laser_sampling_params = laser_sampling_params,
        ligand_smiles = "CC[C@]1(O)C2=C(C(N3CC4=C5[C@@H]([NH3+])CCC6=C5C(N=C4C3=C2)=CC(F)=C6C)=O)COC1=O",

        keep_input_backbone_in_queue = False,
        keep_best_generator_backbone = True, # The highest scoring pose may not necessarily generate higher scoring poses, keeps the pose that has generated the best poses after the first iteration in the queue if not already the best scoring pose.
        rmsd_use_chirality = False,
        self_consistency_ligand_rmsd_threshold = 2.5,
        self_consistency_protein_rmsd_threshold = 1.5,

        use_reduce_protonation = False, # If False, will use RDKit to protonate, these hydrogens will not preserve the input names.
        reduce_hetdict_path = Path('./modified_hetdict.txt').absolute(), # Can set to None if use_reduce_protonation False
        reduce_executable_path = Path('/nfs/polizzi/bfry/programs/reduce/reduce'), # Can set to None if use_reduce_protonation False

        model_checkpoint = Path(LASER_PATH) / 'model_weights/laser_weights_0p1A_noise_ligandmpnn_split.pt',

        num_iterations = 100,
        num_top_backbones_per_round = 3,
        sequences_sampled_at_once = 30,

        boltz1x_executable_path = '/nfs/polizzi/bfry/miniforge3/envs/boltz1x/bin/boltz',
        boltz_inference_devices = (boltz_inference_devices := ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7']),
        boltz1x_disable_nccl_p2p = False, # On some systems with certain graphics cards, NCCL can hang indefinitely. This flag fixes this issue allowing running boltz / NISE with multiple GPUs. https://github.com/NVIDIA/nccl/issues/631 

        sequences_sampled_per_backbone = 64 if not debug else 2 * len(boltz_inference_devices),
        burial_mask_alpha_hull_alpha = 9.0, # Set to a larger number for folds with wider pockets (ex: 7-helix bundle) (Ex: 100.0), see https://github.com/benf549/CARPdock/blob/main/visualize_hull.ipynb

        laser_inference_device = boltz_inference_devices[0],
        laser_inference_dropout = True,
    )
    if use_wandb:
        wandb.init(project='design-campaigns', entity='benf549', config=params)
    main(**params)
