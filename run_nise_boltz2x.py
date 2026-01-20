import io
import os
import sys
import time
import json
import shutil
import warnings
import subprocess
from typing import *
from pathlib import Path
from collections import defaultdict
warnings.filterwarnings("ignore")

# NOTE: If you move the run_nise*.py script, adjust the NISE_DIRECTORY_PATH
NISE_DIRECTORY_PATH = str(Path(os.path.abspath(__file__)).parent)
LASER_PATH = str(Path(NISE_DIRECTORY_PATH) / 'LASErMPNN')
sys.path.append(NISE_DIRECTORY_PATH)

import wandb
import torch
import plotly
import numpy as np
import prody as pr
import pandas as pd
from rdkit import Chem
import plotly.express as px
from rdkit.Chem import AllChem
from rdkit_to_params import Params

from LASErMPNN.run_inference import load_model_from_parameter_dict # type: ignore
from LASErMPNN.run_batch_inference import _run_inference, output_protein_structure, output_ligand_structure # type: ignore
from utility_scripts.burial_calc import compute_fast_ligand_burial_mask
from utility_scripts.calc_symmetry_aware_rmsd import _main as calc_rmsd


def compute_objective_function(confidence_metrics_dict: dict, objective_function: str) -> float:
    """
    Compute the objective function which will be MAXIMIZED by NISE.

    The input to this function is a dictionary of metrics extracted from Boltz-2x
    including design_ligand_plddt, iptm, and affinity metrics if running with
    boltz2_predict_affinity = True.

    The default behavior is to return design_ligand_plddt, but other metrics 
    and combinations of metrics could be used here.
    """

    # NOTE: add your own objective function in another elif block here!

    if objective_function == 'ligand_plddt':
        return confidence_metrics_dict['design_ligand_plddt']

    elif objective_function == 'ligand_plddt_and_iptm':
        return confidence_metrics_dict['design_ligand_plddt'] + confidence_metrics_dict['iptm']

    elif objective_function == 'iptm':
        return confidence_metrics_dict['iptm']

    elif objective_function == 'pbind':
        return confidence_metrics_dict['affinity_probability_binary']

    elif objective_function == 'ligand_plddt_and_pbind':
        return confidence_metrics_dict['design_ligand_plddt'] + confidence_metrics_dict['affinity_probability_binary']

    elif objective_function == 'iptm_and_pbind':
        return confidence_metrics_dict['iptm'] + confidence_metrics_dict['affinity_probability_binary']

    else:
        raise ValueError(f'Objective_function strategy {objective_function} is not implemented.')


def get_boltz_yaml_boilerplate(sequence: str, smiles: str, predict_affinity: bool):
    boltz_yaml = f"version: 1\nsequences:\n  - protein:\n      id: A\n      sequence: {sequence}\n      msa: empty\n  - ligand:\n      id: B\n      smiles: '{smiles}'\n"
    if predict_affinity:
        boltz_yaml += "properties:\n  - affinity:\n      binder: B\n"
    return boltz_yaml
        

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
        sequences_sampled_at_once, boltz_inference_devices, ligand_smiles, boltz2x_executable_path, 
        use_reduce_protonation, keep_input_backbone_in_queue, keep_best_generator_backbone, use_boltz_conformer_potentials,
        boltz2_predict_affinity, drop_rmsd_mask_atoms_from_ligand_plddt_calc, use_boltz_1x, 
        boltz2_disable_kernels, boltz2_disable_nccl_p2p, objective_function, fixed_identity_residue_indices, 
        align_on_binding_site, burial_mask_alpha_hull_alpha, boltz2_cache_directory, boltz2_sampling_steps, **kwargs
    ):
        self.debug = debug
        self.ligand_3lc = ligand_3lc
        self.boltz_inference_devices = boltz_inference_devices
        self.use_boltz_conformer_potentials = use_boltz_conformer_potentials
        self.use_boltz_1x = use_boltz_1x
        self.boltz2x_executable_path = boltz2x_executable_path
        self.predict_affinity = boltz2_predict_affinity
        self.use_reduce_protonation = use_reduce_protonation
        self.boltz2_cache_directory = boltz2_cache_directory
        self.boltz2_sampling_steps = boltz2_sampling_steps
        self.boltz2_disable_kernels = boltz2_disable_kernels
        self.boltz2_disable_nccl_p2p = boltz2_disable_nccl_p2p 
        self.objective_function = objective_function
        self.align_on_binding_site = align_on_binding_site
        self.fixed_identity_residue_indices = fixed_identity_residue_indices

        if self.fixed_identity_residue_indices is not None:
            print(f'Fixing residues: {self.fixed_identity_residue_indices}')

        if self.use_boltz_1x and self.predict_affinity:
            raise ValueError('Cannot use boltz1x with affinity prediction.')

        if 'pbind' in self.objective_function and (not self.predict_affinity):
            raise ValueError(f"predict_affinity must be True to use objective function {self.objective_function}")

        self.rmsd_use_chirality = rmsd_use_chirality
        self.self_consistency_ligand_rmsd_threshold = self_consistency_ligand_rmsd_threshold
        self.self_consistency_protein_rmsd_threshold = self_consistency_protein_rmsd_threshold
        self.keep_input_backbone_in_queue = keep_input_backbone_in_queue 
        self.drop_rmsd_mask_atoms_from_ligand_plddt_calc = drop_rmsd_mask_atoms_from_ligand_plddt_calc

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
        
        # Whether to keep the pose that generates the highest confidence sequences.
        self.keep_best_generator_backbone = keep_best_generator_backbone
        self.backbone_to_best_generation = defaultdict(float)

        if len(self.backbone_queue) > 1:
            raise NotImplementedError(f"More than one input backbone not currently supported.")

        construct_helper_files(self.sdf_path, self.params_path, self.backbone_queue[0][0], ligand_smiles)
        self.ligand_rmsd_mask_atoms = ligand_rmsd_mask_atoms

        assert self.sdf_path.exists(), f"Error creating SDF file {self.sdf_path}"
        assert self.params_path.exists(), f"Error creating params file {self.params_path}"
    
    def sample_sequences(self, backbone_path: str) -> Tuple[List[pr.AtomGroup], List[str], List[float], List[float]]:
        laser_nll, laser_bs_nll = [], []
        sampled_proteins, sampled_sequences = [], []
        num_sampled = 0
        while num_sampled < self.sequences_sampled_per_backbone:
            remaining_samples = self.sequences_sampled_per_backbone - num_sampled
            num_seq_to_sample = min(remaining_samples, self.sequences_sampled_at_once)

            backbone_path_ = backbone_path
            if self.fixed_identity_residue_indices is not None:
                # NOTE: there might be a better way to do this so that we don't lose information stored in the b-factor columns of backbones that we sample, 
                # but the important pLDDT information should be recorded in the log steps anyways.
                protein = pr.parsePDB(str(backbone_path))
                protein.setBetas(0.0)
                fixed_selection = protein.select(self.fixed_identity_residue_indices)
                if fixed_selection is None:
                    raise ValueError(f'ProDy encounted an error selecting residues with selection string: {self.fixed_identity_residue_indices}')
                fixed_selection.setBetas(1.0)
                backbone_path_ = Path(backbone_path).parent / f'{Path(backbone_path).stem}_fixbeta.pdb'
                pr.writePDB(str(backbone_path_), protein)
                self.laser_sampling_params.update({'fix_beta': True})

            sampling_output, full_atom_coords, nh_coords, sampled_probs, batch_data, protein_complex_data = _run_inference(self.model, self.model_params, Path(backbone_path_), num_seq_to_sample, **self.laser_sampling_params)
            protein_complex_data = protein_complex_data[0] # type: ignore
            for idx in range(num_seq_to_sample):
                curr_batch_mask = batch_data.batch_indices == idx

                curr_probs = sampled_probs[curr_batch_mask]
                nll = (-1 * torch.log10(curr_probs)).cpu().numpy().mean()
                bs_nll = (-1 * torch.log10(sampled_probs[curr_batch_mask][batch_data.first_shell_ligand_contact_mask[curr_batch_mask]])).cpu().numpy().mean()

                out_prot = output_protein_structure(full_atom_coords[curr_batch_mask], sampling_output.sampled_sequence_indices[curr_batch_mask], protein_complex_data.residue_identifiers, nh_coords[curr_batch_mask], curr_probs)
                out_lig = output_ligand_structure(protein_complex_data.ligand_info)

                out_complex = out_prot + out_lig
                out_complex.setTitle('LASErMPNN/NISE Generated Protein')
                sampled_proteins.append(out_complex)
                sampled_sequences.append(out_prot.ca.getSequence())

                laser_nll.append(nll)
                laser_bs_nll.append(bs_nll)

            num_sampled += num_seq_to_sample
        
        return sampled_proteins, sampled_sequences, laser_nll, laser_bs_nll

    def identify_backbone_candidates(self, sorted_designs_boltz: Sequence[Path], sorted_designs_laser: Sequence[Path], sampled_backbone_paths: Sequence[Path], reduce_executable_path, reduce_hetdict_path):
        assert len(sorted_designs_laser) == len(sorted_designs_boltz), f"Error: different number of designs in" # {boltz_output_subdir} and {laser_output_subdir}."
        smi_mol = Chem.MolFromSmiles(self.ligand_smiles)

        log_data = defaultdict(list)
        for laser, boltz, bb_path in zip(sorted_designs_laser, sorted_designs_boltz, sampled_backbone_paths):
            laser_prot = pr.parsePDB(str(laser))
            boltz_string_str = open(boltz, 'r').read()
            boltz_string_io = io.StringIO(boltz_string_str)
            boltz_prot = pr.parsePDBStream(boltz_string_io)

            confidence_data = {'laser_output_pdb_path': laser, 'boltz_output_pdb_path': boltz}
            with (boltz.parent / f'confidence_{boltz.stem}.json').open('r') as f:
                confidence_data.update(json.load(f))

            design_iptm = confidence_data['iptm']
            design_bind_probability, design_predicted_affinity = torch.nan, torch.nan
            if self.predict_affinity:
                with (boltz.parent / f'affinity_{boltz.stem.replace("_model_0", "")}.json').open('r') as f:
                    confidence_data.update(json.load(f))
                    design_bind_probability = confidence_data['affinity_probability_binary']
                    design_predicted_affinity = confidence_data['affinity_pred_value']

            log_data['affinity_probability_binary'].append(design_bind_probability)
            log_data['affinity_pred_value'].append(design_predicted_affinity)
            log_data['iptms'].append(design_iptm)

            try:
                protein_rmsd, ligand_rmsd, laser_to_boltz_name_mapping = calc_rmsd(laser_prot, boltz_prot, self.ligand_smiles, self.rmsd_use_chirality, self.ligand_rmsd_mask_atoms, align_on_binding_site=self.align_on_binding_site)
                if len(laser_to_boltz_name_mapping) == 0:
                    raise ValueError('No atoms in common between laser and boltz structures.')
                boltz_to_laser_name_mapping = {v: k for k, v in laser_to_boltz_name_mapping.items()}
            except:
                protein_rmsd = np.nan
                ligand_rmsd = np.nan
                laser_to_boltz_name_mapping = {}

            log_data['ligand_rmsds'].append(ligand_rmsd)
            log_data['protein_rmsds'].append(protein_rmsd)

            if len(laser_to_boltz_name_mapping) == 0:
                print(f'{boltz}: Failed to map names between laser and boltz structures.')
                log_data['ligand_is_buried'].append(False)
                log_data['ligand_plddts'].append(torch.nan)
                log_data['protein_plddts'].append(torch.nan)
                continue

            # Remap the boltz structure ligand atoms with the name mapping.
            boltz_prot_only = boltz_prot.select('chid A')
            boltz_lig_only = boltz_prot.select('chid B and not element H')
            boltz_lig_only.setNames([boltz_to_laser_name_mapping[x] for x in boltz_lig_only.getNames()])
            boltz_lig_only.setResnames([self.ligand_3lc for _ in range(len(boltz_lig_only.getResnames()))])
            boltz_coords = boltz_lig_only.getCoords()

            atoms_enforced_buried_mask = np.array([x in self.ligand_atoms_enforce_buried for x in boltz_lig_only.getNames()])
            atoms_enforced_exposed_mask = np.array([x in self.ligand_atoms_enforce_exposed for x in boltz_lig_only.getNames()])

            pdb_output_path = str(boltz)
            pr.writePDB(pdb_output_path, boltz_prot_only + boltz_lig_only)

            # Compute ligand pLDDT over relevant atoms.
            rmsd_mask = np.array([x not in self.ligand_rmsd_mask_atoms for x in boltz_lig_only.getNames()])
            if not self.drop_rmsd_mask_atoms_from_ligand_plddt_calc:
                rmsd_mask = np.ones_like(rmsd_mask)
            design_ligand_plddt = boltz_lig_only.getBetas()[rmsd_mask].mean() / 100
            design_protein_plddt = boltz_prot_only.copy().ca.getBetas().mean() / 100

            log_data['ligand_plddts'].append(design_ligand_plddt)
            log_data['protein_plddts'].append(design_protein_plddt)
            confidence_data['design_ligand_plddt'] = design_ligand_plddt
            confidence_data['design_protein_plddt'] = design_protein_plddt

            # Check ligand burial constraints are obeyed in predicted structure.
            all_buried_mask = compute_fast_ligand_burial_mask(boltz_prot.ca.getCoords(), boltz_coords[atoms_enforced_buried_mask], num_rays=5, alpha=self.burial_mask_alpha_hull_alpha)
            none_buried_mask = compute_fast_ligand_burial_mask(boltz_prot.ca.getCoords(), boltz_coords[atoms_enforced_exposed_mask], num_rays=5, alpha=self.burial_mask_alpha_hull_alpha)
            
            if (all_buried_mask.all().item() and (not none_buried_mask.any().item())) or self.debug:
                log_data['ligand_is_buried'].append(True)
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
                        # Protonate using RDKit
                        ligand_string = io.StringIO()
                        pr.writePDBStream(ligand_string, boltz_lig_only.copy())
                        pdb_mol = Chem.MolFromPDBBlock(ligand_string.getvalue())
                        pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
                        pdb_mol = AllChem.AddHs(pdb_mol, addCoords=True)

                        ligand_prody = pr.parsePDBStream(io.StringIO(Chem.MolToPDBBlock(pdb_mol)))
                        ligand_prody.setResnames(self.ligand_3lc)
                        ligand_prody.setChids('B')
                        pr.writePDB(pdb_output_path, boltz_prot_only.copy() + ligand_prody)

                    score = compute_objective_function(confidence_data, self.objective_function)
                    self.backbone_to_best_generation[bb_path] = max(score, self.backbone_to_best_generation[bb_path])
                    self.backbone_queue.append((pdb_output_path, score))
            else:
                log_data['ligand_is_buried'].append(False)
        
        self.backbone_queue = sorted(self.backbone_queue, key=lambda x: float(x[1]), reverse=True)[:self.top_k]

        # Adds the backbone which has generated the best scoring pose if it's not in the queue already.
        if self.keep_best_generator_backbone and len(self.backbone_to_best_generation) > 0:
            top_backbone = max(self.backbone_to_best_generation.items(), key=lambda x: x[1])
            print(top_backbone)
            if not (top_backbone[0] in [x[0] for x in self.backbone_queue]):
                self.backbone_queue = self.backbone_queue[:-1] + [top_backbone]

        return sorted_designs_laser, sorted_designs_boltz, log_data
    
    def log(self, use_wandb: bool, dataframe: pd.DataFrame):

        self.sampling_metadata['min_protein_rmsd'] = min(self.sampling_metadata['min_protein_rmsd'], dataframe['protein_rmsds'].dropna().min())
        self.sampling_metadata['min_ligand_rmsd'] = min(self.sampling_metadata['min_ligand_rmsd'], dataframe['ligand_rmsds'].dropna().min())

        logs = dict(self.sampling_metadata)

        logs['mean_sampled_ligand_plddt'] = dataframe['ligand_plddts'].dropna().mean()
        logs['mean_sampled_protein_plddt'] = dataframe['protein_plddts'].dropna().mean()
        logs['mean_sampled_protein_rmsd'] = dataframe['protein_rmsds'].dropna().mean()
        logs['mean_sampled_ligand_rmsd'] = dataframe['ligand_rmsds'].dropna().mean()
        logs['num_sequences_sampled'] = len(dataframe)

        logs['mean_affinity_probability'] = dataframe['affinity_probability_binary'].mean()
        logs['mean_affinity_pred_value'] = dataframe['affinity_pred_value'].mean()
        logs['mean_iptm'] = dataframe['iptms'].mean()

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
    

def predict_complex_structures(
    boltz_inputs_dir, boltz2x_executable_path, boltz_inference_devices, 
    boltz_output_dir, use_potentials, use_boltz_1x, disable_kernels, disable_nccl_p2p, boltz2_cache_directory, boltz2_sampling_steps, debug
):
    device_ints = [x.split(':')[-1] for x in boltz_inference_devices]
    command = f'{boltz2x_executable_path} predict {boltz_inputs_dir} --devices {len(device_ints)} --out_dir {boltz_output_dir} --output_format pdb --override --sampling_steps {int(boltz2_sampling_steps)} --sampling_steps_affinity {int(boltz2_sampling_steps)}'
    command = f'CUDA_VISIBLE_DEVICES={",".join(device_ints)} {command}'

    if disable_nccl_p2p:
        command = f'NCCL_P2P_DISABLE=1 {command}'

    if use_potentials:
        command += f' --use_potentials'

    if use_boltz_1x:
        command += f' --model boltz1'

    if disable_kernels:
        command += f' --no_kernels'

    if boltz2_cache_directory is not None:
        command += f' --cache {boltz2_cache_directory}'

    print(command)

    try:
        # Boltz sometimes completes with a nonzero exit code despite completing successfully. 
        # If not all expected files were generated NISE will crash at the log step.
        subprocess.run(command, shell=True, check=False, stdout=subprocess.DEVNULL if not debug else None, stderr=subprocess.DEVNULL if not debug else None)
    except:
        print('Boltz crashed! This might be fine, trying to recover...')
        pass


def main(use_wandb, reduce_executable_path, reduce_hetdict_path, **kwargs):

    design_campaign = DesignCampaign(**kwargs)

    for iidx in range(design_campaign.num_iterations):

        # Run laser on all backbone queue inputs.
        all_sampled_proteins, backbone_sample_indices, sampled_backbone_path, all_sampled_sequences = [], [], [], []
        laser_nll, laser_bs_nll = [], []
        for bidx, (backbone_path, score) in enumerate(design_campaign.backbone_queue):
            sampled_proteins, sampled_sequences, nlls, bs_nlls = design_campaign.sample_sequences(backbone_path)
            all_sampled_proteins.extend(sampled_proteins)
            backbone_sample_indices.extend([bidx] * len(sampled_proteins))
            sampled_backbone_path.extend([design_campaign.backbone_queue[bidx][0]] * len(sampled_proteins))
            all_sampled_sequences.extend(sampled_sequences)
            laser_nll.extend(nlls)
            laser_bs_nll.extend(bs_nlls)

        # Sanity check output shapes before trying to fold or log anything.
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
                boltz_input_output_path = boltz_input_dir / f'chunk_{idx}_seq_{seq_idx}.yaml'
                all_boltz_input_path_names.append(boltz_input_output_path.stem)
                with boltz_input_output_path.open('w') as f:
                    f.write(get_boltz_yaml_boilerplate(seq, design_campaign.ligand_smiles, design_campaign.predict_affinity))

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
            
            predict_complex_structures(
                boltz_input_dir, design_campaign.boltz2x_executable_path, design_campaign.boltz_inference_devices, 
                sampling_subdir, design_campaign.use_boltz_conformer_potentials, design_campaign.use_boltz_1x, 
                design_campaign.boltz2_disable_kernels, design_campaign.boltz2_disable_nccl_p2p, design_campaign.boltz2_cache_directory, design_campaign.boltz2_sampling_steps, design_campaign.debug
            )
            curr_tries += 1

        assert all([x.exists() for x in all_boltz_model_paths]), f"Error: not all boltz predictions were written to disk."

        # Identify any new backbone candidates.
        sorted_designs_laser, sorted_designs_rosetta, log_data = design_campaign.identify_backbone_candidates(all_boltz_model_paths, all_laser_output_paths, sampled_backbone_path, reduce_executable_path, reduce_hetdict_path)

        with open(sampling_subdir / 'backbone_queue.txt', 'w') as f:
            f.write('\n'.join([f"{x[1]}\t{x[0]}" for x in design_campaign.backbone_queue]))

        iidx_data = {
            'sampled_backbone_path': sampled_backbone_path,
            'laser_nll': laser_nll,
            'laser_bs_nll': laser_bs_nll,
            'laser_paths': sorted_designs_laser,
            'rosetta_paths': sorted_designs_rosetta,
            'sequences': all_sampled_sequences,
            'curr_idx': [iidx] * len(laser_nll),
        }
        iidx_data.update(dict(log_data))

        # Write data to disk.
        iidx_dataframe = pd.DataFrame(iidx_data)
        iidx_dataframe.to_pickle(design_campaign.sampling_dataframe_path / f'iter_{iidx}_data.pkl')

        design_campaign.log(use_wandb, iidx_dataframe)


if __name__ == "__main__":

    laser_sampling_params = {
        'sequence_temp': 0.5, 'first_shell_sequence_temp': 0.7, 
        'chi_temp': 1e-6, 'seq_min_p': 0.0, 'chi_min_p': 0.0,
        'disable_pbar': True, 'disabled_residues_list': ['X', 'C'], # Disables cysteine sampling by default.
        # ==================================================================================================== 
        # Optional: Pass a prody selection string of the form ('resnum 1 or resnum 3 or resnum 5...') to 
        # specify residues over which to constrain the sampling of ALA or GLY residues. 
        # This string can be generated using './identify_surface_residues.ipynb'
        # ==================================================================================================== 
        'budget_residue_sele_string': None, 
        'ala_budget': 4, 'gly_budget': 0, # May sample up to 4 Ala and 0 Gly over the selected region if not None.
        'disable_charged_fs': True, # Disables sampling D,E,K,R residues for buried residues around the ligand. 
    }


    params = dict(
        debug = (debug := True),
        use_wandb = (use_wandb := False),

        input_dir = Path('./debug/').resolve(),

        ligand_3lc = 'GG2', # Should match CCD code if using reduce.
        ligand_rmsd_mask_atoms = set(), # Atoms to IGNORE in RMSD calculation. 
        ligand_atoms_enforce_buried = set(), # Atoms to enforce remain buried inside convex hull when selecting new backbones.
        ligand_atoms_enforce_exposed = set(), # Atoms to enforce remain exposed relative to the convex hull when selecting new backbones. I would suggest only using this for linker regions attached to your ligand or clearly exposed charged polar groups.
        laser_sampling_params = laser_sampling_params,
        ligand_smiles = 'COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N',

        objective_function = (objective_function := 'ligand_plddt'), # Current options: {'ligand_plddt', 'iptm', 'ligand_plddt_and_iptm', 'pbind', 'ligand_plddt_and_pbind', 'iptm_and_pbind'}, Check the top of the file for implemented strategies, if you find an alternative strategy to work well please make a git commit so others can test it out as well!
        drop_rmsd_mask_atoms_from_ligand_plddt_calc = True,
        keep_input_backbone_in_queue = False,
        keep_best_generator_backbone = True, # The highest scoring pose may not necessarily generate higher scoring poses, keeps the pose that has generated the best poses after the first iteration in the queue if not already the best scoring pose.
        rmsd_use_chirality = False, # Will fail to compute RMSD on mismatched chirality ligands, might be bugged...
        self_consistency_ligand_rmsd_threshold = 2.5,
        self_consistency_protein_rmsd_threshold = 2.5,

        align_on_binding_site = False, # If align on binding site is True, protein RMSD (and self_consistency_protein_rmsd_threshold above) becomes binding site RMSD. Useful if you have a large protein with floppy regions away from the ligand. Binding site is computed as residues with sidechain atoms within 5.0A of the ligand in the lasermpnn output structure.
        fixed_identity_residue_indices = None, # An optional prody selection string of the form "resindex 0 2 3 4 5" or "resnum 1 3 5"...

        use_reduce_protonation = False, # If False, will use RDKit to protonate, these hydrogens will not preserve the input names and aren't placed conditioned on the sidechains but REDUCE sometimes drops hydrogens if geometry changes outside of expected bounds.
        reduce_hetdict_path = Path('./modified_hetdict.txt').absolute(), # Can set to None if use_reduce_protonation False
        reduce_executable_path = None, # Can set to None if use_reduce_protonation False

        model_checkpoint = Path(LASER_PATH) / 'model_weights/laser_weights_0p1A_nothing_heldout.pt',

        num_iterations = 35,
        num_top_backbones_per_round = 3,
        sequences_sampled_at_once = 30,

        boltz2x_executable_path = str((Path(NISE_DIRECTORY_PATH) / '.venv/bin/boltz').absolute()),
        boltz2_cache_directory = None, # Optional path to the boltz weights, can be used to avoid redownloading weights that have already been cached on your machine not in the default location.
        boltz2_sampling_steps = 200,
        boltz_inference_devices = (boltz_inference_devices := ['cuda:0',]), # a list of multiple torch-style device strings
        use_boltz_conformer_potentials = True, # Use Boltz-x mode, this is almost always better.
        boltz2_predict_affinity = True if ('pbind' in objective_function) else False,
        use_boltz_1x = False, # Run the same script using --model boltz-1, multi-device inference with this seems bugged with boltz v2.1.1
        boltz2_disable_kernels = False, # Disables cuEquivariance kernels, this is likely not necessary.
        boltz2_disable_nccl_p2p = False, # On some systems with certain graphics cards, NCCL can hang indefinitely. This flag fixes this issue allowing running boltz / NISE with multiple GPUs. https://github.com/NVIDIA/nccl/issues/631 

        sequences_sampled_per_backbone = 64 if not debug else 1 * len(boltz_inference_devices),
        burial_mask_alpha_hull_alpha = 9.0, # Set to a larger number for folds with wider pockets (ex: 7-helix bundle) (Ex: 100.0), see https://github.com/benf549/CARPdock/blob/main/visualize_hull.ipynb

        laser_inference_device = boltz_inference_devices[0],
        laser_inference_dropout = True,
    )
    if use_wandb:
        wandb.init(project='design-campaigns', entity='benf549', config=params)
    main(**params)
