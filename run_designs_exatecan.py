#!/usr/bin/env python

import io
import os
import sys
import time
import shutil
import asyncio
import requests
import subprocess
from collections import defaultdict
from typing import *

import wandb
import torch
import prody as pr
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.express as px
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit_to_params import Params

CURR_DIR_PATH = str(Path(os.path.abspath(__file__)).parent)
LASER_PATH = str(Path(CURR_DIR_PATH) / 'LASErMPNN')
sys.path.append(LASER_PATH)

from run_inference import load_model_from_parameter_dict # type: ignore
from run_batch_inference import _run_inference, output_protein_structure, output_ligand_structure # type: ignore
from utility_scripts.burial_calc import compute_fast_ligand_burial_mask
from utility_scripts.calc_symmetry_aware_rmsd import _main as calc_rmsd

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

    ligand = input_protein.select('hetero').copy()
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
    B.setChids(['L' for _ in range(len(ligand.getResnames()))])
    B.setResnums([1 for _ in range(len(ligand.getResnames()))])
    A = input_protein.select('not hetero').copy() + B
    pr.writePDB('test2.pdb', A)

    ligname = lignames_set.pop()
    p = Params.from_mol(pdb_mol, name=ligname)
    p.dump(params_path) # type: ignore


def setup_boltz_workers(
    boltz_inference_device: List[str], boltz_python_path: Path, boltz_flask_server_script_path: Path, boltz_num_recycles: int, 
    boltz_num_diffusion_steps: int, ligand_smiles: str, worker_init_port: int
) -> List[Tuple[subprocess.Popen, str]]:
    assert len(boltz_inference_device) > 0, 'No devices specified for Boltz sampling.'

    worker_processes = []
    for idx in range(len(boltz_inference_device)):
        port_idx = worker_init_port + idx
        worker_port = f'http://localhost:{port_idx}'

        worker_process = subprocess.Popen([
            str(boltz_python_path), str(boltz_flask_server_script_path),
            str(port_idx), str(boltz_num_recycles), str(boltz_num_diffusion_steps), ligand_smiles, boltz_inference_device[idx]
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        worker_processes.append((worker_process, worker_port))
    
    print('Waiting for Boltz workers to boot...')
    boltz_server_boot_queue = {port: False for _, port in worker_processes}
    while not all(boltz_server_boot_queue.values()):
        time.sleep(1)
        for port, booted in boltz_server_boot_queue.items():
            if booted:
                continue
            try:
                requests.get(port)
                boltz_server_boot_queue[port] = True
            except requests.exceptions.ConnectionError:
                pass
    
    print('Boltz workers booted!')
    return worker_processes


class BoltzWorkerManager:
    def __init__(self, boltz_inference_device, boltz_python_path, boltz_flask_server_script_path, boltz_num_recycles, boltz_num_diffusion_steps, ligand_smiles, worker_init_port):
        self.worker_processes = setup_boltz_workers(
            boltz_inference_device, boltz_python_path, boltz_flask_server_script_path, 
            boltz_num_recycles, boltz_num_diffusion_steps, ligand_smiles, worker_init_port
        )
    
    def __del__(self):

        if not hasattr(self, 'worker_processes'):
            return 

        for process, _ in self.worker_processes:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

class DesignCampaign:
    def __init__(self, 
        model_checkpoint, input_dir, ligand_rmsd_mask_atoms, ligand_burial_mask_atoms, laser_inference_device, debug, ligand_3lc,
        rmsd_use_chirality, self_consistency_ligand_rmsd_threshold, self_consistency_protein_rmsd_threshold,
        laser_inference_dropout, num_iterations, num_top_backbones_per_round, laser_sampling_params, sequences_sampled_per_backbone, 
        sequences_sampled_at_once, boltz_inference_device, boltz_python_path, boltz_flask_server_script_path, 
        boltz_num_recycles, boltz_num_diffusion_steps, ligand_smiles, worker_init_port, boltz_num_predicted_per_batch, **kwargs
    ):
        self.debug = debug
        self.ligand_3lc = ligand_3lc

        self.rmsd_use_chirality = rmsd_use_chirality
        self.self_consistency_ligand_rmsd_threshold = self_consistency_ligand_rmsd_threshold
        self.self_consistency_protein_rmsd_threshold = self_consistency_protein_rmsd_threshold

        self.num_iterations = num_iterations
        self.sequences_sampled_per_backbone = sequences_sampled_per_backbone
        self.sequences_sampled_at_once = sequences_sampled_at_once
        self.top_k = num_top_backbones_per_round

        self.ligand_rmsd_mask_atoms = ligand_rmsd_mask_atoms
        self.ligand_burial_mask_atoms = ligand_burial_mask_atoms
        self.ligand_smiles = ligand_smiles

        self.laser_sampling_params = laser_sampling_params
        self.laser_inference_dropout = laser_inference_dropout
        
        self.sampling_metadata = defaultdict(float)
        self.sampling_metadata['min_protein_rmsd'] = float('inf')
        self.sampling_metadata['min_ligand_rmsd'] = float('inf')

        self.input_backbones_path, self.sampling_dataframe_path, self.sampled_backbones_path, self.sdf_path, self.params_path = handle_directory_creation(input_dir, model_checkpoint)
        self.model, _, self.model_params = load_model_from_parameter_dict(model_checkpoint, torch.device(laser_inference_device))

        # Set model to eval mode, enable inference dropout if specified.
        self.model.eval()
        if self.laser_inference_dropout:
            for module in self.model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.train()

        # Set path and priority for input backbones.
        self.backbone_queue = [(x, torch.inf) for x in self.input_backbones_path.iterdir() if x.is_file() and x.suffix == '.pdb']
        if len(self.backbone_queue) > 1:
            raise NotImplementedError(f"More than one input backbone not currently supported.")

        construct_helper_files(self.sdf_path, self.params_path, self.backbone_queue[0][0], ligand_smiles)

        input_prot = pr.parsePDB(str(self.backbone_queue[0][0]))
        input_prot_lig_heavy = input_prot.select('hetero and not element H')

        self.boltz_num_predicted_per_batch = boltz_num_predicted_per_batch 
        self.boltz_worker_manager = BoltzWorkerManager(
            boltz_inference_device, boltz_python_path, boltz_flask_server_script_path, 
            boltz_num_recycles, boltz_num_diffusion_steps, ligand_smiles, worker_init_port
        )

        self.ligand_rmsd_mask_atoms = ligand_rmsd_mask_atoms

        assert self.sdf_path.exists(), f"Error creating SDF file {self.sdf_path}"
        assert self.params_path.exists(), f"Error creating params file {self.params_path}"
    
    def __del__(self):
        if hasattr(self, 'boltz_worker_manager') and self.boltz_worker_manager:
            del self.boltz_worker_manager

    def sample_sequences(self, backbone_path: str) -> List[pr.AtomGroup]:
        sampled_proteins = []
        sampled_sequences = []
        num_sampled = 0
        while num_sampled < self.sequences_sampled_per_backbone:
            remaining_samples = self.sequences_sampled_per_backbone - num_sampled
            num_seq_to_sample = min(remaining_samples, self.sequences_sampled_at_once)

            sampling_output, full_atom_coords, nh_coords, sampled_probs, batch_data, protein_complex_data = _run_inference(self.model, self.model_params, str(backbone_path), num_seq_to_sample, **self.laser_sampling_params)
            for idx in range(num_seq_to_sample):
                curr_batch_mask = batch_data.batch_indices == idx
                out_prot = output_protein_structure(full_atom_coords[curr_batch_mask], sampling_output.sampled_sequence_indices[curr_batch_mask], protein_complex_data.residue_identifiers, nh_coords[curr_batch_mask], sampled_probs[curr_batch_mask])
                out_lig = output_ligand_structure(protein_complex_data.ligand_info)
                out_complex = out_prot + out_lig
                sampled_proteins.append(out_complex)
                sampled_sequences.append(out_prot.ca.getSequence())

            num_sampled += num_seq_to_sample
        
        return sampled_proteins, sampled_sequences

    def identify_backbone_candidates(self, boltz_output_subdir: Path, laser_output_subdir: Path):

        ligand_rmsds = []
        protein_rmsds = []
        ligand_plddts = []
        ligand_is_buried = []

        sorted_designs_laser = sorted([x for x in laser_output_subdir.iterdir() if x.is_file() and x.suffix == '.pdb'], key=lambda x: float(x.stem.split('_')[2]))
        sorted_designs_boltz = sorted([x for x in boltz_output_subdir.iterdir() if x.is_file() and x.suffix == '.pdb'], key=lambda x: float(x.stem.split('_')[2]))

        assert len(sorted_designs_laser) == len(sorted_designs_boltz), f"Error: different number of designs in {boltz_output_subdir} and {laser_output_subdir}."
        for laser, boltz in zip(sorted_designs_laser, sorted_designs_boltz):

            laser_prot = pr.parsePDB(str(laser))

            boltz_string_str = open(boltz, 'r').read()
            boltz_string_io = io.StringIO(boltz_string_str)
            boltz_prot = pr.parsePDBStream(boltz_string_io)
            boltz_bfacs = boltz_prot.select('hetero and not element H').getBetas().mean()

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
            boltz_lig_only = boltz_prot.select('hetero and not element H')
            boltz_lig_only.setNames([boltz_to_laser_name_mapping[x] for x in boltz_lig_only.getNames()])
            boltz_lig_only.setResnames([self.ligand_3lc for _ in range(len(boltz_lig_only.getResnames()))])
            boltz_coords = boltz_lig_only.getCoords()
            burial_mask = np.array([x not in self.ligand_burial_mask_atoms for x in boltz_lig_only.getNames()])
            pr.writePDB(str(boltz), boltz_prot_only + boltz_lig_only)

            # Check ligand burial in boltz structure.
            ligand_heavy_atom_mask = compute_fast_ligand_burial_mask(boltz_prot.ca.getCoords(), boltz_coords[burial_mask], num_rays=3)
            
            if ligand_heavy_atom_mask.all().item():
                ligand_is_buried.append(True)
                if (ligand_rmsd < self.self_consistency_ligand_rmsd_threshold and protein_rmsd < self.self_consistency_protein_rmsd_threshold) or self.debug:

                    subprocess.run(f'/nfs/polizzi/bfry/programs/reduce/reduce -DB test_hetdict.txt -DROP_HYDROGENS_ON_ATOM_RECORDS -BUILD {boltz} > {boltz}_', shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    shutil.move(f'{boltz}_', boltz)

                    self.backbone_queue.append((boltz, boltz_bfacs))
            else:
                ligand_is_buried.append(False)
        
        self.backbone_queue = sorted(self.backbone_queue, key=lambda x: float(x[1]), reverse=True)[:self.top_k]

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
            logs['max_sampled_backbone_priority'] = min(self.backbone_queue[1:], key=lambda x: x[1])[1]
        except:
            logs['max_sampled_backbone_priority'] = float('-inf')

        print(logs)
        if use_wandb:
            logs['ligand_RMSD_vs_pLDDT_scatter'] = wandb.Plotly(px.scatter(dataframe, x='ligand_rmsds', y='ligand_plddts', color='protein_rmsds', hover_data=['protein_rmsds', 'sequences'], range_color=[0.0, 2.5], range_y=[0.0, 1.0], range_x=[0.0, 5.0])) # type: ignore
            wandb.log(logs)
    

async def predict_complex_structures(all_sampled_sequences: List[str], worker_processes, num_predicted_per_batch) -> List[pr.AtomGroup]:
    aio_running_loop = asyncio.get_running_loop()
    open_ports = [x[1] for x in worker_processes]
    worker_sequences = np.array_split(all_sampled_sequences, len(open_ports))
    boltz_predictions = await asyncio.gather(*[
        _run_boltz(list(worker_sequences[idx]), open_ports[idx], aio_running_loop, num_predicted_per_batch) for idx in range(len(open_ports))
    ])

    return [x for sublist in boltz_predictions for x in sublist]


async def _run_boltz(worker_sequences, worker_port, aio_running_loop, boltz_num_predicted_per_batch) -> List[pr.AtomGroup]:
    boltz_predicted_prody_structures = []
    seq_chunks = np.array_split(worker_sequences, max(len(worker_sequences) // boltz_num_predicted_per_batch, 1))
    for sampled_sequences in seq_chunks:
        async_resp = await aio_running_loop.run_in_executor(None, lambda: requests.post(f'{worker_port}/boltz', json={'sequences': list(sampled_sequences)}))
        resp = async_resp.json()
        if not 'pdb_strs' in resp:
            print(resp, 'Error in Boltz sampling.')
            return [ValueError(f'Error in Boltz sampling: {resp}')]
        
        for pdb_str in resp['pdb_strs']:
            boltz_predicted_prody_structures.append(pr.parsePDBStream(io.StringIO(pdb_str)))
    return boltz_predicted_prody_structures


def compute_laser_scores(protein_sequences_list: Sequence[pr.AtomGroup]) -> Tuple[List[float], List[float]]:
    full_sequence_scores = []
    binding_site_scores = []

    for protein in protein_sequences_list:
        laser_score = (-1 * np.log10(protein.ca.getBetas())).mean()
        laser_score_bs = (-1 * np.log10(protein.select('(same residue as ((protein and not element H) within 5.0 of (hetero and not element H))) and name CA').getBetas())).mean()

        full_sequence_scores.append(laser_score)
        binding_site_scores.append(laser_score_bs)

    return full_sequence_scores, binding_site_scores


def main(use_wandb, **kwargs):

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
        boltz_output_subdir = sampling_subdir / 'boltz_outputs'
        sampling_subdir.mkdir(exist_ok=True)
        laser_output_subdir.mkdir(exist_ok=True)
        boltz_output_subdir.mkdir(exist_ok=True)

        # Run Boltz on laser outputs.
        boltz_predicted_protein_structures = asyncio.run(predict_complex_structures(all_sampled_sequences, design_campaign.boltz_worker_manager.worker_processes, design_campaign.boltz_num_predicted_per_batch))

        for idx, (laser_output_structure, boltz_pred_structure)  in enumerate(zip(all_sampled_proteins, boltz_predicted_protein_structures)):
            laser_output_path = laser_output_subdir / f'laser_output_{idx}.pdb'
            boltz_output_path = boltz_output_subdir / f'boltz_output_{idx}.pdb'

            pr.writePDB(str(laser_output_path), laser_output_structure)
            pr.writePDB(str(boltz_output_path), boltz_pred_structure)

        # Identify any new backbone candidates.
        sorted_designs_laser, sorted_designs_rosetta, ligand_rmsds, protein_rmsds, ligand_plddts, ligand_is_buried = design_campaign.identify_backbone_candidates(boltz_output_subdir, laser_output_subdir)

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
        'sequence_temp': 0.2, 'first_shell_sequence_temp': 1.0, 
        'chi_temp': 1e-6, 'seq_min_p': 0.0, 'chi_min_p': 0.0,
        'disable_pbar': True,
    }

    params = dict(
        debug = (debug := True),
        use_wandb = (use_wandb := (True and not debug)),

        model_checkpoint = Path('/nfs/polizzi/bfry/ligandmpnn_split_last_chance_edge_vecs_optstep_55000.pt'),
        input_dir = Path('./debug/').resolve(),

        ligand_3lc = 'EXA', # Should match CCD if using reduce.
        ligand_rmsd_mask_atoms = {'C19', 'C24'},
        ligand_burial_mask_atoms = {'C17', 'C23', 'C9', 'C10', 'C7', 'C3', 'N3', 'C1', 'C5', 'O3', 'C2', 'C4'},
        laser_sampling_params = laser_sampling_params,
        ligand_smiles = 'CC[C@]1(O)C2=C(C(N3CC4=C5[C@@H]([NH3+])CCC6=C5C(N=C4C3=C2)=CC(F)=C6C)=O)COC1=O',

        rmsd_use_chirality = True,
        self_consistency_ligand_rmsd_threshold = 1.5,
        self_consistency_protein_rmsd_threshold = 1.5,

        num_iterations = 100,
        num_top_backbones_per_round = 3,
        sequences_sampled_per_backbone = 100 if not debug else 8,
        sequences_sampled_at_once = 30,

        worker_init_port = 12389,
        boltz_python_path = Path('/nfs/polizzi/bfry/miniconda3/envs/boltz/bin/python3.9'),
        boltz_flask_server_script_path = Path('/nfs/polizzi/bfry/programs/boltz_tinker/batch_inference_flask_server.py'),
        boltz_num_recycles = 3 if not debug else 1,
        boltz_num_diffusion_steps = 200 if not debug else 100,
        boltz_num_predicted_per_batch = 8,

        laser_inference_device = 'cuda:0',
        laser_inference_dropout = True,
        boltz_inference_device = ['cuda:1', 'cuda:2', 'cuda:3', 'cuda:4', 'cuda:5', 'cuda:6', 'cuda:7'],
    )
    if use_wandb:
        wandb.init(project='design-campaigns', entity='benf549', config=params)
    main(**params)
