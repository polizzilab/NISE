import os
import io
import sys
import pickle
import pathlib
import asyncio
import warnings
from typing import *
from dataclasses import asdict, replace


import torch
import numpy as np
from tqdm import tqdm
#  from aiofile import async_open
import torch.nn.functional as F

# Import Boltz-1 dependencies.
# CURR_DIR_PATH = str(pathlib.Path(os.path.abspath(__file__)).parent.parent.absolute())

from boltz.model.model import Boltz1
from boltz.main import BoltzDiffusionParams
from boltz.data.parse.fasta import parse_fasta
from boltz.data import const
from boltz.data.module.inference import BoltzTokenizer, BoltzFeaturizer, Input, collate
from boltz.data.types import Manifest, Record, StructureInfo, ChainInfo, Structure, Interface
from boltz.data.write.pdb import to_pdb

warnings.filterwarnings("ignore")

def load_model_and_modules(device, predict_args):
    """
    Load the model and modules required for structure prediction.
    """
    cache_dir = pathlib.Path("~/.boltz/").expanduser().absolute()
    checkpoint = cache_dir / "boltz1_conf.ckpt"

    model_module = Boltz1.load_from_checkpoint(
        checkpoint, strict=True, predict_args=predict_args, map_location=device, diffusion_process_args = asdict(BoltzDiffusionParams()), 
    ).to(device)

    # Load CCD
    ccd_path = cache_dir / 'ccd.pkl'
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()

    return model_module, ccd, tokenizer, featurizer


def get_fasta_string(protein_sequence, smiles):
    target_fasta = f">A|protein|empty\n{protein_sequence}"
    if smiles is not None:
        target_fasta += "\n>B|smiles\n{smiles}\n"
    return target_fasta


def get_manifest(protein_sequence, idx, has_ligand: bool = True):
    n_res = len(protein_sequence)
    if has_ligand:
        manifest = Manifest(**{
            "records": [Record(**{
                    "id": f"target_{idx}", 
                    "structure": StructureInfo(**{"resolution": None, "method": None, "deposited": None, "released": None, "revised": None, "num_chains": 2, "num_interfaces": None}), 
                    "chains": [
                        ChainInfo(**{"chain_id": 0, "chain_name": "A", "mol_type": 0, "cluster_id": -1, "msa_id": -1, "num_residues": n_res, "valid": True, "entity_id": 0}),
                        ChainInfo(**{"chain_id": 1, "chain_name": "B", "mol_type": 3, "cluster_id": -1, "msa_id": -1, "num_residues": 1, "valid": True, "entity_id": 1})
                    ], 
                    "interfaces": []
            })]
        })
    else:
        manifest = Manifest(**{
            "records": [Record(**{
                    "id": f"target_{idx}", 
                    "structure": StructureInfo(**{"resolution": None, "method": None, "deposited": None, "released": None, "revised": None, "num_chains": 2, "num_interfaces": None}), 
                    "chains": [
                        ChainInfo(**{"chain_id": 0, "chain_name": "A", "mol_type": 0, "cluster_id": -1, "msa_id": -1, "num_residues": n_res, "valid": True, "entity_id": 0}),
                    ], 
                    "interfaces": []
            })]
        })

    return manifest


def get_batch(target_fasta_str, manifest, device, tokenizer, featurizer, ccd):
    fasta_parser_input = (io.StringIO(target_fasta_str), manifest.records[0].id)
    target = parse_fasta(fasta_parser_input, ccd)
    tokenized_input = tokenizer.tokenize(Input(target.structure, {}))
    features = featurizer.process(
        tokenized_input,
        training=False,
        max_atoms=None,
        max_tokens=None,
        max_seqs=const.max_msa_seqs,
        pad_to_max_seqs=False,
        symmetries={},
    )
    features['record'] = manifest.records[0]

    for i,j in features.items():
        if isinstance(j, torch.Tensor):
            features[i] = j.to(device)
    
    return features, target.structure


def get_collated_batch(sequences, smiles_str, device, tokenizer, featurizer, ccd):
    subbatches, struct_dict = [], {}
    for sidx, sequence in enumerate(sequences):
        target_fasta_str = get_fasta_string(sequence, smiles_str)
        manifest = get_manifest(sequence, sidx, has_ligand = smiles_str is not None)
        batch, structure = get_batch(target_fasta_str, manifest, device, tokenizer, featurizer, ccd)
        subbatches.append(batch)
        struct_dict[f'target_{sidx}'] = structure
    collated_batch = collate(subbatches)
    return collated_batch, struct_dict


def predict_structures(sequences, smiles_str, device, model_module, tokenizer, featurizer, ccd, predict_args):
    torch.set_grad_enabled(False)

    if len(smiles_str) == 0:
        smiles_str = None

    collated_batch, all_structures = get_collated_batch(sequences, smiles_str, device, tokenizer, featurizer, ccd)
    predicted_structure_data = model_module(
        feats=collated_batch, 
        recycling_steps=predict_args['recycling_steps'], 
        num_sampling_steps=predict_args['sampling_steps'], 
        multiplicity_diffusion_train=model_module.training_args.diffusion_multiplicity, 
        diffusion_samples=predict_args['diffusion_samples']
    )

    predicted_structure_data['coords'] = predicted_structure_data['sample_atom_coords']
    predicted_structure_data['masks'] = collated_batch['atom_pad_mask']
    predicted_structure_data['confidence'] = predicted_structure_data['iptm']

    return predicted_structure_data, collated_batch, all_structures


def create_pdb_strings(
    model_output: dict[str, torch.Tensor],
    batch_data: dict[str, torch.Tensor],
    structures: dict[str, Structure],
) -> List[str]:
    """Write the predictions to disk."""

    # Get the records
    records: list[Record] = batch_data["record"] # type: ignore

    # Get the predictions
    coords = model_output["coords"]
    # print(coords.shape)
    # coords = coords.unsqueeze(0)

    pad_masks = model_output["masks"]
    if model_output.get("confidence") is not None:
        confidences = model_output["confidence"]
        confidences = confidences.reshape(len(records), -1).tolist()
    else:
        confidences = [0.0 for _ in range(len(records))]
    

    pdb_strs = []
    # Iterate over the records
    for record, coord, pad_mask, _confidence, plddt in zip(
        records, coords, pad_masks, confidences, model_output['plddt']
    ):
        coord = coord.unsqueeze(0)
        structure = structures[record.id]

        # Compute chain map with masked removed, to be used later
        chain_map = {}
        for i, mask in enumerate(structure.mask):
            if mask:
                chain_map[len(chain_map)] = i

        # Remove masked chains completely
        structure = structure.remove_invalid_chains()

        for model_idx in range(coord.shape[0]):
            # Get model coord
            model_coord = coord[model_idx]
            # Unpad
            coord_unpad = model_coord[pad_mask.bool()]
            coord_unpad = coord_unpad.cpu().numpy()

            # New atom table
            atoms = structure.atoms
            atoms["coords"] = coord_unpad
            atoms["is_present"] = True

            # Mew residue table
            residues = structure.residues
            residues["is_present"] = True

            # Update the structure
            interfaces = np.array([], dtype=Interface)
            new_structure: Structure = replace(
                structure,
                atoms=atoms,
                residues=residues,
                interfaces=interfaces,
            )

            # Update chain info
            chain_info = []
            for chain in new_structure.chains:
                old_chain_idx = chain_map[chain["asym_id"]]
                old_chain_info = record.chains[old_chain_idx]
                new_chain_info = replace(
                    old_chain_info,
                    chain_id=int(chain["asym_id"]),
                    valid=True,
                )
                chain_info.append(new_chain_info)

            pdb_strs.append(to_pdb(new_structure, plddt.tolist()))
    return pdb_strs


#  async def write_pdb_file(file_name, pdb_str):
    #  async with async_open(output_dir / f'boltz_{pathlib.Path(file_name).stem}.pdb', 'w') as f:
        #  await f.write(pdb_str)


#  async def write_all_files(file_names, pdb_strs):
    #  tasks = [write_pdb_file(file_name, pdb_str) for file_name, pdb_str in zip(file_names, pdb_strs)]
    #  await asyncio.gather(*tasks)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predict structures for many protein sequences generated to bind a target smiles string.')
    parser.add_argument('fasta_path', type=str, help='Path to the fasta file containing the protein sequences for batch inference. The fasta file is formatted as: >path1\\n<sequence_1>\\n>path2\\n<sequence_2>... The file stem at the end of the path will be used in the output pdb file name. Ex: /path/to/test_file_1.pdb will be boltz_test_file_1.pdb')
    parser.add_argument('smiles_str', type=str, help='Path to the smiles file containing the smiles strings for batch inference.')

    parser.add_argument('--device', '-d', type=str, default='cuda:6', help='Device to run the model on.')
    parser.add_argument('--recycling_steps', '-r', type=int, default=5, help='Number of recycling steps.')
    parser.add_argument('--sampling_steps', '-s', type=int, default=200, help='Number of sampling steps.')
    parser.add_argument('--sequences_per_batch', '-b', type=int, default=30, help='Number of sequences to predict in a single batch.')
    parser.add_argument('--output_dir', '-o', type=str, default='./boltz_batch_output/', help='Output directory to save the predicted pdb files.')
    parsed_args = parser.parse_args()

    output_dir = pathlib.Path(parsed_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = pathlib.Path(parsed_args.fasta_path)
    assert fasta_path.exists(), f"Path {fasta_path} does not exist."

    all_file_paths, all_sequences = [], []
    with fasta_path.open('r') as f:
        lines = [x.strip() for x in f.readlines()]
        for file, sequence in zip(lines[0::2], lines[1::2]):
            all_file_paths.append(file)
            all_sequences.append(sequence)

    if len(all_sequences) == 0:
        raise ValueError("No sequences found in the fasta file...")
    print(f"Found {len(all_sequences)} sequences in the fasta file.")

    if not len(all_sequences) == len(set([pathlib.Path(x).stem for x in all_file_paths])):
        raise ValueError(f"File stems in the fasta file are not unique.\nEx: {pathlib.Path(all_file_paths[0]).resolve()} has stem {pathlib.Path(all_file_paths[0]).stem}")
    
    device = parsed_args.device
    predict_args = {
        "recycling_steps": parsed_args.recycling_steps,
        "sampling_steps": parsed_args.sampling_steps,
        "diffusion_samples": 1, # I wouldn't change this as it might break my modified pdb writer.
    }

    model_module, ccd, tokenizer, featurizer = load_model_and_modules(device, predict_args)
    sequences_chunked = np.array_split(all_sequences, max(len(all_sequences) // parsed_args.sequences_per_batch, 1))
    file_names_chunked = np.array_split(all_file_paths, max(len(all_file_paths) // parsed_args.sequences_per_batch, 1))

    for file_names, sequences in tqdm(
        zip(file_names_chunked, sequences_chunked), 
        total=len(sequences_chunked), dynamic_ncols=True, desc="Predicting structures..."
    ):
        predicted_structure_data, collated_batch, all_structures = predict_structures(
            sequences, parsed_args.smiles_str, device, 
            model_module, tokenizer, featurizer, ccd, predict_args
        )
        pdb_strs = create_pdb_strings(predicted_structure_data, collated_batch, all_structures)

        # This is slower than the asyncio version below but equivalent.
        for idx, (file_name, pdb_str) in enumerate(zip(file_names, pdb_strs)):
           with open(output_dir / f'boltz_{pathlib.Path(file_name).stem}.pdb', 'w') as f:
               f.write(pdb_str)
        #  asyncio.run(write_all_files(file_names, pdb_strs))

