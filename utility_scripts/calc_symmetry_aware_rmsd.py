import io
import os
import warnings
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import prody as pr
import numpy as np
from typing import Tuple, Dict, Iterable, Optional

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def _main(
    ag1: pr.AtomGroup, ag2: pr.AtomGroup, smiles_string: str, 
    use_chirality: bool = False, reference_names_to_ignore: Optional[Iterable[str]] = None, 
    default_lig_name: str = 'LIG', align_on_binding_site: bool = False
) -> Tuple[float, float, Dict[str, str]]:
    """
    Given two prody atomgroups and a smiles string, compute the minimum RMSD between the two ligands after aligning the protein.
    Considers all possible matching atom permuations between the two ligands.

    Args:
        ag1: The reference prody atomgroup.
        ag2: The mobile prody atomgroup (will be aligned to ag1).
        smiles_string: The smiles string of the ligand.

        use_chirality: Whether to use chirality when matching atoms. 
            If the chirality is mismatched, the ligand won't be matched and the RMSD will be set to infinity and the mapping will be an empty dictionary.

        reference_names_to_ignore: The names of the atoms in the reference ligand pdb file to ignore when matching atoms.
            Allows you to ignore particular rotatable bonds which you don't want in the RMSD calculation.

    Output:
        min_rmsd: The minimum RMSD between the two ligands.
        name_mapping: The mapping of atom names from the reference to the mobile.
    
    Raises:
        ValueError: If RDKit fails to match the smiles string to the ligand structures.
    """
    if isinstance(reference_names_to_ignore, str):
        raise ValueError("reference_names_to_ignore should be an iterable of strings.")

    protein_rmsd = None
    if align_on_binding_site:
        ag1_binding_site = ag1.select('same residue as (protein and not element H and not (name CA or name C or name O or name N) within 5.0 of ((not protein) and not element H)) and name CA')
        if ag1_binding_site is not None:
            ag1_binding_site_str = '(resindex ' + ' '.join(map(str, ag1_binding_site.getResindices())) + ') and name CA'
            ag2_binding_site = ag2.select(ag1_binding_site_str)

            ag2 = pr.calcTransformation(ag2_binding_site, ag1_binding_site).apply(ag2)
            protein_rmsd = pr.calcRMSD(ag2_binding_site, ag1_binding_site)
        else:
            warnings.warn("No binding site found in the reference structure... Aligning on the CA atoms.")
    
    if protein_rmsd is None:
        # Align ag2 on the ag1 ca carbons.
        ag2 = pr.calcTransformation(ag2.ca, ag1.ca).apply(ag2)
        protein_rmsd = pr.calcRMSD(ag1.ca, ag2.ca)

    lig1 = ag1.select('(not protein) and not element H')
    lig2 = ag2.select('(not protein) and not element H')

    if len(set(lig1.getResnames()).pop()) > 3:
        lig1.setResnames(default_lig_name)

    if len(set(lig2.getResnames()).pop()) > 3:
        lig2.setResnames(default_lig_name)

    # pr.writePDB('test1.pdb', ag1)
    # pr.writePDB('test2.pdb', ag2)

    # Convert to rdkit objects.
    ref_stream = io.StringIO()
    mob_stream = io.StringIO()
    pr.writePDBStream(ref_stream, lig1)
    pr.writePDBStream(mob_stream, lig2)
    ref_mol = Chem.MolFromPDBBlock(ref_stream.getvalue())
    mob_mol = Chem.MolFromPDBBlock(mob_stream.getvalue())

    assert ref_mol is not None, "RDKit failed to parse the reference ligand."
    assert mob_mol is not None, "RDKit failed to parse the mobile ligand."

    # Correct bond orders with smiles.
    smi_mol = Chem.MolFromSmiles(smiles_string)
    ref_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, ref_mol)
    mob_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, mob_mol)

    # get the prody atom order for the ref moleucle.
    ref_names_prody = list(lig1.getNames())
    ref_coords = lig1.getCoords()

    # get the prody atom order for the mobile molecule.
    mob_names_prody = list(lig2.getNames())
    mob_coords = lig2.getCoords()

    # Compute all the matches between the two rdkit molecule objects.
    #   from rdkit docs: the ordering of the indices corresponds to the atom ordering
    #       in the query. For example, the first index is for the atom in this molecule that matches the first atom in the query.
    all_matches = ref_mol.GetSubstructMatches(mob_mol, uniquify=False, useChirality=use_chirality)

    min_rmsd = (float('inf'), None)
    for match in all_matches:

        # Use the match to get the indices of the atoms in the prody atomgroups.
        curr_ref_indices = [ref_names_prody.index(ref_mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()) for i in match]
        curr_mob_indices = [mob_names_prody.index(mob_mol.GetAtomWithIdx(i).GetPDBResidueInfo().GetName().strip()) for i in range(len(match))]

        # Compute the name mapping from ref to mobile.
        name_mapping = {ref_names_prody[idx1]: mob_names_prody[idx2] for idx1, idx2 in zip(curr_ref_indices, curr_mob_indices)}

        # If we need to drop any atoms, drop them.
        if reference_names_to_ignore is not None:
            index_mask = [(name not in reference_names_to_ignore) for name in [ref_names_prody[idx] for idx in curr_ref_indices]]
            curr_ref_indices = [idx for idx, mask in zip(curr_ref_indices, index_mask) if mask]
            curr_mob_indices = [idx for idx, mask in zip(curr_mob_indices, index_mask) if mask]

        # Get the coordinates of the atoms in the matched order.
        curr_ref_coords = np.array([ref_coords[idx] for idx in curr_ref_indices])
        curr_mob_coords = np.array([mob_coords[idx] for idx in curr_mob_indices])

        # Compute the RMSD.
        lig_rmsd = pr.calcRMSD(curr_ref_coords, curr_mob_coords)

        # Keep the minimum RMSD and name mapping.
        if lig_rmsd < min_rmsd[0]:
            min_rmsd = (lig_rmsd, name_mapping)

    return protein_rmsd, *min_rmsd


def main(input1: os.PathLike, input2: os.PathLike, smiles_string: str, align_on_binding_site: bool = False) -> Tuple[float, float, Dict[str, str]]:
    # Load both as prody atomgroups.
    if input1.suffix == '.pdb':
        ag1 = pr.parsePDB(str(input1))
    elif input1.suffix == '.cif':
        ag1 = pr.parseMMCIF(str(input1))
    else:
        raise ValueError(f"Input files must be either .pdb or .cif files. Got a '{input1.suffix}' file.")
    
    if input2.suffix == '.pdb':
        ag2 = pr.parsePDB(str(input2))
    elif input2.suffix == '.cif':
        ag2 = pr.parseMMCIF(str(input2))
    else:
        raise ValueError(f"Input files must be either .pdb or .cif files. Got a '{input2.suffix}' file.")

    return _main(ag1, ag2, smiles_string, align_on_binding_site=align_on_binding_site)


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description='Compute the minimum RMSD between two ligands.')
    argparser.add_argument('input1', type=Path, help='The first input file.')
    argparser.add_argument('input2', type=Path, help='The second input file.')
    argparser.add_argument('smiles_string', type=str, help='The smiles string of the ligand.')
    argparser.add_argument('--binding_site', '-b', action='store_true', help='Align on the binding site.')

    parsed_args = argparser.parse_args()

    protein_rmsd, min_rmsd, name_mapping = main(Path(parsed_args.input1), Path(parsed_args.input2), parsed_args.smiles_string, align_on_binding_site=parsed_args.binding_site)
    print(protein_rmsd, min_rmsd, name_mapping)