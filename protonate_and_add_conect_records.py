#!/usr/bin/env python

import io
import argparse
from pathlib import Path
from collections import defaultdict

import prody as pr
from rdkit import Chem
from rdkit.Chem import AllChem


def main(input_pdb_path, output_pdb_path, smiles_string):

    # Load protein into Prody object.
    prot = pr.parsePDB(input_pdb_path)

    # Extract protein and ligand information into separate objects.
    prot_only = prot.select('not hetero').copy()
    lig = prot.select('hetero and not element H').copy()

    setnames = set(lig.getResnames())
    if len(setnames) != 1:
        raise ValueError('There is an issue with the ligand atom names. You might need to rename your ligand or remove any duplicates/extra hetero atoms.')
    tlc = setnames.pop()

    sstream = io.StringIO()
    pr.writePDBStream(sstream, lig)

    # Create ligand from PDB information and smiles string.
    pdb_mol = Chem.MolFromPDBBlock(sstream.getvalue())
    smi_mol = Chem.MolFromSmiles(smiles_string)
    
    # Correct any misinterpreted bond orders.
    pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
    pdb_mol = AllChem.AddHs(pdb_mol, addCoords=True)

    # Extract the connect record info from the RDKit object since prody cannot write these.
    sstream2 = io.StringIO(Chem.MolToPDBBlock(pdb_mol, flavor=(4|8)))
    rdkit_conect = [x for x in sstream2.getvalue().split('\n') if x.startswith('CONECT')]

    # Set the RDKit ligand output resnames, resnums, and chids to be the same.
    modlig = pr.parsePDBStream(sstream2)
    modlig.setResnames(tlc)
    modlig.setResnums(1)
    modlig.setChids('B')
    modlig.setOccupancies(1.0)
    modlig.setBetas(0.0)
    modlig.setTitle(smiles_string)

    # Rename atoms sequentially by element.
    new_atomnames = []
    dd = defaultdict(int)
    elements = [''.join([y for y in x if not y.isnumeric()]) for x in modlig.getNames()]
    for element in elements:
        na = f'{element}{dd[element] + 1}'
        new_atomnames.append(na)
        dd[element] += 1
    modlig.setNames(new_atomnames)

    # Update conect records to be offset by the appropriate atom index once merged with protein atoms.
    prot_len = len(prot_only) + 1
    final_stream = io.StringIO()
    pr.writePDBStream(final_stream, prot_only + modlig)
    offset_conect = []
    for conect in rdkit_conect:
        conect_str, *conect_rcrd = conect.split()
        offset_conect.append(conect_str + ''.join([str(int(x) + prot_len).rjust(5, ' ') for x in conect_rcrd]))

    # write output pdb.
    with open(output_pdb_path, 'w') as f:
        f.write(final_stream.getvalue().rsplit('END', 1)[0] + '\n'.join(offset_conect))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Protonate and add CONECT records to a PDB file')
    parser.add_argument('input', help='Input PDB file path')
    parser.add_argument('smiles', help='SMILES string')
    parser.add_argument('output', help='Output protonated & conect record PDB file path')
    args = parser.parse_args()

    if not Path(args.input).exists():
        raise FileNotFoundError

    main(args.input, args.output, args.smiles)