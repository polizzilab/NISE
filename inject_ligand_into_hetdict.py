#!/usr/bin/env python

import io
import os
import argparse
import prody as pr
from collections import defaultdict
from typing import *


def get_formula(elements: Sequence) -> str:
    """
    Returns chemical by counting unique elements.
    """
    dd = defaultdict(int)
    for e in elements:
        dd[e] += 1

    return ' '.join([f'{i}{j}' if j != 1 else f'{i}' for i,j in dd.items()])


def write_hetdict_str(hetatm_lines, conect_megadict, three_letter_code, formula, name='NISELIGAND'):
    output_string = ''
    output_string += f'RESIDUE   {three_letter_code}     {len(hetatm_lines)}\n'

    # Loop over each atom and write the reduce-formatted bond information.
    for (atom, connected) in conect_megadict:
        connected_padded = []
        for atom_ in connected:
            if len(atom_) == 2:
                atom_ = f'{atom_} '
            connected_padded.append(atom_)
        
        connected_atoms_justified = ''.join(x.center(5) for x in connected_padded)
        if len(atom) == 2:
            atom = f'{atom} '
        output_string += f'CONECT    {atom.rjust(5)}    {len(connected)}{connected_atoms_justified}\n'

    output_string += 'END\n'
    output_string += f'HET    {three_letter_code}             {len(hetatm_lines)}\n'
    output_string += f'HETSYN     {three_letter_code} {name}\n'
    output_string += f'HETNAM     {three_letter_code} {name}\n'
    output_string += f'FORMUL      {three_letter_code}    {formula}\n'

    print(output_string)
    return output_string


def perform_hetdict_injection(input_hetdict_path, output_hetdict_path, output_string, three_letter_code):
    # Find where to inject the hetdict into the new lines. 
    # Overwrites any existing residues with three letter code name
    new_lines = ''
    prev_line = None
    with open(input_hetdict_path, 'r') as f:
        skip = False
        for line in f.readlines():

            if line.startswith(f'RESIDUE   {three_letter_code}'):
                skip = True
                new_lines += output_string

            if skip and prev_line.startswith('FORMUL'):
                skip = False
            
            prev_line = line
            if skip:
                continue

            new_lines += line

    with open(output_hetdict_path, 'w') as f:
        f.write(new_lines)


def main(input_pdb_path: os.PathLike, input_hetdict_path: str, name: str):
    # Read the pdb file.
    ligand_relevant_lines = open(input_pdb_path, 'r').read()
    ligand_ag = pr.parsePDBStream(io.StringIO(ligand_relevant_lines)).hetero
    three_letter_code = ligand_ag.getResnames()[0]

    formula = get_formula(ligand_ag.getElements())

    # Loop over the lines and extract all HETATM and CONECT record lines.
    hetatm_lines = []
    conect_records = []
    for j in ligand_relevant_lines.split('\n'):
        if j.startswith('HETATM'):
            hetatm_lines.append(j.strip())
        if j.startswith('CONECT'):
            conect_records.append(j.strip())

    # Map from atom index to atom name.
    idx_to_name = {int(x.split()[1]): x.split()[2] for x in hetatm_lines}

    # Store index to connected indices.
    conect_megadict = []
    for idx, line in enumerate(conect_records):
        record = list(line.split())
        conect_dict = (idx_to_name[int(record[1])], [idx_to_name[int(x)] for x in record[2:]])
        conect_megadict.append(conect_dict)
    
    # Create Reduce hetdict formatted lines.
    output_string = write_hetdict_str(hetatm_lines, conect_megadict, three_letter_code, formula, name)

    # Write output hetdict with injected residue.
    perform_hetdict_injection(input_hetdict_path, output_hetdict_path, output_string, three_letter_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inject ligand information into a hetdict file.')
    parser.add_argument('input_pdb', type=str, help='Path to input PDB file. Should probably be the output of `./protonate_and_add_conect_records.py`')

    parser.add_argument('--output_hetdict', '-o', default='modifed_hetdict.txt', type=str, help='Path to output hetdict file. Defaults to modified_hetdict.txt')
    parser.add_argument('--input_hetdict', '-i', type=str, default='reduce_wwPDB_het_dict_two_letter_bug_fixed.txt', help='Path to input hetdict file. Defaults to reduce_wwPDB_het_dict_two_letter_bug_fixed.txt')
    parser.add_argument('--name', '-n', type=str, default='NISELIGAND', help='Name of the ligand for reduce (default: NISELIGAND), doesn\'t really matter but can be grepped for in the output hetdict.')
    args = parser.parse_args()

    # Define output_hetdict_path which is used in perform_hetdict_injection but not passed to main
    output_hetdict_path = args.output_hetdict

    main(args.input_pdb, args.input_hetdict, args.name)