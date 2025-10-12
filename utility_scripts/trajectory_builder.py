from typing import *
import subprocess
from collections.abc import Iterable
from pathlib import Path

import io
import yaml
import wandb
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit_to_params import Params
import prody as pr

from abc import ABC, abstractmethod

# TODO: refactor all strategy class types to separate file(s).


class QueueStrategyBase(ABC):
    def __init__(self, keep_input_backbones: bool, keep_best_generator_backbone: bool):
        self.backbone_queue: List[Tuple[Path, float]] = []
        self.keep_input_backbones = keep_input_backbones
        self.keep_best_generator_backbone = keep_best_generator_backbone

    @abstractmethod
    def add_backbones_to_queue(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def get_queue(self):
        return self.backbone_queue


class ProtonationStrategyBase(ABC):
    @abstractmethod
    def protonate_structure(self) -> None:
        raise NotImplementedError("This method should be overridden by subclasses")


class NoProtonationStrategy(ProtonationStrategyBase):
    """A no-op protonation strategy that does nothing."""
    def __init__(self):
        super().__init__()

    def protonate_structure(self) -> None:
        return


class SequenceDesignStrategyBase(ABC):
    @abstractmethod
    def design_sequences(self, *args):
        raise NotImplementedError("This method should be overridden by subclasses")


class StructurePredictionStrategyBase(ABC):
    @abstractmethod
    def predict_structure(self, parent_dir: Path, meta_df: pd.DataFrame, ligand_smiles: str) -> pd.DataFrame:
        raise NotImplementedError("This method should be overridden by subclasses")


class LoggingStrategyBase(ABC):
    """
    Class for logging during the trajectory run.
    """
    def __init__(self):
        self.iteration_index = 0

    @abstractmethod
    def log(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def set_iteration_index(self, iteration_index: int):
        self.iteration_index = iteration_index


class DefaultLoggingStrategy(LoggingStrategyBase):
    """
    Simple print logging.
    """
    def __init__(self, debug: bool):
        super().__init__()
        self.debug = debug

    def log(self):
        print('log!')


class NISETrajectoryBuilder:
    """
    Builder class for constructing a NISE trajectory run.
    Provides methods to set various configuration options and strategies.
    """
    def __init__(
        self, input_backbones_path: Union[Path, str], 
        ligand_smiles: str, devices: List[str], num_iterations: int, debug: bool = False,
        ligand_selection_string: str = 'not protein'
    ):
        super().__init__()

        self.nise_trajectory_config = {}
        self.nise_trajectory_config['input_backbones_path'] = Path(input_backbones_path)
        self.nise_trajectory_config['devices'] = devices
        self.nise_trajectory_config['ligand_smiles'] = ligand_smiles
        self.nise_trajectory_config['num_iterations'] = num_iterations
        self.nise_trajectory_config['debug'] = debug

        # Defines how to select the ligand out of the input file. 
        # Generally 'not protein' is sufficient but amino-acid like ligands might get missed by this.
        self.nise_trajectory_config['ligand_selection_string'] = ligand_selection_string

        self.nise_trajectory_config['protonation_strategy'] = NoProtonationStrategy()
        self.nise_trajectory_config['logging_strategy'] = DefaultLoggingStrategy(debug=debug)

    def set_queue_strategy(self, queue_strategy: QueueStrategyBase) -> 'NISETrajectoryBuilder':
        self.nise_trajectory_config['queue_strategy'] = queue_strategy
        return self

    def set_protonation_strategy(self, protonation_strategy: ProtonationStrategyBase) -> 'NISETrajectoryBuilder':
        self.nise_trajectory_config['protonation_strategy'] = protonation_strategy
        return self
    
    def set_sequence_design_strategy(self, sequence_design_strategy: SequenceDesignStrategyBase) -> 'NISETrajectoryBuilder':
        self.nise_trajectory_config['sequence_design_strategy'] = sequence_design_strategy
        return self

    def set_structure_prediction_strategy(self, structure_prediction_strategy: StructurePredictionStrategyBase) -> 'NISETrajectoryBuilder':
        self.nise_trajectory_config['structure_prediction_strategy'] = structure_prediction_strategy
        return self

    def set_logging_strategy(self, logging_strategy: LoggingStrategyBase) -> 'NISETrajectoryBuilder':
        self.nise_trajectory_config['logging_strategy'] = logging_strategy
        return self

    def get_trajectory(self):
        return NISETrajectory(**self.nise_trajectory_config)


class NISETrajectory:
    """
    A NISE Trajectory is defined by a queue of backbones to process, a protonation strategy for preparing the structure for sequence design,
    a sequence design strategy for designing sequences, a structure prediction strategy for predicting structures,
    and a selection criterion for filtering and ranking structures.
    """
    def __init__(
        self, 
        input_backbones_path: Path,
        devices: List[str], 
        num_iterations: int,
        queue_strategy: QueueStrategyBase,
        protonation_strategy: ProtonationStrategyBase,
        sequence_design_strategy: SequenceDesignStrategyBase,
        structure_prediction_strategy: StructurePredictionStrategyBase,
        logging_strategy: LoggingStrategyBase,
        ligand_smiles: str,
        debug: bool,
        ligand_selection_string: str
    ):
        self.queue_strategy = queue_strategy
        self.protonation_strategy = protonation_strategy
        self.sequence_design_strategy = sequence_design_strategy
        self.structure_prediction_strategy = structure_prediction_strategy
        self.logging_strategy = logging_strategy

        self.input_backbones_path = input_backbones_path.expanduser().resolve().absolute()
        self.devices = devices
        self.num_iterations = num_iterations
        self.ligand_smiles = ligand_smiles
        self.debug = debug
        self.ligand_selection_string = ligand_selection_string

        self.sampled_backbones_path = self.input_backbones_path.parent / 'sampled_backbones'
        self.sampled_dataframes_path = self.input_backbones_path.parent / 'sampled_dataframes'
        self.sampled_backbones_path.mkdir(exist_ok=True)
        self.sampled_dataframes_path.mkdir(exist_ok=True)

    def _setup_run(self):
        """
        Prepare input helper files.
        """
        if not self.input_backbones_path.exists():
            raise FileNotFoundError(f'{self.input_backbones_path} does not exist!')

        input_pdbs = sorted(list(self.input_backbones_path.glob('*.pdb')))
        if len(input_pdbs) < 1:
            raise ValueError(f'No input pdbs found in {self.input_backbones_path}.')

        for backbone in input_pdbs:
            # Load protein-ligand complex and make sure the results are not None.
            input_protein = pr.parsePDB(str(backbone.absolute()))
            assert isinstance(input_protein, pr.AtomGroup), f'Error loading backbone: {backbone}. Did not return AtomGroup'
            ligand = input_protein.select(self.ligand_selection_string)
            assert ligand is not None, f'Found no ligand in {backbone} using selection string: {self.ligand_selection_string}.'
            ligand = ligand.copy()

            # Write ligand PDB file contents to stream.
            pdb_stream = io.StringIO()
            pr.writePDBStream(pdb_stream, ligand)
            ligand_string = pdb_stream.getvalue()

            # Sanity check ligand resnames
            lignames_set = set(ligand.getResnames())
            if len(lignames_set) != 1:
                raise ValueError(f'There should only be one ligand per input file but found two ligand resnames: {lignames_set}')

            # Compute SDF representation of input ligand.
            pdb_mol = Chem.MolFromPDBBlock(ligand_string)
            smi_mol = Chem.MolFromSmiles(self.ligand_smiles)
            pdb_mol = AllChem.AssignBondOrdersFromTemplate(smi_mol, pdb_mol)
            pdb_mol = AllChem.AddHs(pdb_mol, addCoords=True)
            AllChem.ComputeGasteigerCharges(pdb_mol)
            Chem.MolToMolFile(pdb_mol, backbone.with_suffix('.sdf'))

            # Compute Rosetta params file for ligand.
            ligname = lignames_set.pop()
            p = Params.from_mol(pdb_mol, name=ligname)
            p.dump(backbone.with_suffix('.params'))

            if self.queue_strategy.keep_input_backbones:
                init_backbone_and_score = (backbone, float('inf'))
            else:
                init_backbone_and_score = (backbone, 0.0)

            self.queue_strategy.backbone_queue.append(
                init_backbone_and_score
            )

    def run_trajectory(self):
        self._setup_run()
        for iteration_index in range(self.num_iterations):
            self.logging_strategy.set_iteration_index(iteration_index)

            iteration_output_path = self.sampled_backbones_path / f'iteration_{iteration_index:03d}'
            iteration_output_path.mkdir(exist_ok=True)

            meta_df = self.sequence_design_strategy.design_sequences(
                iteration_output_path, 
                [x for x,_ in self.queue_strategy.get_queue()], 
                self.debug 
            )

            meta_df = self.structure_prediction_strategy.predict_structure(
                iteration_output_path, meta_df, self.ligand_smiles
            )


            raise NotImplementedError


class KeepHighestValueTopKQueueStrategy(QueueStrategyBase):
    def __init__(
            self, max_queue_size: int, 
            keep_input_backbones: bool = True, 
            keep_best_generator_backbone: bool = True
    ):
        super().__init__(keep_input_backbones=keep_input_backbones, keep_best_generator_backbone=keep_best_generator_backbone)
        self.max_queue_size = max_queue_size

    def add_backbones_to_queue(self):
        raise NotImplementedError


class RDKitProtonationStrategy(ProtonationStrategyBase):
    def __init__(self):
        super().__init__()

        self.ligand_smiles_to_mol = {}

    def protonate_structure(self, pdb_path: Path, ligand_smiles: str):
        # TODO: cache smiles to rdkit mol conversions.
        raise NotImplementedError


class REDUCEProtonationStrategy(ProtonationStrategyBase):
    def __init__(self, reduce_executable_path: str, ligand_3lc: str):
        super().__init__()
        self.reduce_executable_path = reduce_executable_path
        self.ligand_3lc = ligand_3lc

    def protonate_structure(self):
        raise NotImplementedError


class RosettaProtonationStrategy(ProtonationStrategyBase):
    def __init__(self, rosetta_executable_path: str, params_file_path: str):
        super().__init__()
        self.rosetta_executable_path = rosetta_executable_path
        self.params_file_path = params_file_path

    def protonate_structure(self):
        raise NotImplementedError


class LASErMPNNSequenceDesignStrategy(SequenceDesignStrategyBase):
    def __init__(self, lasermpnn_config: dict):
        super().__init__()
        self.lasermpnn_config = lasermpnn_config

    def design_sequences(self, parent_dir: Path,  backbone_queue_paths: Iterable[Path], debug: bool) -> pd.DataFrame:
        # Sanity check for path handling.
        assert len(list(backbone_queue_paths)) == len(set([x.stem for x in backbone_queue_paths])), 'Backbone paths in the queue must have unique stems!'

        # Feed LASErMPNN a text file with backbone.
        laser_inputs_path = parent_dir / 'lasermpnn_inputs.txt'
        with laser_inputs_path.open('w') as f:
            for backbone_path in backbone_queue_paths:
                assert backbone_path.exists(), f'Backbone path {backbone_path} does not exist!'
                f.write(f"{backbone_path.expanduser().resolve().absolute()}\n")
        
        # Prepare LASErMPNN output directory.
        laser_outputs_path = parent_dir / 'laser_outputs'
        laser_outputs_path.mkdir(exist_ok=True)

        weights_path = self.lasermpnn_config['model_weights']
        seqs_per_backbone = self.lasermpnn_config['sequences_sampled_per_backbone'] if not debug else 8
        sequence_temp = self.lasermpnn_config['sequence_temp']
        first_shell_sequence_temp = self.lasermpnn_config['first_shell_sequence_temp']
        disabled_residues_list = ','.join(self.lasermpnn_config['disabled_residues_list'])

        budget_residue_sele_string = self.lasermpnn_config['budget_residue_sele_string']
        ala_budget = self.lasermpnn_config['ala_budget']
        gly_budget = self.lasermpnn_config['gly_budget']
        if budget_residue_sele_string is None and (ala_budget is not None or gly_budget is not None):
            raise ValueError('If ala_budget or gly_budget is provided, budget_residue_sele_string cannot be None!')

        disable_charged_fs = self.lasermpnn_config['disable_charged_fs']
        fs_calc_ca_distance = self.lasermpnn_config['fs_calc_ca_distance'] = 10.0
        fs_calc_burial_hull_alpha_value = self.lasermpnn_config['fs_calc_burial_hull_alpha_value'] = 9.0

        # Construct the subprocess command to run LASErMPNN.
        command = f'python -m LASErMPNN.run_batch_inference {laser_inputs_path.absolute()} {laser_outputs_path.absolute()} {seqs_per_backbone} -w {weights_path} --silent --sequence_temp {sequence_temp} --first_shell_sequence_temp {first_shell_sequence_temp} --disabled_residues {disabled_residues_list}'
        if budget_residue_sele_string is not None:
            if ala_budget is None or gly_budget is None:
                raise ValueError('If budget_residue_sele_string is provided, ala_budget and gly_budget cannot be None! If you do not want to restrict one of these but do want to restrict the other, set it to a high number (ex: 1000).')
            command += f' --budget_residue_sele_string "{budget_residue_sele_string}" --ala_budget {ala_budget} --gly_budget {gly_budget}'
        
        if disable_charged_fs:
            command += f' --disable_charged_fs --fs_calc_ca_distance {fs_calc_ca_distance} --fs_calc_burial_hull_alpha_value {fs_calc_burial_hull_alpha_value}'

        # Run the constructed subprocess command.
        print(command)
        subprocess.run(command, shell=True)

        # Get an initial dataframe of all the designed sequences.
        all_data = []
        backbone_queue_paths_stem_to_full = {x.stem: x for x in backbone_queue_paths}
        for file in sorted(laser_outputs_path.glob('*/*.pdb')):
            design_data = {'parent_pdb_path': backbone_queue_paths_stem_to_full[file.parent.stem], 'sequence_design_pdb_path': file}
            all_data.append(design_data)
        meta_df = pd.DataFrame(all_data)
        return meta_df


def break_prody_structure_into_chains(prody_structure: pr.AtomGroup) -> dict:
    chain_dict = {}
    for gr in prody_structure.getHierView():
        protein_residues = gr.select('protein and name CA')
        if protein_residues is None:
            continue
        chain_dict[str(gr.getChid())] = protein_residues.getSequence()
    return chain_dict


def get_boltz_yaml_boilerplate(chain_dict: dict, smiles: str, predict_affinity: bool):
    assert 'X' not in chain_dict, "Chain IDs cannot be 'X' as this is reserved for the ligand."

    yaml_dict = { 
        'version': 1, 'sequences': [
            {'protein': {'id': x, 'sequence': y, 'msa': 'empty'}} for x,y in chain_dict.items()
            ] + [{'ligand': {'id': 'X', 'smiles': smiles}}]
    }

    if predict_affinity:
        yaml_dict['properties'] = [{'affinity': {'binder': 'X'}}]

    output = io.StringIO()
    yaml.dump(yaml_dict, output)
    return output.getvalue()


class Boltz2xStructurePredictionStrategy(StructurePredictionStrategyBase):
    def __init__(self, path_to_boltz_executable: str, disable_nccl_p2p: bool = False, disable_kernels: bool = False):
        super().__init__()
        self.path_to_boltz_executable = path_to_boltz_executable
        self.disable_nccl_p2p = disable_nccl_p2p
        self.disable_kernels = disable_kernels

    def _prepare_inputs(self, parent_dir_stem: str, boltz_inputs_dir: Path, chain_to_sequence_map: List[dict], ligand_smiles: str) -> None:
        for idx, chain_map in enumerate(chain_to_sequence_map):
            file_name = f'{parent_dir_stem}_input_{idx:03d}.yaml'
            with open(boltz_inputs_dir / file_name, 'w') as f:
                f.write(get_boltz_yaml_boilerplate(chain_map, ligand_smiles, True))

    def predict_structure(self, parent_dir: Path, meta_df: pd.DataFrame, ligand_smiles: str) -> pd.DataFrame:
        boltz_inputs_dir = parent_dir / 'boltz_inputs'
        boltz_inputs_dir.mkdir(exist_ok=True)

        meta_df['chain_to_sequence_map'] = meta_df.sequence_design_pdb_path.apply(
            lambda x: break_prody_structure_into_chains(
                pr.parsePDB(str(x))
            )
        )

        self._prepare_inputs(
            parent_dir.stem, boltz_inputs_dir, meta_df.chain_to_sequence_map.tolist(), ligand_smiles
        )

        boltz_outputs_dir = (parent_dir / 'boltz_outputs').absolute()
        command = f'{self.path_to_boltz_executable} predict {boltz_inputs_dir} --out_dir {boltz_outputs_dir}'
        if self.disable_nccl_p2p:
            command = f'NCCL_P2P_DISABLE=1 {command}'

        if self.disable_kernels:
            command += ' --no_kernels'

        print(command)
        subprocess.run(command, shell=True)
        raise NotImplementedError


class WANDBLoggingStrategy(LoggingStrategyBase):
    def __init__(self, wandb_project:str, wandb_entity: str, debug: bool, use_wandb: bool = True):
        super().__init__()

        self.debug = debug
        self.use_wandb = use_wandb and not debug

        if self.use_wandb:
            wandb.init(project=wandb_project, entity=wandb_entity)

    def log(self):
        print('log!')
        if self.use_wandb:
            # wandb.log({})
            raise NotImplementedError
