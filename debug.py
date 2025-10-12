from pathlib import Path
from utility_scripts.trajectory_builder import (
    Boltz2xStructurePredictionStrategy, NISETrajectoryBuilder, KeepHighestValueTopKQueueStrategy, 
    RDKitProtonationStrategy, LASErMPNNSequenceDesignStrategy
)

# Result of `which boltz` with boltz conda environment active.
boltz_executable_path = '/nfs/polizzi/bfry/miniforge3/envs/boltz2_retry/bin/boltz'

lasermpnn_config = {
    'model_weights': '/nfs/polizzi/bfry/programs/LASErMPNN/model_weights/laser_weights_0p1A_nothing_heldout.pt',
    'sequence_temp': 0.5, 'first_shell_sequence_temp': 0.7, 
    'chi_temp': 1e-6, 'seq_min_p': 0.0, 'chi_min_p': 0.0,
    'disable_pbar': True, 'disabled_residues_list': ['X', 'C'], # Disables cysteine sampling by default.

    # ==================================================================================================== 
    # Optional: Pass a prody selection string of the form ('resnum 1 or resnum 3 or resnum 5...') to 
    # specify residues over which to constrain the sampling of ALA or GLY residues. 
    # This string can be generated using './identify_surface_residues.ipynb'
    # ==================================================================================================== 
    # 'budget_residue_sele_string': None, 
    # 'ala_budget': None, 'gly_budget': None, 
    'budget_residue_sele_string': 'protein', 
    'ala_budget': 4, 'gly_budget': 0, # May sample up to 4 Ala and 0 Gly over the selected region if not None.

    'disable_charged_fs': True, # Disables sampling D,E,K,R residues for buried residues around the ligand. 
    'fs_calc_ca_distance': 10.0, # Distance cutoff for defining first shell residues around the ligand.
    'fs_calc_burial_hull_alpha_value': 9.0, # Alpha value for defining burial of residues around the ligand, check out https://github.com/benf549/CARPdock/blob/main/visualize_hull.ipynb to tune.
    'sequences_sampled_per_backbone': 64,
}


class DesignScoreModule:
    def __init__(self):
        pass
        
    def get_score() -> float:
        raise NotImplementedError

    def get_log_metadata() -> float:
        raise NotImplementedError
    

trajectory_builder = NISETrajectoryBuilder(
    input_backbones_path=Path('./debug_campaign/input_backbones/'), 
    ligand_smiles = 'CC[C@]1(O)C2=C(C(N3CC4=C5[C@@H]([NH3+])CCC6=C5C(N=C4C3=C2)=CC(F)=C6C)=O)COC1=O', 
    devices = ['cuda:0', 'cuda:1'], num_iterations = 10, debug = True
)
trajectory_builder.set_queue_strategy(KeepHighestValueTopKQueueStrategy(
    max_queue_size=5, keep_input_backbones=False, keep_best_generator_backbone=True
))
trajectory_builder.set_protonation_strategy(RDKitProtonationStrategy())
trajectory_builder.set_sequence_design_strategy(LASErMPNNSequenceDesignStrategy(lasermpnn_config=lasermpnn_config))
trajectory_builder.set_structure_prediction_strategy(Boltz2xStructurePredictionStrategy(path_to_boltz_executable=boltz_executable_path))

# Run the trajectory
trajectory = trajectory_builder.get_trajectory()
trajectory.run_trajectory()