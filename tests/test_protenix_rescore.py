"""Offline unit tests for the Protenix structural-confidence rescore additions.

These exercise the pure helpers (no GPU, no NISE model env): the objective-function
dispatch, the Protenix input-JSON builder, and the summary-confidence parser. Heavy
deps are mocked so the module imports on any machine; main() is __main__-guarded so
importing does not launch a campaign.

Run:
    python tests/test_protenix_rescore.py     # or: pytest tests/test_protenix_rescore.py
"""

import sys
import json
import tempfile
from pathlib import Path
from unittest import mock

# Mock heavy / GPU / model deps so we can import the script and test pure helpers.
_HEAVY = [
    "wandb", "torch", "plotly", "plotly.express", "prody", "pandas", "numpy",
    "rdkit", "rdkit.Chem", "rdkit.Chem.AllChem", "rdkit_to_params",
    "LASErMPNN", "LASErMPNN.run_inference", "LASErMPNN.run_batch_inference",
    "utility_scripts", "utility_scripts.burial_calc",
    "utility_scripts.calc_symmetry_aware_rmsd",
]
for _m in _HEAVY:
    sys.modules.setdefault(_m, mock.MagicMock())

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_nise_boltz2x as nise  # noqa: E402


def test_get_protenix_input_entry():
    e = nise.get_protenix_input_entry("chunk_0_seq_3", "MKEA", "c1ccccc1")
    assert e["name"] == "chunk_0_seq_3"
    assert e["covalent_bonds"] == []
    seqs = e["sequences"]
    # protein is chain 0, ligand chain 1 (matches the Boltz complex order)
    assert seqs[0]["proteinChain"]["sequence"] == "MKEA"
    assert seqs[1]["ligand"]["ligand"] == "c1ccccc1"


def test_consensus_objectives():
    d = {"iptm": 0.8, "protenix_iptm": 0.5, "design_ligand_plddt": 0.9,
         "protenix_ligand_plddt": 0.6, "affinity_probability_binary": 0.7}
    assert nise.compute_objective_function(d, "iptm_consensus") == 0.5          # min(0.8, 0.5)
    assert nise.compute_objective_function(d, "ligand_plddt_consensus") == 0.6  # min(0.9, 0.6)
    assert abs(nise.compute_objective_function(d, "pbind_and_protenix_iptm") - 1.2) < 1e-9
    # existing objectives are unchanged
    assert nise.compute_objective_function(d, "iptm") == 0.8
    assert nise.compute_objective_function(d, "ligand_plddt") == 0.9


def test_consensus_penalizes_oracle_disagreement():
    # A design Boltz loves but Protenix doubts must score LOWER than one both like.
    hacked = {"iptm": 0.95, "protenix_iptm": 0.20,
              "design_ligand_plddt": 0.95, "protenix_ligand_plddt": 0.25}
    honest = {"iptm": 0.85, "protenix_iptm": 0.82,
              "design_ligand_plddt": 0.88, "protenix_ligand_plddt": 0.80}
    for obj in ("iptm_consensus", "ligand_plddt_consensus"):
        assert (nise.compute_objective_function(honest, obj)
                > nise.compute_objective_function(hacked, obj))


def test_parse_protenix_summaries():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        name = "chunk_1_seq_2"
        sub = root / "predictions" / name
        sub.mkdir(parents=True)
        # synthetic summary (verified schema: chain_plddt is [protein, ligand], 0-1)
        (sub / f"{name}_summary_confidence_sample_0.json").write_text(json.dumps({
            "iptm": 0.73, "ptm": 0.81, "plddt": 88.0,
            "chain_plddt": [0.90, 0.65], "ranking_score": 0.70,
        }))
        m = nise.parse_protenix_summaries(root)
    assert name in m
    assert m[name]["protenix_iptm"] == 0.73
    assert m[name]["protenix_ligand_plddt"] == 0.65   # ligand chain index 1
    assert m[name]["protenix_ranking_score"] == 0.70


if __name__ == "__main__":
    test_get_protenix_input_entry()
    test_consensus_objectives()
    test_consensus_penalizes_oracle_disagreement()
    test_parse_protenix_summaries()
    print("ALL OFFLINE TESTS PASS")
