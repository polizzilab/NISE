"""Offline unit tests for the counter-target selectivity objectives.

Exercises the pure helpers (no GPU / model env): the selectivity objective
dispatch and the Boltz binding-metric parser. Heavy deps are mocked so the module
imports anywhere; main() is __main__-guarded so importing does not run a campaign.

Run:
    python tests/test_selectivity.py     # or: pytest tests/test_selectivity.py
"""

import sys
import json
import math
import tempfile
from pathlib import Path
from unittest import mock

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


def test_selectivity_objectives():
    d = {"iptm": 0.8, "offtarget_iptm": 0.3,
         "affinity_probability_binary": 0.7, "offtarget_affinity_probability_binary": 0.2}
    assert abs(nise.compute_objective_function(d, "iptm_selectivity") - 0.5) < 1e-9
    assert abs(nise.compute_objective_function(d, "pbind_selectivity") - 0.5) < 1e-9
    # existing objectives unaffected
    assert nise.compute_objective_function(d, "iptm") == 0.8


def test_selectivity_rewards_discrimination():
    # A design that binds on-target but not off-target must beat one that binds both.
    selective = {"iptm": 0.85, "offtarget_iptm": 0.20}
    promiscuous = {"iptm": 0.90, "offtarget_iptm": 0.85}
    assert (nise.compute_objective_function(selective, "iptm_selectivity")
            > nise.compute_objective_function(promiscuous, "iptm_selectivity"))


def _write_boltz_design(root, name, iptm, pbind=None):
    pred = root / name
    pred.mkdir(parents=True)
    model = pred / f"{name}_model_0.pdb"
    model.write_text("")
    (pred / f"confidence_{name}_model_0.json").write_text(json.dumps({"iptm": iptm}))
    if pbind is not None:
        (pred / f"affinity_{name}.json").write_text(
            json.dumps({"affinity_probability_binary": pbind}))
    return model


def test_parse_boltz_binding_metrics():
    with tempfile.TemporaryDirectory() as d:
        model = _write_boltz_design(Path(d), "chunk_0_seq_1", iptm=0.66, pbind=0.44)
        m = nise.parse_boltz_binding_metrics(model, predict_affinity=True)
    assert m["iptm"] == 0.66
    assert m["affinity_probability_binary"] == 0.44


def test_parse_without_affinity():
    with tempfile.TemporaryDirectory() as d:
        model = _write_boltz_design(Path(d), "chunk_0_seq_0", iptm=0.5)
        m = nise.parse_boltz_binding_metrics(model, predict_affinity=False)
    assert m["iptm"] == 0.5
    assert math.isnan(m["affinity_probability_binary"])


if __name__ == "__main__":
    test_selectivity_objectives()
    test_selectivity_rewards_discrimination()
    test_parse_boltz_binding_metrics()
    test_parse_without_affinity()
    print("ALL SELECTIVITY TESTS PASS")
