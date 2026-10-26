"""Benchmark harness: cases round-trip through files, the reference passes its own gate,
and a wrong implementation is rejected."""
import json

import numpy as np
import pytest

from bench import gate
from bench.__main__ import main
from bench.cases import CASES, CaseData, build_case
from bench.interface import ReferenceEvaluator, load_impl


def test_case_roundtrip(tmp_path):
    case = build_case("lih-12q")
    case.save(tmp_path)
    back = CaseData.load(tmp_path / "lih-12q.npz")
    assert back.excitations == case.excitations and back.hf_index == case.hf_index
    assert np.allclose(back.h2, case.h2) and back.e_core == case.e_core
    meta = json.loads((tmp_path / "lih-12q.json").read_text())
    assert meta["n_qubits"] == 12 and meta["n_params"] == case.n_params


def test_reference_matches_stored_answers():
    case = build_case("lih-12q")
    ref = gate.load_reference("lih-12q")
    impl = ReferenceEvaluator(case)
    for pt in ref["points"]:
        e, g = impl.energy_and_gradient(pt["theta"])
        assert gate.check_point(e, g, pt, ref["e_fci"])["passed"]
    assert abs(ref["points"][0]["energy"] - case.e_hf) < 1e-10  # theta = 0 is the HF state


def test_gate_rejects_wrong_results():
    pt = gate.load_reference("lih-12q")["points"][1]
    assert not gate.check_point(pt["energy"] + 1e-9, pt["gradient"], pt)["passed"]
    bad = pt["gradient"].copy()
    bad[3] += 1e-6
    assert not gate.check_point(pt["energy"], bad, pt)["passed"]
    assert not gate.check_point(-1e9, pt["gradient"], pt, e_fci=-8.0)["passed"]


def test_cli_run_and_plugin(tmp_path):
    plugin = tmp_path / "plug.py"
    plugin.write_text("from bench.interface import ReferenceEvaluator\n"
                      "class Wrong(ReferenceEvaluator):\n"
                      "    def energy_and_gradient(self, theta):\n"
                      "        e, g = super().energy_and_gradient(theta)\n"
                      "        return e + 1e-6, g\n")
    out = tmp_path / "r.json"
    assert main(["run", "lih-12q", "-o", str(out)]) == 0
    assert json.loads(out.read_text())["gate"] == "PASS"
    assert main(["run", "lih-12q", "--impl", f"{plugin}:Wrong"]) == 1
    assert load_impl(None) is ReferenceEvaluator


def test_vqe_matches_reference():
    assert main(["vqe", "lih-12q"]) == 0


def test_starter_template_passes():
    assert main(["run", "lih-12q", "--impl", "examples/bench/my_evaluator.py:MyEvaluator"]) == 0


def test_cases_are_shipped(monkeypatch):
    # the shipped integrals make every case identical on every machine: PySCF is not needed
    from qcore import MolecularProblem

    def no_pyscf(*a, **k):
        raise AssertionError("build_case must not rebuild a shipped case")
    monkeypatch.setattr(MolecularProblem, "from_library", no_pyscf)
    for name in CASES:
        case = build_case(name)
        ref = gate.load_reference(name)
        assert case.n_params == ref["n_params"] and abs(case.e_hf - ref["e_hf"]) < 1e-10


@pytest.mark.parametrize("name", ["lih-12q", "h2o-16q"])
def test_shipped_case_matches_pyscf_up_to_orbital_signs(name):
    case, fresh = build_case(name), build_case(name, from_pyscf=True)
    assert fresh.excitations == case.excitations and fresh.hf_index == case.hf_index
    assert abs(fresh.e_core - case.e_core) < 1e-10 and abs(fresh.e_hf - case.e_hf) < 1e-10
    assert np.allclose(np.abs(fresh.h1), np.abs(case.h1), atol=1e-8)
    assert np.allclose(np.abs(fresh.h2), np.abs(case.h2), atol=1e-8)


def test_reference_passes_gate_h2o_16q():
    # failed when the case was rebuilt with PySCF on a machine whose LAPACK picks other orbital signs
    assert main(["run", "h2o-16q"]) == 0


@pytest.mark.slow
def test_reference_passes_gate_h2o_20q():
    assert main(["run", "h2o-20q"]) == 0
