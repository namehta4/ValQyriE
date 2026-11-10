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


def test_run_without_reference_records_results(tmp_path):
    # the hidden case: no answers on the team's machine, so results are recorded for the judges
    case = build_case("lih-12q")
    case.name = "hidden-x"
    case.save(tmp_path)
    out = tmp_path / "r.json"
    assert main(["run", "--case-file", str(tmp_path / "hidden-x.npz"), "-o", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["gate"] == "UNCHECKED" and rep["case_sha256"] == case.fingerprint()
    ref = gate.load_reference("lih-12q")  # same integrals, so the recorded results match its answers
    for rec, pt in zip(rep["points"], ref["points"]):
        assert gate.check_point(rec["energy"], rec["gradient"], pt)["passed"]


def test_run_single_point(tmp_path):
    out = tmp_path / "r.json"
    assert main(["run", "lih-12q", "--point", "2", "--repeats", "2", "-o", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert [p["point"] for p in rep["points"]] == [2] and rep["timed_point"] == 2
    assert rep["evaluations"] == 3 and rep["repeats"] == 2


def _report(case, gate_="PASS", repeats=4, ranks=1, sha=None):
    pts = [{"point": i, "energy": 0.0, "gradient": []} for i in range(3)]
    return {"case": case, "case_sha256": sha or build_case(case).fingerprint(), "points": pts,
            "gate": gate_, "repeats": repeats, "ranks": ranks, "eval_s_timed": 1.0}


def test_submit_check(tmp_path, capsys):
    from bench.submission import check_tree
    d = tmp_path / "stage2" / "ministage1"
    d.mkdir(parents=True)
    (d / "fixed.json").write_text(json.dumps(_report("h2o-24q")))
    (d / "hidden.json").write_text(json.dumps(_report("lih-12q", gate_="UNCHECKED", repeats=0) | {"case": "hidden-a"}))
    (d / "code.tar.gz").write_bytes(b"x")
    (d / "notes.md").write_text("notes")
    assert check_tree(tmp_path) == {"2.1": []}
    assert main(["submit-check", str(tmp_path)]) == 0

    (d / "fixed.json").write_text(json.dumps(_report("h2o-24q", gate_="FAIL", repeats=1, sha="0")))
    (d / "notes.md").unlink()
    (tmp_path / "stage9" / "ministage1").mkdir(parents=True)
    res = check_tree(tmp_path)
    text = " ".join(res["2.1"])
    assert "gate FAIL" in text and "--repeats 4" in text and "differs from the shipped" in text
    assert "notes.md: missing" in text and res["?"] == ["unknown folder stage9/ministage1"]
    assert main(["submit-check", str(tmp_path)]) == 1
