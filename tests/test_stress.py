"""Stress tool: workloads verify themselves, faults are detected, the runner aggregates correctly."""
import numpy as np
import pytest

from stress import peaks, runner
from stress.__main__ import main
from stress.sysinfo import collect, flops_per_cycle_fp64
from stress.workloads import DGEMM, Statevector, Triad, VQEEval, vqe_payload


def test_sysinfo_and_ceilings():
    si = collect()
    assert si.physical_cores >= 1 and len(si.core_cpus) == si.physical_cores
    fpc, basis = flops_per_cycle_fp64(si)
    assert basis
    theo = peaks.theoretical(si, freq_ghz=2.0, flops_per_cycle=16)
    assert theo["peak_fp64_gflops"] == pytest.approx(si.physical_cores * 32)
    cap = peaks.capacity(si)
    if cap:
        assert cap["max_qubits_state_only"] > cap["max_qubits_circuit_sim"] >= cap["max_qubits_vqe_with_sparse_h"]


@pytest.mark.parametrize("wl", [DGEMM(n=128), Statevector(n_qubits=12, n_rotations=8), Triad(n_elements=1 << 16)])
def test_workloads_pass_and_are_deterministic(wl):
    w1, fp1, e1 = wl.step()
    w2, fp2, e2 = wl.step()
    assert e1 is None and e2 is None and w1 > 0 and fp1 == fp2


def test_vqe_workload():
    wl = VQEEval(vqe_payload("H2"))
    _, fp1, err = wl.step()
    assert err is None and wl.step()[1] == fp1


def test_dgemm_detects_corruption():
    wl = DGEMM(n=64)
    c = wl.a @ wl.b
    assert wl.check(c) is None
    bad = c.copy()
    bad[3, 5] += 1.0                      # wrong integer
    assert "Freivalds" in wl.check(bad)
    bad = c.copy()
    bad.view(np.uint64)[0, 0] ^= 1        # single flipped mantissa bit
    assert wl.check(bad) is not None


def test_statevector_detects_corruption():
    wl = Statevector(n_qubits=10, n_rotations=4)
    # perturb one forward rotation by 1e-8 rad: the inverse no longer cancels it
    orig = wl._rot
    calls = {"n": 0}

    def faulty(psi, x, z, phase, a):
        calls["n"] += 1
        return orig(psi, x, z, phase, a + (1e-8 if calls["n"] == 1 else 0.0))

    wl._rot = faulty
    assert wl.step()[2] is not None


def test_runner_two_workers():
    res = runner.run([("dgemm", {"n": 128}), ("statevector", {"n_qubits": 10, "n_rotations": 4})],
                     [0, 0], duration=1.0)
    assert not res["lost_workers"]
    for a in res["aggregate"].values():
        assert a["steps"] > 0 and a["fails"] == 0 and a["inconsistent_cpus"] == []


def test_cli_short_run(tmp_path):
    out = tmp_path / "r.json"
    rc = main(["run", "--duration", "1", "--no-peak", "--workers", "2", "--workloads", "dgemm,statevector",
               "--qubits", "10", "--dgemm-n", "128", "--interval", "1", "-o", str(out)])
    assert rc == 0 and out.exists()


def test_build_problem_spec():
    from stress.workloads import build_problem
    p = build_problem({"molecule": "LiH", "bond_length": 2.0, "basis": "sto-3g", "active_space": [2, 3]})
    assert p.n_qubits == 6 and "r=2.0000" in p.name
    q = build_problem({"geometry": "H 0 0 0; H 0 0 0.74", "name": "H2x"})
    assert q.n_qubits == 4 and q.name == "H2x"
    for bad in ({"molecule": "H2", "charge": 1}, {"geometry": "H 0 0 0; H 0 0 1", "bond_length": 1},
                {"molecule": "H2", "geometry": "H 0 0 0"}, {"molecule": "H2", "colour": 1},
                {"molecule": "H2", "active_space": 3}):
        with pytest.raises(ValueError):
            build_problem(bad)


def test_cli_case_file(tmp_path):
    import json
    cfg = tmp_path / "case.json"
    cfg.write_text(json.dumps({"workloads": ["vqe"], "duration": 100, "no_peak": True, "workers": "2",
                               "vqe": {"molecule": "LiH", "bond_length": 2.0, "active_space": [2, 3]}}))
    out = tmp_path / "r.json"
    assert main(["run", "--config", str(cfg), "--duration", "1", "--interval", "1", "-o", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["args"]["duration"] == 1 and rep["args"]["vqe_problem"]["bond_length"] == 2.0
    cfg.write_text(json.dumps({"mode": "burst"}))
    with pytest.raises(SystemExit):
        main(["run", "--config", str(cfg)])


def test_vqe_workload_screened():
    full = vqe_payload({"molecule": "LiH"})
    scr = vqe_payload({"molecule": "LiH", "screen": True})
    assert scr["screened"] and scr["circuit"].n_params < full["circuit"].n_params
    assert VQEEval(scr).step()[2] is None
    with pytest.raises(ValueError):
        vqe_payload({"molecule": "LiH", "screen": "yes"})


def test_vqe_workload_auto_active_space():
    pay = vqe_payload({"molecule": "H2O", "active_space": {"method": "auto", "max_qubits": 8}})
    assert pay["n_qubits"] <= 8 and pay["auto_active"]["orbitals"]
    assert VQEEval(pay).step()[2] is None


def test_short_mixed_run_warns(tmp_path):
    # duration 0: each worker runs one dgemm step and stops before statevector's turn: untested, so WARN
    import json
    out = tmp_path / "r.json"
    rc = main(["run", "--mode", "mixed", "--duration", "0", "--no-peak", "--workers", "2",
               "--workloads", "dgemm,statevector", "--qubits", "10", "--dgemm-n", "128", "-o", str(out)])
    rep = json.loads(out.read_text())
    assert rc == 0 and rep["verdict"] == "WARN"
    agg = rep["nodes"][0]["phases"][0]["result"]["aggregate"]
    assert agg["dgemm"]["steps"] > 0 and agg["statevector"]["steps"] == 0
    assert agg["statevector"]["inconsistent_cpus"] == []


def test_dead_workers_fail():
    from stress.__main__ import summarize
    phase = {"workloads": ["dgemm"], "result": {"aggregate": {}, "lost_workers": [0, 1], "workers": [],
                                                "telemetry_summary": {}}}
    node = {"host": "n0", "phases": [phase]}
    assert summarize([node], node["phases"], None, {}, 60, lambda *a: None) == "FAIL"
