import numpy as np
import pytest

from qcore import CHEMICAL_ACCURACY, MolecularProblem
from VQE import VQE, AdaptVQE, HardwareEfficient, KUpCCGSD, get_ansatz


def finite_diff(v, x, k=6, h=1e-6):
    e = np.eye(len(x))
    return np.array([(v.cost(x + h * e[i]) - v.cost(x - h * e[i])) / (2 * h) for i in range(min(k, len(x)))])


@pytest.mark.parametrize("ansatz", ["uccsd", "uccgsd", "kupccgsd", "quccsd", "hea"])
def test_adjoint_gradient(ansatz):
    p = MolecularProblem.from_library("LiH")
    v = VQE(p, ansatz)
    x = np.random.default_rng(0).normal(size=v.circuit.n_params) * 0.1
    _, g = v.cost_and_grad(x)
    assert np.allclose(g[:6], finite_diff(v, x), atol=1e-7)


@pytest.mark.parametrize("mapping", ["jw", "bk", "parity"])
def test_uccsd_h2_exact(mapping):
    r = VQE(MolecularProblem.from_library("H2", mapping=mapping), "uccsd").run()
    assert abs(r.error) < 1e-8


def test_uccsd_lih_mapping_invariant():
    es = [VQE(MolecularProblem.from_library("LiH", mapping=m), "uccsd").run().energy for m in ("jw", "bk")]
    assert abs(es[0] - es[1]) < 1e-7
    assert 0 < es[0] - MolecularProblem.from_library("LiH").e_fci < 5e-5  # variational, near-exact


def test_variational_principle_and_symmetry():
    p = MolecularProblem.from_library("H2O")
    r = VQE(p, "uccsd").run()
    assert r.energy >= p.e_fci - 1e-10
    assert abs(r.error) < CHEMICAL_ACCURACY
    assert abs(r.particle_number - p.n_electrons) < 1e-8
    assert abs(r.spin_squared) < 1e-6


def test_hea_with_penalty():
    p = MolecularProblem.from_library("H2")
    r = VQE(p, HardwareEfficient(layers=2), number_penalty=0.5, spin_penalty=0.5).run()
    assert abs(r.error) < 1e-6


def test_kupccgsd_h4():
    p = MolecularProblem.from_library("H4")
    r = VQE(p, KUpCCGSD(k=2), maxiter=3000).run()
    assert abs(r.error) < CHEMICAL_ACCURACY


@pytest.mark.parametrize("pool", ["fermionic", "qeb", "qubit"])
def test_adapt_lih(pool):
    p = MolecularProblem.from_library("LiH")
    r = AdaptVQE(p, pool=pool, grad_tol=1e-3).run()
    assert abs(r.error) < CHEMICAL_ACCURACY
    assert r.energy >= p.e_fci - 1e-10


def test_open_shell_vqe():
    p = MolecularProblem.from_library("O2")
    r = VQE(p, "uccsd").run()
    assert abs(r.error) < CHEMICAL_ACCURACY
    assert abs(r.spin_squared - 2.0) < 1e-2


def test_unknown_ansatz():
    with pytest.raises(ValueError):
        get_ansatz("nope")
