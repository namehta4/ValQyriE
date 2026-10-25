import numpy as np
import pytest

from qcore import CHEMICAL_ACCURACY, MolecularProblem
from VQE import VQE, AdaptVQE, HardwareEfficient, KUpCCGSD, get_ansatz


def finite_diff(v, x, k=6, h=1e-6):
    e = np.eye(len(x))
    return np.array([(v.cost(x + h * e[i]) - v.cost(x - h * e[i])) / (2 * h) for i in range(min(k, len(x)))])


@pytest.mark.parametrize("ansatz", ["uccsd", "uccgsd", "kupccgsd", "quccsd", "hea", "fabric", "lucj", "spa"])
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


def test_screened_uccsd_same_energy():
    p = MolecularProblem.from_library("H2O")
    full, screened = get_ansatz("uccsd"), get_ansatz("uccsd", screen_tol=1e-8)
    assert screened.build(p).n_params < full.build(p).n_params
    e_full, e_scr = VQE(p, full).run().energy, VQE(p, screened).run().energy
    assert abs(e_full - e_scr) < 1e-8
    with pytest.raises(ValueError):
        get_ansatz("uccgsd", screen_tol=1e-8).build(p)


def test_ceo_pool_structure():
    from VQE.adapt import pool_operators
    p = MolecularProblem.from_library("H2")
    ops = pool_operators(p, "ceo")
    singles = [o for o in ops if not o.members]
    ceos = [o for o in ops if o.members]
    assert len(singles) == 2 and len(ceos) == 2          # one opposite-spin support, T_a +- T_b
    for o in ceos:
        assert len(o.generator) == 4 and o.cnots == 9    # OVP-CEOs: 4 Pauli strings (Ramoa et al. Eq. 25-26)
        a, b = o.members
        sa, sb = a.to_pauli_sum(4).to_sparse(), b.to_pauli_sum(4).to_sparse()
        assert abs(sa @ sb - sb @ sa).max() < 1e-12     # same-support QEs commute: MVP form is exact
    lih = MolecularProblem.from_library("LiH")
    sizes = {len(o.members) for o in pool_operators(lih, "ceo") if o.members}
    assert sizes == {2, 3}                               # opposite-spin and same-spin supports
    with pytest.raises(ValueError):
        pool_operators(MolecularProblem.from_library("H2", mapping="bk"), "ceo")


@pytest.mark.parametrize("tetris", [False, True])
def test_ceo_adapt_lih(tetris):
    p = MolecularProblem.from_library("LiH")
    r = AdaptVQE(p, pool="ceo", tetris=tetris).run()
    assert abs(r.error) < CHEMICAL_ACCURACY and r.energy >= p.e_fci - 1e-10
    assert r.cnot_count > 0 and sum(r.operators_per_iteration) == len(r.selected)
    assert 0 < r.cnot_depth <= r.cnot_count
    acc = r.at_accuracy()
    assert acc is not None and acc["cnots"] <= r.cnot_count and acc["iteration"] <= r.n_iterations


def test_tetris_adds_disjoint_operators():
    p = MolecularProblem.from_library("H4")
    ad = AdaptVQE(p, pool="qeb", tetris=True)
    r = ad.run()
    assert abs(r.error) < CHEMICAL_ACCURACY
    assert max(r.operators_per_iteration) > 1            # several operators in one iteration
    plain = AdaptVQE(p, pool="qeb").run()
    assert r.n_iterations < plain.n_iterations
    rng = np.random.default_rng(1)                       # the chosen supports never overlap
    grads = rng.standard_normal(len(ad.ops))
    used = 0
    for k in ad._choose(grads):
        assert not ad.ops[k].support & used
        used |= ad.ops[k].support


@pytest.mark.parametrize("mapping", ["jw", "bk"])
@pytest.mark.parametrize("name", ["fabric", "lucj", "spa"])
def test_new_ansatze_conserve_symmetry(name, mapping):
    p = MolecularProblem.from_library("H4", mapping=mapping)
    c = get_ansatz(name).build(p)
    psi = c.state(0.3 * np.random.default_rng(2).standard_normal(c.n_params))
    assert abs(np.real(np.vdot(psi, p.number_diagonal * psi)) - 4) < 1e-10
    assert abs(np.real(np.vdot(psi, p.sz_diagonal * psi))) < 1e-10
    if name != "lucj":  # LUCJ conserves N and S_z only (as in Motta et al.)
        assert abs(p.spin_squared(psi)) < 1e-10
    assert p.energy(psi) >= p.e_fci - 1e-10


def test_spa_assignment():
    from VQE import SPA
    p = MolecularProblem.from_library("H6")
    sets = SPA().assignment(p)
    assert len(sets) == 3 and sorted(a for s in sets for a in s) == [3, 4, 5]
    assert SPA(pairs=[[3], [4], [5]]).build(p).n_params == 3
    with pytest.raises(ValueError):
        SPA(pairs=[[3, 4], [4], [5]]).assignment(p)


def test_fci_interface_roundtrip():
    from pyscf import fci
    from qcore.fci_interface import from_civec, sector_table, to_civec
    for mol, mp in [("H2O", "jw"), ("O2", "bk"), ("LiH", "parity")]:
        p = MolecularProblem.from_library(mol, mapping=mp)
        t = sector_table(p.mapping, p.n_orbitals, p.n_alpha, p.n_beta)
        psi = np.zeros(1 << p.n_qubits, complex)
        psi[p.sector_indices] = np.random.default_rng(0).standard_normal(p.sector_indices.size)
        psi /= np.linalg.norm(psi)
        ci = to_civec(psi, t).real
        e = fci.direct_spin1.energy(p.h1, p.h2, ci, p.n_orbitals, (p.n_alpha, p.n_beta)) + p.e_core
        assert abs(e - p.energy(psi)) < 1e-10
        assert np.abs(from_civec(ci, t, p.n_qubits) - psi).max() < 1e-12


def test_orbital_optimization():
    from VQE import OrbitalOptimizedVQE, PUCCD, UCCSD
    lih = MolecularProblem.from_library("LiH", active_space=(2, 3))
    r = OrbitalOptimizedVQE(lih, UCCSD()).run()            # UCCSD is exact for 2 electrons
    assert abs(r.error) < 1e-7 and r.energy < r.e_initial
    h4 = MolecularProblem.from_library("H4", bond_length=2.5)
    r = OrbitalOptimizedVQE(h4, PUCCD()).run()
    assert all(b <= a + 1e-9 for a, b in zip(r.vqe_energies, r.vqe_energies[1:]))  # monotone
    assert r.energy >= r.e_casscf - 1e-9 and r.error < 1e-2 < r.e_initial - r.e_casscf
