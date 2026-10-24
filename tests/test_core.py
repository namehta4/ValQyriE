"""Pauli algebra, mappings, Hamiltonians and the simulator against independent dense references."""
import numpy as np
import pytest
from scipy.linalg import expm

from qcore import Circuit, MolecularProblem, PauliSum, get_mapping
from qcore.excitations import doubles, excitation_generator, singles

I2 = np.eye(2)
PAULI = {"I": I2, "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.diag([1, -1])}


def dense_label(n, label):
    """Kron in little-endian convention (qubit 0 = least significant bit)."""
    ops = {int(t[1:]): t[0] for t in label.split()}
    m = np.array([[1.0]])
    for q in reversed(range(n)):
        m = np.kron(m, PAULI[ops.get(q, "I")])
    return m


def random_label(rng, n):
    return " ".join(f"{p}{q}" for q, p in enumerate(rng.choice(list("IXYZ"), n)) if p != "I")


def test_pauli_strings_match_dense():
    rng = np.random.default_rng(0)
    n = 4
    for _ in range(30):
        la, lb = random_label(rng, n), random_label(rng, n)
        a, b = PauliSum.from_string(n, la, 0.3), PauliSum.from_string(n, lb, -1.2j)
        assert np.allclose(a.to_dense(), 0.3 * dense_label(n, la))
        assert np.allclose((a * b).to_dense(), a.to_dense() @ b.to_dense())
        assert np.allclose((a + b).dagger().to_dense(), (a + b).to_dense().conj().T)
        psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        assert np.allclose((a + b).apply(psi), (a + b).to_dense() @ psi)


def test_sparse_fwht_path_matches_direct():
    rng = np.random.default_rng(1)
    n = 5
    labels = [random_label(rng, n) for _ in range(60)]
    op = sum((PauliSum.from_string(n, l, rng.normal()) for l in labels), PauliSum.zero(n))
    dense = np.zeros((1 << n, 1 << n), complex)
    for l, c in op.terms():
        dense += c * dense_label(n, l)
    assert np.allclose(op.to_dense(), dense)


@pytest.mark.parametrize("name", ["jw", "bk", "parity"])
def test_canonical_anticommutation(name):
    n = 6
    m = get_mapping(name, n)
    a = [m.annihilation(p).to_dense() for p in range(n)]
    for p in range(n):
        for q in range(n):
            assert np.allclose(a[p] @ a[q].conj().T + a[q].conj().T @ a[p], np.eye(1 << n) * (p == q))
            assert np.allclose(a[p] @ a[q] + a[q] @ a[p], 0)


@pytest.mark.parametrize("name", ["jw", "bk", "parity"])
def test_occupation_decoding(name):
    m = get_mapping(name, 6)
    rng = np.random.default_rng(2)
    for _ in range(10):
        occ = rng.integers(0, 2, 6)
        assert np.array_equal(m.occupations(np.array([m.encode(occ)]))[0], occ)


@pytest.mark.parametrize("mapping", ["jw", "bk", "parity"])
@pytest.mark.parametrize("mol", ["H2", "LiH", "H2O"])
def test_hamiltonian_reproduces_pyscf(mol, mapping):
    p = MolecularProblem.from_library(mol, mapping=mapping)
    hf = np.zeros(1 << p.n_qubits, complex)
    hf[p.hf_index] = 1
    assert abs(p.energy(hf) - p.e_hf) < 1e-9
    e, psi = p.exact_ground_state()
    assert abs(e - p.e_fci) < 1e-8
    assert p.hamiltonian.is_hermitian()


def test_literature_values():
    # H2/STO-3G at 0.7414 A (e.g. O'Malley et al. PRX 6, 031007; OpenFermion data)
    p = MolecularProblem.from_library("H2")
    assert len(p.hamiltonian) == 15
    assert abs(p.e_fci - (-1.1372701746)) < 1e-8
    assert len(MolecularProblem.from_library("LiH").hamiltonian) == 631


def test_open_shell_triplet():
    p = MolecularProblem.from_library("O2")
    e, psi = p.exact_ground_state()
    assert abs(e - p.e_fci) < 1e-8
    assert abs(p.spin_squared(psi) - 2.0) < 1e-6


@pytest.mark.parametrize("mapping", ["jw", "bk"])
def test_block_rotation_is_exact(mapping):
    p = MolecularProblem.from_library("H4", mapping=mapping)
    rng = np.random.default_rng(3)
    psi0 = rng.normal(size=1 << p.n_qubits) + 1j * rng.normal(size=1 << p.n_qubits)
    psi0 /= np.linalg.norm(psi0)
    for exc in singles(4, 2, 2)[:3] + doubles(4, 2, 2)[:5]:
        g = excitation_generator(p.mapping, exc)
        c = Circuit(p.n_qubits, psi0)
        c.add_generator(g, c.new_parameter())
        ref = expm(0.731 * g.to_pauli_sum(p.n_qubits).to_dense()) @ psi0
        assert np.allclose(c.state([0.731]), ref, atol=1e-12)
