"""Independent cross-check against OpenFermion (skipped if it is not installed).

The same PySCF integrals are handed to OpenFermion's own second-quantisation and
fermion-to-qubit code; Hamiltonians and excitation generators must agree term by term.
"""
import numpy as np
import pytest

of = pytest.importorskip("openfermion")
from openfermion.chem.molecular_data import spinorb_from_spatial  # noqa: E402

from qcore import MolecularProblem  # noqa: E402
from qcore.excitations import doubles, excitation_generator, singles  # noqa: E402


def of_fermion_hamiltonian(p):
    # OpenFermion convention: two_body[p,q,r,s] multiplies a_p^ a_q^ a_r a_s, built from (ps|qr)
    h1so, h2so = spinorb_from_spatial(p.h1, p.h2.transpose(0, 2, 3, 1))
    return of.get_fermion_operator(of.InteractionOperator(p.e_core, h1so, 0.5 * h2so))


def as_dict(pauli_sum):
    return {tuple((int(t[1:]), t[0]) for t in label.split()): c for label, c in pauli_sum.terms()}


def assert_same_terms(ref, ours, tol=1e-10):
    keys = set(ref) | set(ours)
    worst = max(abs(ref.get(k, 0) - ours.get(k, 0)) for k in keys)
    assert worst < tol, f"max coefficient difference {worst:.3e}"


@pytest.mark.parametrize("mol", ["H2", "LiH", "H2O", "O2"])
def test_jordan_wigner_hamiltonian(mol):
    p = MolecularProblem.from_library(mol, mapping="jw")
    assert_same_terms(of.jordan_wigner(of_fermion_hamiltonian(p)).terms, as_dict(p.hamiltonian))


@pytest.mark.parametrize("mol", ["H2", "LiH", "H2O"])
def test_bravyi_kitaev_hamiltonian(mol):
    p = MolecularProblem.from_library(mol, mapping="bk")
    ref = of.bravyi_kitaev(of_fermion_hamiltonian(p), p.n_qubits)
    assert_same_terms(ref.terms, as_dict(p.hamiltonian))


@pytest.mark.parametrize("mol", ["H2", "LiH"])
def test_parity_hamiltonian(mol):
    p = MolecularProblem.from_library(mol, mapping="parity")
    ref = of.binary_code_transform(of_fermion_hamiltonian(p), of.parity_code(p.n_qubits))
    assert_same_terms(ref.terms, as_dict(p.hamiltonian))


def test_bravyi_kitaev_tree_isospectral():
    # OpenFermion's Fenwick-tree BK lays the tree out differently, so terms differ but
    # the spectrum must be identical.
    p = MolecularProblem.from_library("LiH", mapping="bk")
    ref = of.get_sparse_operator(of.bravyi_kitaev_tree(of_fermion_hamiltonian(p), p.n_qubits))
    w_ref = np.linalg.eigvalsh(ref.toarray())
    w_ours = np.linalg.eigvalsh(p.hamiltonian_sparse.toarray())
    assert np.allclose(w_ref, w_ours, atol=1e-9)


@pytest.mark.parametrize("mol", ["LiH", "H2O"])
def test_ground_state_energy(mol):
    p = MolecularProblem.from_library(mol)
    sparse = of.get_sparse_operator(of.jordan_wigner(of_fermion_hamiltonian(p)), p.n_qubits)
    e_ref, _ = of.jw_get_ground_state_at_particle_number(sparse, p.n_electrons)
    assert abs(e_ref - p.e_fci) < 1e-8
    assert abs(p.exact_ground_state()[0] - e_ref) < 1e-8


def test_excitation_generators():
    p = MolecularProblem.from_library("LiH", mapping="jw")
    args = (p.n_orbitals, p.n_alpha, p.n_beta)
    for occ, virt in singles(*args) + doubles(*args):
        t = of.FermionOperator(tuple((v, 1) for v in virt) + tuple((o, 0) for o in reversed(occ)))
        ref = of.jordan_wigner(t - of.hermitian_conjugated(t))
        ours = excitation_generator(p.mapping, (occ, virt)).to_pauli_sum(p.n_qubits)
        assert_same_terms(ref.terms, as_dict(ours))
