"""Second-quantised electronic Hamiltonian -> qubit Hamiltonian.

    H = E_core + sum_PQ h_PQ a_P^dag a_Q
               + 1/2 sum_PQRS (PR|QS) a_P^dag a_Q^dag a_S a_R

with (pr|qs) chemist-notation MO integrals.  Using
a_P^dag a_Q^dag a_S a_R = E_PR E_QS - delta_QR E_PS  (E_PQ = a_P^dag a_Q),

    H = E_core + sum_PS [h_PS - 1/2 sum_Q (PQ|QS)] E_PS + 1/2 sum (PR|QS) E_PR E_QS,

which is evaluated fully vectorised over spin-allowed pairs.
"""
from __future__ import annotations

import numpy as np

from .mappings import FermionQubitMapping
from .pauli import PauliSum, parity_sign, sum_ops


def _padded_hopping_table(mapping: FermionQubitMapping, pairs):
    w = 4
    ex = np.zeros((len(pairs), w), dtype=np.uint64)
    ez = np.zeros_like(ex)
    ec = np.zeros((len(pairs), w), dtype=np.complex128)
    for i, (p, q) in enumerate(pairs):
        e = mapping.hopping(p, q)
        k = len(e)
        if k > w:
            raise RuntimeError("hopping operator with more than 4 Pauli terms")
        ex[i, :k], ez[i, :k], ec[i, :k] = e.x, e.z, e.c
    return ex, ez, ec


def molecular_qubit_hamiltonian(h1: np.ndarray, eri: np.ndarray, e_core: float,
                                mapping: FermionQubitMapping, tol: float = 1e-12,
                                chunk: int = 100_000) -> PauliSum:
    """Qubit Hamiltonian from spatial-orbital integrals (h1: n x n, eri: n^4 chemist)."""
    norb = h1.shape[0]
    nq = 2 * norb
    if mapping.n_qubits != nq:
        raise ValueError("mapping size does not match 2 * n_orbitals")

    # spin-allowed (P, R) pairs: same spin
    pairs, spatial = [], []
    for s in (0, 1):
        for p in range(norb):
            for r in range(norb):
                pairs.append((2 * p + s, 2 * r + s))
                spatial.append((p, r))
    spatial = np.array(spatial)
    ex, ez, ec = _padded_hopping_table(mapping, pairs)

    # one-body part (with the delta correction from normal ordering)
    h_eff = h1 - 0.5 * np.einsum("pqqs->ps", eri)
    one_c = h_eff[spatial[:, 0], spatial[:, 1]]
    ops = [PauliSum(nq, ex.ravel(), ez.ravel(), (ec * one_c[:, None]).ravel())]

    # two-body part: 1/2 sum_{u,v} (pr|qs) E_u E_v
    v = eri[spatial[:, 0][:, None], spatial[:, 1][:, None],
            spatial[:, 0][None, :], spatial[:, 1][None, :]]
    uu, vv = np.nonzero(np.abs(v) > tol)
    for start in range(0, uu.size, chunk):
        u, w = uu[start:start + chunk], vv[start:start + chunk]
        x = ex[u][:, :, None] ^ ex[w][:, None, :]
        z = ez[u][:, :, None] ^ ez[w][:, None, :]
        sign = parity_sign(ez[u][:, :, None] & ex[w][:, None, :])
        c = 0.5 * v[u, w][:, None, None] * ec[u][:, :, None] * ec[w][:, None, :] * sign
        ops.append(PauliSum(nq, x.ravel(), z.ravel(), c.ravel()))

    ops.append(PauliSum.identity(nq, e_core))
    h = sum_ops(ops, nq)
    h = (h + h.dagger()) * 0.5  # remove round-off anti-Hermitian residue
    return h.chop(tol)


def number_operator(mapping: FermionQubitMapping) -> PauliSum:
    return sum_ops((mapping.number(p) for p in range(mapping.n_modes)), mapping.n_qubits)


def sz_operator(mapping: FermionQubitMapping) -> PauliSum:
    n = mapping.n_modes
    return sum_ops((mapping.number(p) * (0.5 if p % 2 == 0 else -0.5) for p in range(n)),
                   mapping.n_qubits)


def s2_operator(mapping: FermionQubitMapping) -> PauliSum:
    """Total spin S^2 = S_- S_+ + S_z (S_z + 1) for interleaved spin orbitals."""
    norb = mapping.n_modes // 2
    nq = mapping.n_qubits
    s_plus = sum_ops((mapping.hopping(2 * k, 2 * k + 1) for k in range(norb)), nq)
    s_minus = s_plus.dagger()
    sz = sz_operator(mapping)
    return (s_minus * s_plus + sz * sz + sz).chop()
