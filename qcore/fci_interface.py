"""Exchange of states between the qubit simulator and PySCF's FCI module.

PySCF stores a state of the (N_alpha, N_beta) sector as a CI matrix ``ci[I_alpha, I_beta]``
over occupation strings, with the determinant |I_alpha I_beta> = (alpha creators, ascending)
(beta creators, ascending) |vac>. The simulator's basis state |b> is the encoded occupation
of the interleaved spin orbitals (2p = alpha, 2p+1 = beta). The phase between the two is
found by applying the mapping's own creation operators, so the conversion is exact for every
encoding (Jordan-Wigner, Bravyi-Kitaev, parity).

This gives the simulator access to PySCF's reduced density matrices, which drive orbital
optimization (``VQE.orbital_opt``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pauli import parity_sign


@dataclass
class SectorTable:
    index: np.ndarray        # qubit basis index of each determinant, shape (n_a_strings * n_b_strings,)
    phase: np.ndarray        # psi[index] * phase = PySCF CI coefficient
    shape: tuple[int, int]   # (n_a_strings, n_b_strings)


def sector_table(mapping, n_orbitals: int, n_alpha: int, n_beta: int) -> SectorTable:
    from pyscf.fci import cistring
    astr = np.asarray(cistring.make_strings(range(n_orbitals), n_alpha), dtype=np.int64)
    bstr = np.asarray(cistring.make_strings(range(n_orbitals), n_beta), dtype=np.int64)
    a = np.repeat(astr, bstr.size)
    b = np.tile(bstr, astr.size)
    occ = np.zeros((a.size, 2 * n_orbitals), dtype=bool)
    for p in range(n_orbitals):
        occ[:, 2 * p] = (a >> p) & 1
        occ[:, 2 * p + 1] = (b >> p) & 1
    # interleaved ascending product a^dag_{m1} a^dag_{m2} ... |vac> = amp |idx>, applied right to left
    idx = np.zeros(a.size, dtype=np.uint64)
    amp = np.ones(a.size, dtype=np.complex128)
    for m in reversed(range(2 * n_orbitals)):
        sel = occ[:, m]
        if not sel.any():
            continue
        cr = mapping.creation(m)
        x = np.uint64(cr.x[0])
        assert np.all(cr.x == cr.x[0]), "creation operator spans several X masks"
        coef = sum(c * parity_sign(idx[sel] & np.uint64(z)) for z, c in zip(cr.z, cr.c))
        amp[sel] *= coef
        idx[sel] ^= x
    # interleaved order -> PySCF order (all alpha, then all beta): move each alpha_q left past
    # the occupied beta_p with p < q
    reorder = np.zeros(a.size, dtype=np.int64)
    for q in range(n_orbitals):
        below = np.zeros(a.size, dtype=np.int64)
        for p in range(q):
            below += occ[:, 2 * p + 1]
        reorder += occ[:, 2 * q] * below
    sign = np.where(reorder % 2, -1.0, 1.0)
    # |b> = conj(amp) * interleaved = conj(amp) * sign * |PySCF det>
    return SectorTable(idx, np.conj(amp) * sign, (astr.size, bstr.size))


def to_civec(psi: np.ndarray, table: SectorTable) -> np.ndarray:
    return (psi[table.index] * table.phase).reshape(table.shape)


def from_civec(ci: np.ndarray, table: SectorTable, n_qubits: int) -> np.ndarray:
    psi = np.zeros(1 << n_qubits, dtype=np.complex128)
    psi[table.index] = np.asarray(ci).reshape(-1) / table.phase
    return psi


def _parts(ci):
    ci = np.asarray(ci)
    if np.iscomplexobj(ci):
        return [np.ascontiguousarray(ci.real), np.ascontiguousarray(ci.imag)]
    return [ci]


def make_rdm1(ci, norb, nelec):
    """Spin-traced 1-RDM, real part (all that a real Hamiltonian / real orbital rotation sees)."""
    from pyscf.fci import direct_spin1
    return sum(direct_spin1.make_rdm1(c, norb, nelec) for c in _parts(ci))


def make_rdm1s(ci, norb, nelec):
    from pyscf.fci import direct_spin1
    dms = [direct_spin1.make_rdm1s(c, norb, nelec) for c in _parts(ci)]
    return tuple(sum(d[s] for d in dms) for s in range(2))


def make_rdm12(ci, norb, nelec):
    from pyscf.fci import direct_spin1
    out = [direct_spin1.make_rdm12(c, norb, nelec) for c in _parts(ci)]
    return sum(o[0] for o in out), sum(o[1] for o in out)
