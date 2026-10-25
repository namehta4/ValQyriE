"""Excitation operators (fermionic and qubit) used by ansatze, ADAPT pools and GQE pools.

Spin-orbital convention: mode 2k = spatial orbital k (alpha), 2k+1 = (beta).
An excitation ``(occ, virt)`` denotes T = a^dag_{v1} a^dag_{v2} ... a_{o2} a_{o1}
and its anti-Hermitian generator G = T - T^dag.  Under any linear encoding the
Pauli strings of a single- or double-excitation generator mutually commute,
so exp(theta G) is implemented exactly by a product of Pauli rotations.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from .circuit import Generator
from .mappings import FermionQubitMapping, JordanWigner
from .pauli import PauliSum

Excitation = tuple[tuple[int, ...], tuple[int, ...]]


def _spin(p: int) -> int:
    return p % 2


def occupied_virtual(n_orbitals: int, n_alpha: int, n_beta: int):
    occ = [2 * k for k in range(n_alpha)] + [2 * k + 1 for k in range(n_beta)]
    virt = [2 * k for k in range(n_alpha, n_orbitals)] + [2 * k + 1 for k in range(n_beta, n_orbitals)]
    return sorted(occ), sorted(virt)


def singles(n_orbitals, n_alpha, n_beta, generalized=False) -> list[Excitation]:
    if generalized:
        modes = range(2 * n_orbitals)
        return [((p,), (q,)) for p, q in combinations(modes, 2) if _spin(p) == _spin(q)]
    occ, virt = occupied_virtual(n_orbitals, n_alpha, n_beta)
    return [((i,), (a,)) for i in occ for a in virt if _spin(i) == _spin(a)]


def doubles(n_orbitals, n_alpha, n_beta, generalized=False) -> list[Excitation]:
    """Sz-conserving double excitations."""
    if generalized:
        pairs = list(combinations(range(2 * n_orbitals), 2))
        out = []
        for a in range(len(pairs)):
            for b in range(a + 1, len(pairs)):
                (i, j), (k, l) = pairs[a], pairs[b]
                if {i, j} & {k, l}:
                    continue
                if _spin(i) + _spin(j) == _spin(k) + _spin(l):
                    out.append(((i, j), (k, l)))
        return out
    occ, virt = occupied_virtual(n_orbitals, n_alpha, n_beta)
    return [((i, j), (a, b)) for i, j in combinations(occ, 2) for a, b in combinations(virt, 2)
            if _spin(i) + _spin(j) == _spin(a) + _spin(b)]


def pair_doubles(n_orbitals, n_alpha, n_beta, generalized=False) -> list[Excitation]:
    """Seniority-zero pair excitations (p_alpha p_beta) -> (q_alpha q_beta)."""
    if generalized:
        return [((2 * p, 2 * p + 1), (2 * q, 2 * q + 1)) for p, q in combinations(range(n_orbitals), 2)]
    n_pair = min(n_alpha, n_beta)
    return [((2 * i, 2 * i + 1), (2 * a, 2 * a + 1))
            for i in range(n_pair) for a in range(max(n_alpha, n_beta), n_orbitals)]


def fermionic_generator(mapping: FermionQubitMapping, exc: Excitation) -> Generator:
    occ, virt = exc
    ops = [(v, 1) for v in virt] + [(o, 0) for o in reversed(occ)]
    t = mapping.ladder_product(ops)
    return Generator.from_pauli_sum(t - t.dagger(), label=f"f{occ}->{virt}")


def qubit_generator(n_qubits: int, exc: Excitation) -> Generator:
    """Qubit excitation (QEB, Yordanov et al. PRA 102, 062612): fermionic operator without
    the Jordan-Wigner parity strings.  Only meaningful with the Jordan-Wigner encoding."""
    occ, virt = exc
    t = PauliSum.identity(n_qubits)
    for v in virt:  # Q^dag = (X - iY)/2 = X(1 + Z)/2 in X^x Z^z form
        t = t * PauliSum(n_qubits, [1 << v, 1 << v], [0, 1 << v], [0.5, 0.5])
    for o in occ:   # Q = (X + iY)/2 = X(1 - Z)/2
        t = t * PauliSum(n_qubits, [1 << o, 1 << o], [0, 1 << o], [0.5, -0.5])
    return Generator.from_pauli_sum(t - t.dagger(), label=f"q{occ}->{virt}")


def excitation_generator(mapping: FermionQubitMapping, exc: Excitation, kind: str = "fermionic") -> Generator:
    if kind == "fermionic":
        return fermionic_generator(mapping, exc)
    if kind == "qubit":
        if not isinstance(mapping, JordanWigner):
            raise ValueError("qubit excitations require the Jordan-Wigner mapping")
        return qubit_generator(mapping.n_qubits, exc)
    raise ValueError(f"unknown excitation kind {kind!r}")


SCREEN_TOL = 1e-8


def screen_excitations(problem, excitations, tol: float = SCREEN_TOL) -> list[Excitation]:
    """Drop double excitations that cannot lower the energy because of the molecule's symmetry.

    A double is kept if its energy gradient at the Hartree-Fock state, |<HF|[H, G]|HF>|, exceeds
    ``tol``. Doubles that change the spatial symmetry have an exactly zero gradient (and an
    optimal amplitude of zero). Singles are always kept: Brillouin's theorem makes every single's
    HF gradient vanish, so the test cannot tell them apart. Meant for occupied -> virtual
    excitations; generalized ones can have a zero HF gradient and still matter.
    """
    h = problem.hamiltonian_sparse
    hf = np.zeros(1 << problem.n_qubits, dtype=np.complex128)
    hf[problem.hf_index] = 1.0
    h_hf = h @ hf
    kept = []
    for e in excitations:
        if len(e[0]) == 1:
            kept.append(e)
            continue
        g = fermionic_generator(problem.mapping, e).to_pauli_sum(problem.n_qubits).apply(hf)
        if abs(2 * np.real(np.vdot(h_hf, g))) > tol:
            kept.append(e)
    return kept


def pauli_string_pool(mapping: FermionQubitMapping, excitations) -> list[Generator]:
    """Qubit-ADAPT pool (Tang et al. PRX Quantum 2, 020310): individual Pauli strings of the
    fermionic generators with Z operators removed, each as its own generator i*sigma."""
    seen, out = set(), []
    for exc in excitations:
        g = fermionic_generator(mapping, exc)
        for x, z in zip(g.x, g.z):
            z_red = z & x  # keep only Y components (drop parity Z's)
            key = (x, z_red)
            if key in seen:
                continue
            seen.add(key)
            # need an odd number of Y's for a real (number-preserving-friendly) generator
            if bin(x & z_red).count("1") % 2 == 1:
                out.append(Generator([x], [z_red], [1.0], label=f"P(x={x:#x},z={z_red:#x})"))
    return out
