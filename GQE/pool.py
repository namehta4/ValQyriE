"""GQE operator pool: tokens are (generator, time) pairs, token j -> U_j = exp(t_j G_j).

Following Nakaji et al., "The generative quantum eigensolver (GQE) and its
application for ground state search" (arXiv:2401.09253), the vocabulary is the
Cartesian product of a set of anti-Hermitian generators (by default the UCCSD
excitations, which preserve particle number and S_z) and a discrete grid of
evolution times of both signs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qcore.circuit import Circuit, Generator, Rotation
from qcore.excitations import (SCREEN_TOL, doubles, excitation_generator, pauli_string_pool,
                               screen_excitations, singles)

DEFAULT_TIMES = (0.00625, 0.0125, 0.025, 0.05, 0.1)


@dataclass
class OperatorPool:
    generators: list[Generator]
    times: tuple[float, ...]
    n_qubits: int
    initial_state: int

    def __post_init__(self):
        ts = sorted({float(s) * t for t in self.times for s in (1, -1)})
        self.tokens = [(k, t) for k in range(len(self.generators)) for t in ts]
        # one block list per generator (coeff 1); the token's time is the rotation angle
        self._blocks = []
        for g in self.generators:
            c = Circuit(self.n_qubits)
            c.add_generator(g, param=None, scale=1.0)
            self._blocks.append(c.ops)
        self._idx = np.arange(1 << self.n_qubits, dtype=np.uint64)

    @classmethod
    def from_problem(cls, problem, kind: str = "uccsd", times=DEFAULT_TIMES,
                     screen_tol: float | None = SCREEN_TOL) -> "OperatorPool":
        """Build the pool from excitations of ``problem``.

        ``screen_tol``: drop double excitations whose energy gradient at the HF state,
        |<HF|[H, G]|HF>|, is below this value (symmetry-forbidden doubles); singles are
        always kept since Brillouin's theorem makes their HF gradient vanish.  ``None``
        keeps the full pool.
        """
        n, na, nb, m = problem.n_orbitals, problem.n_alpha, problem.n_beta, problem.mapping
        gen = kind in ("generalized", "qeb", "qubit")
        excs = singles(n, na, nb, gen) + doubles(n, na, nb, gen)
        if kind not in ("uccsd", "generalized", "qeb", "qubit"):
            raise ValueError(f"unknown pool kind {kind!r}")
        if screen_tol is not None:
            excs = screen_excitations(problem, excs, screen_tol)
        if kind == "qubit":
            gens = pauli_string_pool(m, excs)
        else:
            gens = [excitation_generator(m, e, kind="qubit" if kind == "qeb" else "fermionic") for e in excs]
        return cls(gens, tuple(times), problem.n_qubits, problem.hf_index)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def describe(self, seq) -> list[str]:
        return [f"exp({t:+.5f} * {self.generators[k].label})" for k, t in (self.tokens[j] for j in seq)]

    def state(self, seq) -> np.ndarray:
        psi = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        psi[self.initial_state] = 1.0
        idx = self._idx
        for j in seq:
            k, t = self.tokens[int(j)]
            for blk in self._blocks[k]:
                blk: Rotation
                psi = blk.rotate(psi, t, blk.apply_generator(psi, idx), idx)
        return psi

    def parametrized_circuit(self, seq) -> tuple[Circuit, np.ndarray]:
        """Circuit with one free angle per token, initialised to the token times."""
        circ = Circuit(self.n_qubits, self.initial_state)
        x0 = []
        for j in seq:
            k, t = self.tokens[int(j)]
            circ.add_generator(self.generators[k], circ.new_parameter(self.generators[k].label))
            x0.append(t)
        return circ, np.array(x0)
