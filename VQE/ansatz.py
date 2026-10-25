"""Variational ansatze.

Every ansatz implements ``build(problem) -> Circuit`` and ``initial_parameters``.
Chemistry-inspired ansatze start from the Hartree-Fock determinant and are
built from exact exponentials of excitation generators, applied in a fixed
(disentangled / first-order Trotter) product order, optionally repeated
``reps`` times with independent parameters.

Available (``get_ansatz(name, **kw)``):

  uccsd      Unitary coupled cluster, singles + doubles (spin-orbital, Sz-conserving)
  uccd       UCC doubles only
  uccgsd     Generalized UCCSD (all spin-conserving singles/doubles)
  kupccgsd   k-UpCCGSD (Lee et al., JCTC 15, 311 (2019)): generalized pair doubles +
             spin-complemented generalized singles, k layers
  puccd      pair UCCD (seniority-zero)
  quccsd     qubit-excitation-based UCCSD (Yordanov et al.), Jordan-Wigner only
  hea        hardware-efficient ansatz (RY/RZ layers + CNOT/CZ entanglers)
  custom     user-supplied list of Generators / Pauli strings
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qcore.circuit import Circuit, Generator
from qcore.excitations import (doubles, excitation_generator, pair_doubles, screen_excitations, singles)
from qcore.pauli import PauliSum


class Ansatz:
    name = "ansatz"

    def build(self, problem) -> Circuit:
        raise NotImplementedError

    def initial_parameters(self, circuit: Circuit, rng: np.random.Generator | None = None) -> np.ndarray:
        return np.zeros(circuit.n_params)

    def __repr__(self) -> str:
        fields = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in fields.items())})"


@dataclass(repr=False)
class UCC(Ansatz):
    """Unitary coupled-cluster family.

    excitations: "sd", "d", "s"; generalized: use generalized excitations;
    kind: "fermionic" or "qubit"; reps: number of Trotter-like repetitions,
    each with its own parameters (``share_params=True`` -> one set, angle / reps);
    screen_tol: drop symmetry-forbidden doubles (``qcore.excitations.screen_excitations``,
    1e-8 is the usual value; None keeps all). Not available for generalized excitations.
    """
    excitations: str = "sd"
    generalized: bool = False
    kind: str = "fermionic"
    reps: int = 1
    share_params: bool = False
    name: str = "uccsd"
    screen_tol: float | None = None

    def excitation_list(self, problem):
        args = (problem.n_orbitals, problem.n_alpha, problem.n_beta, self.generalized)
        out = []
        if "s" in self.excitations:
            out += singles(*args)
        if "d" in self.excitations:
            out += doubles(*args)
        if self.screen_tol is not None:
            if self.generalized:
                raise ValueError("screening needs occupied -> virtual excitations; not available for "
                                 f"{self.name}")
            out = screen_excitations(problem, out, self.screen_tol)
        return out

    def build(self, problem) -> Circuit:
        circ = Circuit(problem.n_qubits, problem.hf_index)
        excs = self.excitation_list(problem)
        gens = [excitation_generator(problem.mapping, e, self.kind) for e in excs]
        shared = [circ.new_parameter(g.label) for g in gens] if self.share_params else None
        for r in range(self.reps):
            for i, g in enumerate(gens):
                p = shared[i] if shared else circ.new_parameter(f"{g.label}#{r}")
                circ.add_generator(g, p, scale=1.0 / self.reps if shared else 1.0)
        return circ


def UCCSD(**kw) -> UCC:
    return UCC(excitations="sd", name="uccsd", **kw)


def UCCD(**kw) -> UCC:
    return UCC(excitations="d", name="uccd", **kw)


def UCCGSD(**kw) -> UCC:
    return UCC(excitations="sd", generalized=True, name="uccgsd", **kw)


def QUCCSD(**kw) -> UCC:
    return UCC(excitations="sd", kind="qubit", name="quccsd", **kw)


@dataclass(repr=False)
class KUpCCGSD(Ansatz):
    """k-UpCCGSD: per layer, generalized pair doubles + generalized singles where the
    alpha and beta single excitations share one parameter (spin-complemented).
    Parameters start at small random values: all-zero is a saddle point (Brillouin).
    """
    k: int = 2
    init_scale: float = 0.01
    name: str = "kupccgsd"

    def build(self, problem) -> Circuit:
        n, m = problem.n_orbitals, problem.mapping
        circ = Circuit(problem.n_qubits, problem.hf_index)
        from itertools import combinations
        pairs = pair_doubles(n, problem.n_alpha, problem.n_beta, generalized=True)
        for layer in range(self.k):
            for p, q in combinations(range(n), 2):
                param = circ.new_parameter(f"s{p}{q}#{layer}")
                for s in (0, 1):
                    circ.add_generator(excitation_generator(m, ((2 * p + s,), (2 * q + s,))), param)
            for exc in pairs:
                circ.add_generator(excitation_generator(m, exc), circ.new_parameter(f"d{exc}#{layer}"))
        return circ

    def initial_parameters(self, circuit, rng=None):
        rng = rng or np.random.default_rng(0)
        return self.init_scale * rng.standard_normal(circuit.n_params)


@dataclass(repr=False)
class PUCCD(Ansatz):
    """Pair UCCD: seniority-zero pair excitations from occupied to virtual spatial orbitals."""
    name: str = "puccd"

    def build(self, problem) -> Circuit:
        circ = Circuit(problem.n_qubits, problem.hf_index)
        for exc in pair_doubles(problem.n_orbitals, problem.n_alpha, problem.n_beta):
            circ.add_generator(excitation_generator(problem.mapping, exc), circ.new_parameter(str(exc)))
        return circ


@dataclass(repr=False)
class HardwareEfficient(Ansatz):
    """Hardware-efficient ansatz (Kandala et al., Nature 549, 242 (2017) style).

    layers of single-qubit rotations (``rotations``) followed by an entangling
    layer (``entangler`` on ``entanglement`` pairs), plus a final rotation layer.
    ``initial_state``: "hf" (Hartree-Fock determinant) or "zero".
    Does not conserve particle number: use the VQE number/spin penalties.
    """
    layers: int = 2
    rotations: tuple = ("ry", "rz")
    entanglement: str = "linear"  # linear | circular | full
    entangler: str = "cx"         # cx | cz
    initial_state: str = "hf"
    init_scale: float = 0.05
    name: str = "hea"

    def _pairs(self, n):
        if self.entanglement == "linear":
            return [(i, i + 1) for i in range(n - 1)]
        if self.entanglement == "circular":
            return [(i, (i + 1) % n) for i in range(n)] if n > 2 else [(0, 1)]
        if self.entanglement == "full":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        raise ValueError(self.entanglement)

    def _rot_layer(self, circ, n, tag):
        for q in range(n):
            for r in self.rotations:
                label = {"rx": "X", "ry": "Y", "rz": "Z"}[r.lower()]
                circ.add_pauli_rotation(f"{label}{q}", -0.5, circ.new_parameter(f"{r}{q}#{tag}"))

    def build(self, problem) -> Circuit:
        n = problem.n_qubits
        init = problem.hf_index if self.initial_state == "hf" else 0
        circ = Circuit(n, init)
        for layer in range(self.layers):
            self._rot_layer(circ, n, layer)
            for a, b in self._pairs(n):
                circ.add_gate(self.entangler, (a, b))
        self._rot_layer(circ, n, self.layers)
        return circ

    def initial_parameters(self, circuit, rng=None):
        rng = rng or np.random.default_rng(0)
        return self.init_scale * rng.standard_normal(circuit.n_params)


@dataclass(repr=False)
class CustomAnsatz(Ansatz):
    """Product of exp(theta_k G_k) for user-supplied generators.

    ``generators`` may contain ``Generator`` objects, anti-Hermitian ``PauliSum``s, or
    Hermitian Pauli labels such as ``"X0 Y1"`` (meaning exp(i theta sigma)).
    """
    generators: list = field(default_factory=list)
    from_hf: bool = True
    name: str = "custom"

    def build(self, problem) -> Circuit:
        n = problem.n_qubits
        circ = Circuit(n, problem.hf_index if self.from_hf else 0)
        for g in self.generators:
            if isinstance(g, str):
                g = Generator.from_pauli_sum(1j * PauliSum.from_string(n, g), label=g)
            elif isinstance(g, PauliSum):
                g = Generator.from_pauli_sum(g)
            circ.add_generator(g, circ.new_parameter(g.label))
        return circ


ANSATZE = {
    "uccsd": UCCSD, "uccd": UCCD, "uccgsd": UCCGSD, "quccsd": QUCCSD,
    "kupccgsd": KUpCCGSD, "upccgsd": KUpCCGSD, "puccd": PUCCD,
    "hea": HardwareEfficient, "hardware_efficient": HardwareEfficient,
}


def get_ansatz(name: str, **kw) -> Ansatz:
    try:
        return ANSATZE[name.lower().replace("-", "")](**kw)
    except KeyError:
        raise ValueError(f"unknown ansatz {name!r}; choose from {sorted(ANSATZE)}") from None
