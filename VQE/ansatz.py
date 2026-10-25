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
  fabric     symmetry-preserving gate fabric of Givens rotations (Anselmetti et al., NJP 23,
             113010 (2021)): conserves N, S_z and S^2, nearest-neighbour orbitals only
  lucj       local unitary cluster Jastrow (Motta et al., Chem. Sci. 14, 11213 (2023))
  spa        separable pair approximation (Kottmann & Aspuru-Guzik, PRA 105, 032449 (2022));
             use with orbital optimization (VQE.orbital_opt)
  custom     user-supplied list of Generators / Pauli strings
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from qcore.circuit import Circuit, Generator
from qcore.excitations import (doubles, excitation_generator, imaginary_hopping_generator,
                               number_number_generator, pair_doubles, screen_excitations, singles)
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
    orbital_dependent = True  # pair ansatz: orbital optimization also rotates active orbitals

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
    orbital_dependent = True

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


def _spin_adapted_rotation(circ, mapping, p, q, param, scale=1.0, imaginary=False):
    """exp(scale theta (E_pq - E_qp)) (or its imaginary counterpart) on spatial orbitals p, q:
    the same rotation for alpha and beta, which commute, so two exact exponentials."""
    for s in (0, 1):
        g = (imaginary_hopping_generator(mapping, 2 * q + s, 2 * p + s) if imaginary
             else excitation_generator(mapping, ((2 * p + s,), (2 * q + s,))))
        circ.add_generator(g, param, scale)


def _brick_wall(n):
    """Nearest-neighbour pairs (p, p+1): even bonds, then odd bonds."""
    return [(p, p + 1) for p in range(0, n - 1, 2)] + [(p, p + 1) for p in range(1, n - 1, 2)]


@dataclass(repr=False)
class GateFabric(Ansatz):
    """Quantum-number-preserving gate fabric (Anselmetti, Wierichs, Gogolin & Parrish, New J.
    Phys. 23, 113010 (2021)).

    Each layer is a brick wall of 4-qubit QNP gates on neighbouring spatial orbitals (p, p+1);
    a gate is a pair exchange exp(theta (P^dag_{p+1} P_p - h.c.)) followed by a spin-adapted
    orbital rotation (a Givens rotation, the same for alpha and beta). Every gate conserves
    particle number, S_z and S^2, and under Jordan-Wigner acts on four adjacent qubits only, so
    this is a hardware-style ansatz without the symmetry breaking of ``hea``. ``layers``
    defaults to the number of orbitals (fewer layers are much less accurate). All parameters
    start at small random values (zero is a saddle point).
    """
    layers: int | None = None
    init_scale: float = 0.01
    name: str = "fabric"

    def build(self, problem) -> Circuit:
        n, m = problem.n_orbitals, problem.mapping
        circ = Circuit(problem.n_qubits, problem.hf_index)
        for layer in range(self.layers or n):
            for p, q in _brick_wall(n):
                exc = ((2 * p, 2 * p + 1), (2 * q, 2 * q + 1))
                circ.add_generator(excitation_generator(m, exc), circ.new_parameter(f"px{p}{q}#{layer}"))
                _spin_adapted_rotation(circ, m, p, q, circ.new_parameter(f"or{p}{q}#{layer}"))
        return circ

    def initial_parameters(self, circuit, rng=None):
        rng = rng or np.random.default_rng(0)
        return self.init_scale * rng.standard_normal(circuit.n_params)


@dataclass(repr=False)
class LUCJ(Ansatz):
    """Local unitary cluster Jastrow ansatz (Motta, Sung, Whaley, Head-Gordon & Shee, Chem.
    Sci. 14, 11213 (2023)):

        |psi> = prod_mu  exp(K_mu) exp(i J_mu) exp(-K_mu) |HF>

    * exp(K) is a general orbital rotation (the same for both spins), built as a brick wall of
      n_orbitals layers of nearest-neighbour complex Givens rotations (a real and an imaginary
      generator each) followed by orbital phases: n^2 real parameters, the dimension of U(n).
      The same parameters, with opposite sign and in reverse order, give exp(-K).
    * J is a density-density (Jastrow) operator restricted by ``interactions``:
      "square" -- same-spin neighbours n_{p,s} n_{p+1,s} and opposite spins on the same orbital
      n_{p,a} n_{p,b}; "heavy-hex" -- as square, opposite-spin terms only on every fourth
      orbital; "all" -- every pair (the non-local UCJ ansatz).

    Conserves N and S_z exactly; like the spin-balanced ansatz of the paper, the same-spin
    density terms do not commute with S^2, so <S^2> is only approximately conserved.
    The paper initializes from CCSD amplitudes; here the parameters start small and random
    (``init_scale``), since all-zero is a saddle point. The state is complex; energies are real.
    """
    layers: int = 1
    interactions: str = "square"
    init_scale: float = 0.05
    name: str = "lucj"
    orbital_dependent = True

    def _jastrow_terms(self, n):
        same = [(p, p + 1) for p in range(n - 1)]
        if self.interactions == "square":
            opp = [(p, p) for p in range(n)]
        elif self.interactions == "heavy-hex":
            opp = [(p, p) for p in range(0, n, 4)]
        elif self.interactions == "all":
            same = [(p, q) for p in range(n) for q in range(p + 1, n)]
            opp = [(p, q) for p in range(n) for q in range(p, n)]
        else:
            raise ValueError(f"interactions must be square, heavy-hex or all, got {self.interactions!r}")
        return same, opp

    def build(self, problem) -> Circuit:
        n, m = problem.n_orbitals, problem.mapping
        circ = Circuit(problem.n_qubits, problem.hf_index)
        same, opp = self._jastrow_terms(n)
        givens = [pq for _ in range(n) for pq in _brick_wall(n)] if n > 1 else []
        for mu in range(self.layers):
            g_params = [(circ.new_parameter(f"K{p}{q}re#{mu}"), circ.new_parameter(f"K{p}{q}im#{mu}"))
                        for p, q in givens]
            phases = [circ.new_parameter(f"phi{p}#{mu}") for p in range(n)]
            # exp(-K) = (D G)^dag = G^dag D^dag: phases first, then the Givens network reversed
            for p, ph in enumerate(phases):
                for s in (0, 1):
                    circ.add_generator(number_number_generator(m, 2 * p + s, 2 * p + s), ph, -1.0)
            for (p, q), (re, im) in reversed(list(zip(givens, g_params))):
                _spin_adapted_rotation(circ, m, p, q, im, -1.0, imaginary=True)
                _spin_adapted_rotation(circ, m, p, q, re, -1.0)
            # exp(i J)
            for p, q in same:
                par = circ.new_parameter(f"Jaa{p}{q}#{mu}")
                for s in (0, 1):
                    circ.add_generator(number_number_generator(m, 2 * p + s, 2 * q + s), par)
            for p, q in opp:  # n_{p,a} n_{q,b} + n_{q,a} n_{p,b}: one parameter keeps S^2
                par = circ.new_parameter(f"Jab{p}{q}#{mu}")
                circ.add_generator(number_number_generator(m, 2 * p, 2 * q + 1), par)
                if p != q:
                    circ.add_generator(number_number_generator(m, 2 * q, 2 * p + 1), par)
            # exp(K) = D G
            for (p, q), (re, im) in zip(givens, g_params):
                _spin_adapted_rotation(circ, m, p, q, re)
                _spin_adapted_rotation(circ, m, p, q, im, imaginary=True)
            for p, ph in enumerate(phases):
                for s in (0, 1):
                    circ.add_generator(number_number_generator(m, 2 * p + s, 2 * p + s), ph)
        return circ

    def initial_parameters(self, circuit, rng=None):
        rng = rng or np.random.default_rng(0)
        return self.init_scale * rng.standard_normal(circuit.n_params)


@dataclass(repr=False)
class SPA(Ansatz):
    """Separable pair approximation (Kottmann & Aspuru-Guzik, Phys. Rev. A 105, 032449 (2022)).

    Every electron pair lives in its own set of spatial orbitals: its occupied orbital i plus the
    virtual orbitals assigned to it. Within that set the pair is rotated by pair excitations
    exp(theta (P^dag_a P_i - h.c.)) applied one after another from i, which reach any real pair
    state in the set. Pairs in disjoint sets commute, so the state is a product of pair functions
    (separable): very shallow circuits, but only accurate in suitable orbitals, so it is meant to
    be used with orbital optimization (``VQE.orbital_opt``; ``orbital_dependent``).

    ``pairs``: optional list, one entry per occupied active orbital, of the virtual active orbitals
    (indices into the active space) assigned to it. By default each virtual goes to the occupied
    orbital with the largest exchange integral (ia|ia), keeping the sets as equal in size as
    possible. Closed shells only.
    """
    pairs: list | None = None
    name: str = "spa"
    orbital_dependent = True

    def assignment(self, problem) -> list[list[int]]:
        if problem.n_alpha != problem.n_beta:
            raise ValueError("SPA needs a closed-shell reference (N_alpha == N_beta)")
        n_occ, n = problem.n_alpha, problem.n_orbitals
        virt = list(range(n_occ, n))
        if self.pairs is not None:
            if len(self.pairs) != n_occ or sorted(a for s in self.pairs for a in s) != sorted(set(a for s in self.pairs for a in s)) \
                    or any(a not in virt for s in self.pairs for a in s):
                raise ValueError(f"pairs must give {n_occ} disjoint lists of virtual orbitals from {virt}")
            return [list(s) for s in self.pairs]
        h2 = problem.h2
        cap = -(-len(virt) // n_occ)  # ceil
        sets = [[] for _ in range(n_occ)]
        cand = sorted(((abs(h2[i, a, i, a]), i, a) for i in range(n_occ) for a in virt), reverse=True)
        placed = set()
        for _, i, a in cand:
            if a not in placed and len(sets[i]) < cap:
                sets[i].append(a)
                placed.add(a)
        return [sorted(s) for s in sets]

    def build(self, problem) -> Circuit:
        m = problem.mapping
        circ = Circuit(problem.n_qubits, problem.hf_index)
        for i, virts in enumerate(self.assignment(problem)):
            for a in virts:
                exc = ((2 * i, 2 * i + 1), (2 * a, 2 * a + 1))
                circ.add_generator(excitation_generator(m, exc), circ.new_parameter(f"spa{i}->{a}"))
        return circ


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
    "fabric": GateFabric, "gatefabric": GateFabric, "qnp": GateFabric,
    "lucj": LUCJ, "spa": SPA,
}


def get_ansatz(name: str, **kw) -> Ansatz:
    try:
        return ANSATZE[name.lower().replace("-", "")](**kw)
    except KeyError:
        raise ValueError(f"unknown ansatz {name!r}; choose from {sorted(ANSATZE)}") from None
