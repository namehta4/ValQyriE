"""ADAPT-VQE (Grimsley et al., Nat. Commun. 10, 3007 (2019)).

At each step the pool operator A with the largest energy gradient
dE/dtheta|_0 = <psi|[H, A]|psi> = 2 Re <H psi | A psi> is appended, then all
parameters are re-optimised (warm-started).  Pools:

  fermionic   spin-orbital singles + doubles (original ADAPT-VQE)
  qeb         qubit-excitation-based singles + doubles (Yordanov et al.), JW only
  qubit       individual Pauli strings (qubit-ADAPT, Tang et al. PRX Quantum 2, 020310)
  generalized generalized fermionic singles + doubles
  ceo         coupled exchange operators (CEO-ADAPT-VQE, Ramoa et al., npj Quantum Inf. 2025,
              arXiv:2407.08696), JW only

CEO pool: the qubit excitations (QEs) acting on the same four qubits (two for opposite-spin
sets, three for same-spin sets) are combined into one-parameter sums and differences
theta (T_a +- T_b) (OVP-CEOs, 9-CNOT circuits). The OVP-CEO with the largest gradient is
selected; if more than one of the QEs on its qubits has a nonzero gradient, those QEs are
added instead with one parameter each (an MVP-CEO, 13-CNOT circuit), otherwise the OVP-CEO
itself is added. Single QEs are their own CEOs.

``tetris=True`` (TETRIS-ADAPT-VQE, Anastasiou et al., arXiv:2209.10562): each iteration adds, in order of decreasing gradient, every operator with a
nonzero gradient whose qubits are disjoint from those already chosen in that iteration, then
optimises. This gives shallower circuits (operators on disjoint qubits run in parallel) and
fewer iterations. It suits the pools whose operators act on at most four qubits (qeb, qubit,
ceo); fermionic operators carry Jordan-Wigner strings and rarely have disjoint supports.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import reduce
from itertools import combinations

import numpy as np
from scipy.optimize import minimize

from qcore.circuit import Circuit, Generator
from qcore.excitations import doubles, excitation_generator, pauli_string_pool, qubit_generator, singles
from qcore.mappings import JordanWigner

from .vqe import VQEResult

POOLS = ("fermionic", "generalized", "qeb", "qubit", "ceo")
# CNOT count / CNOT depth of the published circuits: single QE 2/2 and double QE 13/11 (Yordanov
# et al., PRA 102, 062612 (2020)); OVP-CEO 9/7 and MVP-CEO 13 (Ramoa et al. 2025; its depth is not
# given, so the count is used as an upper bound)
CNOTS = {"single_qe": 2, "double_qe": 13, "ovp_ceo": 9, "mvp_ceo": 13}
CNOT_DEPTH = {"single_qe": 2, "double_qe": 11, "ovp_ceo": 7, "mvp_ceo": 13}


def build_pool(problem, pool: str = "fermionic") -> list[Generator]:
    """The selection generators of a pool (one parameter each)."""
    return [op.generator for op in pool_operators(problem, pool)]


@dataclass
class PoolOperator:
    generator: Generator     # one-parameter form; its gradient drives the selection
    support: int             # bit mask of the qubits it acts on (TETRIS)
    cnots: int               # CNOTs of its circuit
    depth: int               # CNOT depth of its circuit
    members: tuple = ()      # CEO only: the QEs on the same qubits (the MVP form)


def _support(g: Generator) -> int:
    return reduce(lambda a, b: a | b, (x | z for x, z in zip(g.x, g.z)), 0)


def _ladder_cnots(g: Generator) -> int:
    """CNOT-ladder implementation of each Pauli rotation: 2 (weight - 1)."""
    return sum(2 * (bin(x | z).count("1") - 1) for x, z in zip(g.x, g.z) if bin(x | z).count("1") > 1)


def _ladder(g: Generator) -> PoolOperator:
    c = _ladder_cnots(g)  # a ladder is sequential: depth = count
    return PoolOperator(g, _support(g), c, c)


def _qe(g: Generator) -> PoolOperator:
    kind = "single_qe" if bin(_support(g)).count("1") == 2 else "double_qe"
    return PoolOperator(g, _support(g), CNOTS[kind], CNOT_DEPTH[kind])


def _ceo_pool(problem) -> list[PoolOperator]:
    if not isinstance(problem.mapping, JordanWigner):
        raise ValueError("the CEO pool is built from qubit excitations and requires the Jordan-Wigner mapping")
    n, na, nb, nq = problem.n_orbitals, problem.n_alpha, problem.n_beta, problem.n_qubits
    ops = []
    for e in singles(n, na, nb, True):
        ops.append(_qe(qubit_generator(nq, e)))
    by_support: dict[int, list[Generator]] = {}
    for e in doubles(n, na, nb, True):
        g = qubit_generator(nq, e)
        by_support.setdefault(_support(g), []).append(g)
    for support, qes in by_support.items():
        if len(qes) == 1:
            ops.append(_qe(qes[0]))
            continue
        for a, b in combinations(qes, 2):
            for sign, tag in ((1, "+"), (-1, "-")):
                op = a.to_pauli_sum(nq) + sign * b.to_pauli_sum(nq)
                g = Generator.from_pauli_sum(op, label=f"ceo{tag}[{a.label}, {b.label}]")
                ops.append(PoolOperator(g, support, CNOTS["ovp_ceo"], CNOT_DEPTH["ovp_ceo"], tuple(qes)))
    return ops


def pool_operators(problem, pool: str = "fermionic") -> list[PoolOperator]:
    n, na, nb = problem.n_orbitals, problem.n_alpha, problem.n_beta
    m = problem.mapping
    if pool == "ceo":
        return _ceo_pool(problem)
    if pool == "fermionic":
        return [_ladder(excitation_generator(m, e)) for e in singles(n, na, nb) + doubles(n, na, nb)]
    if pool == "generalized":
        return [_ladder(excitation_generator(m, e)) for e in singles(n, na, nb, True) + doubles(n, na, nb, True)]
    if pool == "qeb":
        return [_qe(excitation_generator(m, e, kind="qubit")) for e in singles(n, na, nb, True) + doubles(n, na, nb, True)]
    if pool == "qubit":
        return [_ladder(g) for g in pauli_string_pool(m, singles(n, na, nb, True) + doubles(n, na, nb, True))]
    raise ValueError(f"unknown pool {pool!r}; choose from {', '.join(POOLS)}")


@dataclass
class AdaptResult(VQEResult):
    selected: list = field(default_factory=list)
    gradient_norms: list = field(default_factory=list)
    step_energies: list = field(default_factory=list)
    operators_per_iteration: list = field(default_factory=list)
    cnot_count: int = 0      # sum over added operators of their published-circuit CNOT counts
    cnot_depth: int = 0      # CNOT depth, operators scheduled as early as their qubits allow
    resources: list = field(default_factory=list)  # per iteration: params, CNOTs, CNOT depth

    def at_accuracy(self, tol: float = 1.6e-3) -> dict | None:
        """Resources of the shortest circuit along the run whose error is below ``tol``."""
        return next(({"iteration": i + 1, **r} for i, (e, r) in enumerate(zip(self.step_energies, self.resources))
                     if e - self.e_fci < tol), None)

    def __str__(self) -> str:
        return (f"{super().__str__()} | CNOTs = {self.cnot_count} | CNOT depth = {self.cnot_depth}")


class AdaptVQE:
    def __init__(self, problem, pool: str = "fermionic", grad_tol: float = 1e-3, max_steps: int = 150,
                 energy_tol: float | None = None, optimizer: str = "BFGS", tetris: bool = False,
                 zero_tol: float = 1e-8, verbose: bool = False):
        """``zero_tol``: gradients below this count as zero (TETRIS candidates and the
        OVP/MVP choice of the CEO pool)."""
        self.problem = problem
        self.pool_name = pool
        self.ops = pool_operators(problem, pool)
        self.pool = [op.generator for op in self.ops]
        self.pool_ops = [g.to_pauli_sum(problem.n_qubits).to_sparse() for g in self.pool]
        self._member_ops = {}
        self.grad_tol = grad_tol
        self.max_steps = max_steps
        self.energy_tol = energy_tol
        self.optimizer = optimizer
        self.tetris = tetris
        self.zero_tol = zero_tol
        self.verbose = verbose

    def _gradient(self, gen: Generator, hpsi, psi) -> float:
        a = self._member_ops.get(id(gen))
        if a is None:
            a = self._member_ops[id(gen)] = gen.to_pauli_sum(self.problem.n_qubits).to_sparse()
        return 2 * float(np.real(np.vdot(hpsi, a @ psi)))

    def _choose(self, grads) -> list[int]:
        order = np.argsort(-np.abs(grads), kind="stable")
        if not self.tetris:
            return [int(order[0])]
        chosen, used = [], 0
        for k in order:
            if abs(grads[k]) < self.zero_tol:
                break
            if self.ops[k].support & used:
                continue
            chosen.append(int(k))
            used |= self.ops[k].support
        return chosen

    def _add(self, circ, k, hpsi, psi) -> tuple[list[str], int]:
        """Append pool operator k; returns the labels of the new parameters, its CNOTs and depth."""
        op = self.ops[k]
        if op.members:  # CEO: MVP form if several QEs on these qubits have nonzero gradients
            nonzero = [q for q in op.members if abs(self._gradient(q, hpsi, psi)) >= self.zero_tol]
            if len(nonzero) > 1:
                for q in nonzero:
                    circ.add_generator(q, circ.new_parameter(q.label))
                return ([f"mvp[{', '.join(q.label for q in nonzero)}]"] * len(nonzero), CNOTS["mvp_ceo"],
                        CNOT_DEPTH["mvp_ceo"])
        circ.add_generator(op.generator, circ.new_parameter(op.generator.label))
        return [op.generator.label], op.cnots, op.depth

    def run(self) -> AdaptResult:
        t0 = time.perf_counter()
        p = self.problem
        h = p.hamiltonian_sparse
        circ = Circuit(p.n_qubits, p.hf_index)
        theta = np.zeros(0)
        selected, gnorms, energies, per_iter, resources = [], [], [], [], []
        nfev, cnots = 0, 0
        busy = np.zeros(p.n_qubits, dtype=int)  # CNOT layer at which each qubit becomes free
        e_prev = p.energy(circ.state(theta))
        for step in range(self.max_steps):
            psi = circ.state(theta)
            hpsi = h @ psi
            grads = np.array([2 * np.real(np.vdot(hpsi, a @ psi)) for a in self.pool_ops])
            gnorm = float(np.linalg.norm(grads))
            gnorms.append(gnorm)
            if gnorm < self.grad_tol:
                if self.verbose:
                    print(f"ADAPT converged: |g| = {gnorm:.2e}")
                break
            added = []
            for k in self._choose(grads):
                labels, c, d = self._add(circ, k, hpsi, psi)
                added.append(labels[0])
                cnots += c
                qubits = [q for q in range(p.n_qubits) if self.ops[k].support >> q & 1]
                busy[qubits] = busy[qubits].max() + d
                theta = np.append(theta, np.zeros(len(labels)))
            selected += added
            per_iter.append(len(added))
            res = minimize(lambda x: circ.energy_and_gradient(x, h), theta, jac=True,
                           method=self.optimizer, tol=1e-12, options={"maxiter": 2000})
            theta, nfev = res.x, nfev + res.nfev
            e = res.fun
            energies.append(e)
            resources.append({"params": circ.n_params, "cnots": cnots, "cnot_depth": int(busy.max())})
            if self.verbose:
                what = added[0] if len(added) == 1 else f"{len(added)} operators"
                print(f"step {step + 1:3d}: +{what:<28s} |g|={gnorm:.3e}  E={e:.10f}  "
                      f"err={e - p.e_fci:.2e}  params={circ.n_params}  CNOTs={cnots}  depth={busy.max()}")
            if self.energy_tol is not None and abs(e_prev - e) < self.energy_tol:
                break
            e_prev = e
        psi = circ.state(theta)
        return AdaptResult(
            energy=p.energy(psi), parameters=theta, e_hf=p.e_hf, e_fci=p.e_fci,
            n_iterations=len(per_iter), n_evaluations=nfev, history=energies,
            wall_time=time.perf_counter() - t0, circuit=circ.depth_summary(),
            spin_squared=p.spin_squared(psi),
            particle_number=float(np.real(np.vdot(psi, p.number_diagonal * psi))), state=psi,
            selected=selected, gradient_norms=gnorms, step_energies=energies,
            operators_per_iteration=per_iter, cnot_count=cnots, cnot_depth=int(busy.max()), resources=resources,
            converged=gnorms[-1] < self.grad_tol if gnorms else True)
