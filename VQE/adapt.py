"""ADAPT-VQE (Grimsley et al., Nat. Commun. 10, 3007 (2019)).

At each step the pool operator A with the largest energy gradient
dE/dtheta|_0 = <psi|[H, A]|psi> = 2 Re <H psi | A psi> is appended, then all
parameters are re-optimised (warm-started).  Pools:

  fermionic   spin-orbital singles + doubles (original ADAPT-VQE)
  qeb         qubit-excitation-based singles + doubles (Yordanov et al.), JW only
  qubit       individual Pauli strings (qubit-ADAPT, Tang et al. PRX Quantum 2, 020310)
  generalized generalized fermionic singles + doubles
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from qcore.circuit import Circuit, Generator
from qcore.excitations import doubles, excitation_generator, pauli_string_pool, singles

from .vqe import VQEResult


def build_pool(problem, pool: str = "fermionic") -> list[Generator]:
    n, na, nb = problem.n_orbitals, problem.n_alpha, problem.n_beta
    m = problem.mapping
    if pool == "fermionic":
        excs = singles(n, na, nb) + doubles(n, na, nb)
        return [excitation_generator(m, e) for e in excs]
    if pool == "generalized":
        excs = singles(n, na, nb, True) + doubles(n, na, nb, True)
        return [excitation_generator(m, e) for e in excs]
    if pool == "qeb":
        excs = singles(n, na, nb, True) + doubles(n, na, nb, True)
        return [excitation_generator(m, e, kind="qubit") for e in excs]
    if pool == "qubit":
        excs = singles(n, na, nb, True) + doubles(n, na, nb, True)
        return pauli_string_pool(m, excs)
    raise ValueError(f"unknown pool {pool!r}")


@dataclass
class AdaptResult(VQEResult):
    selected: list = field(default_factory=list)
    gradient_norms: list = field(default_factory=list)
    step_energies: list = field(default_factory=list)


class AdaptVQE:
    def __init__(self, problem, pool: str = "fermionic", grad_tol: float = 1e-3, max_steps: int = 150,
                 energy_tol: float | None = None, optimizer: str = "BFGS", verbose: bool = False):
        self.problem = problem
        self.pool_name = pool
        self.pool = build_pool(problem, pool)
        self.pool_ops = [g.to_pauli_sum(problem.n_qubits).to_sparse() for g in self.pool]
        self.grad_tol = grad_tol
        self.max_steps = max_steps
        self.energy_tol = energy_tol
        self.optimizer = optimizer
        self.verbose = verbose

    def run(self) -> AdaptResult:
        t0 = time.perf_counter()
        p = self.problem
        h = p.hamiltonian_sparse
        circ = Circuit(p.n_qubits, p.hf_index)
        theta = np.zeros(0)
        selected, gnorms, energies = [], [], []
        nfev = 0
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
            k = int(np.argmax(np.abs(grads)))
            gen = self.pool[k]
            circ.add_generator(gen, circ.new_parameter(gen.label))
            selected.append(gen.label)
            theta = np.append(theta, 0.0)
            res = minimize(lambda x: circ.energy_and_gradient(x, h), theta, jac=True,
                           method=self.optimizer, tol=1e-12, options={"maxiter": 2000})
            theta, nfev = res.x, nfev + res.nfev
            e = res.fun
            energies.append(e)
            if self.verbose:
                print(f"step {step + 1:3d}: +{gen.label:<28s} |g|={gnorm:.3e}  E={e:.10f}  "
                      f"err={e - p.e_fci:.2e}")
            if self.energy_tol is not None and abs(e_prev - e) < self.energy_tol:
                break
            e_prev = e
        psi = circ.state(theta)
        return AdaptResult(
            energy=p.energy(psi), parameters=theta, e_hf=p.e_hf, e_fci=p.e_fci,
            n_iterations=len(selected), n_evaluations=nfev, history=energies,
            wall_time=time.perf_counter() - t0, circuit=circ.depth_summary(),
            spin_squared=p.spin_squared(psi),
            particle_number=float(np.real(np.vdot(psi, p.number_diagonal * psi))), state=psi,
            selected=selected, gradient_norms=gnorms, step_energies=energies,
            converged=gnorms[-1] < self.grad_tol if gnorms else True)
