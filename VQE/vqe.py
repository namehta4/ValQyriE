"""Variational Quantum Eigensolver (Peruzzo et al., Nat. Commun. 5, 4213 (2014)).

Exact statevector energies and adjoint-method gradients on CPU.  Optimizers:
any ``scipy.optimize.minimize`` method (BFGS, L-BFGS-B, CG, SLSQP, COBYLA,
Nelder-Mead, Powell, ...) plus built-in Adam and SPSA.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from qcore.circuit import Circuit
from qcore.problem import CHEMICAL_ACCURACY, HARTREE_TO_KCAL

from .ansatz import Ansatz, get_ansatz

GRADIENT_FREE = {"cobyla", "nelder-mead", "powell", "cobyqa", "spsa"}


@dataclass
class VQEResult:
    energy: float
    parameters: np.ndarray
    e_hf: float
    e_fci: float
    n_iterations: int
    n_evaluations: int
    history: list = field(default_factory=list)
    converged: bool = True
    message: str = ""
    wall_time: float = 0.0
    circuit: dict = field(default_factory=dict)
    spin_squared: float = float("nan")
    particle_number: float = float("nan")
    state: np.ndarray | None = field(default=None, repr=False)

    @property
    def error(self) -> float:
        return self.energy - self.e_fci

    @property
    def correlation_recovered(self) -> float:
        denom = self.e_fci - self.e_hf
        return float("nan") if abs(denom) < 1e-12 else (self.energy - self.e_hf) / denom

    @property
    def chemically_accurate(self) -> bool:
        return abs(self.error) < CHEMICAL_ACCURACY

    def __str__(self) -> str:
        return (f"E_VQE = {self.energy:.10f} Ha | E_FCI = {self.e_fci:.10f} Ha | "
                f"error = {self.error:.3e} Ha ({self.error * HARTREE_TO_KCAL:.4f} kcal/mol) | "
                f"corr. recovered = {100 * self.correlation_recovered:.2f}% | "
                f"<S^2> = {self.spin_squared:.4f} | <N> = {self.particle_number:.4f} | "
                f"params = {self.circuit.get('n_params')} | iters = {self.n_iterations} | "
                f"evals = {self.n_evaluations} | {self.wall_time:.2f}s")


def adam(fun_grad, x0, maxiter=500, lr=0.02, beta1=0.9, beta2=0.999, eps=1e-8, gtol=1e-6, callback=None):
    x = np.array(x0, float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    nfev = 0
    for t in range(1, maxiter + 1):
        f, g = fun_grad(x)
        nfev += 1
        if np.linalg.norm(g) < gtol:
            break
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        x -= lr * (m / (1 - beta1 ** t)) / (np.sqrt(v / (1 - beta2 ** t)) + eps)
        if callback:
            callback(x)
    return x, t, nfev


def spsa(fun, x0, maxiter=500, a=0.1, c=0.1, alpha=0.602, gamma=0.101, rng=None, callback=None):
    """Simultaneous-perturbation stochastic approximation (Spall 1998)."""
    rng = rng or np.random.default_rng(0)
    x = np.array(x0, float)
    big_a = 0.1 * maxiter
    for k in range(maxiter):
        ak, ck = a / (k + 1 + big_a) ** alpha, c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        g = (fun(x + ck * delta) - fun(x - ck * delta)) / (2 * ck) * delta
        x -= ak * g
        if callback:
            callback(x)
    return x, maxiter, 2 * maxiter


class VQE:
    """Run VQE for a ``MolecularProblem`` with any ``Ansatz``.

    Parameters
    ----------
    problem : qcore.problem.MolecularProblem
    ansatz : Ansatz instance or name (see ``VQE.ansatz.ANSATZE``)
    optimizer : scipy method name, "adam" or "spsa"
    number_penalty, spin_penalty : weights mu of mu (N - N0)^2 and mu (S_z - S_z0)^2 added to the
        cost (useful for symmetry-breaking ansatze such as HEA); reported energies exclude them.
    """

    def __init__(self, problem, ansatz: Ansatz | str = "uccsd", optimizer: str = "BFGS",
                 maxiter: int = 1000, tol: float = 1e-10, number_penalty: float = 0.0,
                 spin_penalty: float = 0.0, seed: int = 0, verbose: bool = False,
                 optimizer_options: dict | None = None):
        self.problem = problem
        self.ansatz = get_ansatz(ansatz) if isinstance(ansatz, str) else ansatz
        self.optimizer = optimizer
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self.optimizer_options = optimizer_options or {}
        self.circuit: Circuit = self.ansatz.build(problem)
        self.hamiltonian = problem.hamiltonian_sparse
        self.penalty = None
        if number_penalty or spin_penalty:
            sz0 = 0.5 * (problem.n_alpha - problem.n_beta)
            self.penalty = (number_penalty * (problem.number_diagonal - problem.n_electrons) ** 2
                            + spin_penalty * (problem.sz_diagonal - sz0) ** 2)

    def cost(self, theta) -> float:
        return self.circuit.energy(theta, self.hamiltonian, self.penalty)

    def cost_and_grad(self, theta):
        return self.circuit.energy_and_gradient(theta, self.hamiltonian, self.penalty)

    def run(self, initial_parameters=None) -> VQEResult:
        t0 = time.perf_counter()
        circ = self.circuit
        x0 = (self.ansatz.initial_parameters(circ, self.rng) if initial_parameters is None
              else np.asarray(initial_parameters, float))
        history, nfev = [], [0]

        def fun(x):
            nfev[0] += 1
            return self.cost(x)

        def fun_grad(x):
            nfev[0] += 1
            return self.cost_and_grad(x)

        def cb(x, *_):
            e = self.cost(x)
            history.append(e)
            if self.verbose:
                print(f"  iter {len(history):4d}  E = {e:.10f}")

        method = self.optimizer.lower()
        converged, message = True, ""
        if circ.n_params == 0:
            x, nit = x0, 0
        elif method == "adam":
            x, nit, _ = adam(fun_grad, x0, maxiter=self.maxiter, callback=cb, **self.optimizer_options)
        elif method == "spsa":
            x, nit, _ = spsa(fun, x0, maxiter=self.maxiter, rng=self.rng, callback=cb,
                             **self.optimizer_options)
        else:
            gradient_free = method in GRADIENT_FREE
            opts = {"maxiter": self.maxiter, **self.optimizer_options}
            if method in {"bfgs", "cg"}:
                opts.setdefault("gtol", 1e-7)
            res = minimize(fun if gradient_free else fun_grad, x0, jac=None if gradient_free else True,
                           method=self.optimizer, tol=self.tol, options=opts, callback=cb)
            x, nit, converged, message = res.x, res.get("nit", len(history)), bool(res.success), str(res.message)

        psi = circ.state(x)
        p = self.problem
        return VQEResult(
            energy=p.energy(psi), parameters=x, e_hf=p.e_hf, e_fci=p.e_fci,
            n_iterations=int(nit), n_evaluations=nfev[0], history=history, converged=converged,
            message=message, wall_time=time.perf_counter() - t0, circuit=circ.depth_summary(),
            spin_squared=p.spin_squared(psi),
            particle_number=float(np.real(np.vdot(psi, p.number_diagonal * psi))), state=psi)
