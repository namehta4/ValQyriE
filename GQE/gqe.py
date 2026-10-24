"""Generative Quantum Eigensolver (GQE / GPT-QE).

Nakaji et al., "The generative quantum eigensolver (GQE) and its application for
ground state search", arXiv:2401.09253 (2024).

A GPT generates sequences of pool tokens j_1..j_L; the circuit
U = U_{j_L} ... U_{j_1} acts on the Hartree-Fock state and its energy E(j) is
computed exactly on the CPU statevector simulator.  The model is trained by
*logit matching*: the summed (w_sum, default) or mean logits W(j) of the generated tokens are
regressed onto the energy, so that sampling with p(j) ~ exp(-beta W(j))
concentrates on low-energy circuits as the inverse temperature beta is raised.

  loss = "exp"  :  < (exp(-W) - exp(-s (E - E_ref)))^2 >     (paper, Eq. for GPT-QE)
         "mse"  :  < (W - s (E - E_ref))^2 >

E_ref is the Hartree-Fock energy and s an energy scale.  No circuit parameters
are optimised during training; optionally, the best circuit's angles are refined
afterwards with a gradient-based VQE step (reported separately).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize

from qcore.problem import CHEMICAL_ACCURACY, HARTREE_TO_KCAL

from .model import GPT
from .pool import DEFAULT_TIMES, OperatorPool


@dataclass
class GQEConfig:
    seq_len: int | None = None       # default: 2 * number of pool generators, clipped to [4, 30]
    pool: str = "uccsd"              # uccsd | generalized | qeb | qubit
    times: tuple = DEFAULT_TIMES
    screen_tol: float | None = 1e-8  # drop symmetry-forbidden doubles (None = full pool)
    iterations: int = 300
    batch_size: int = 64
    beta_start: float = 5.0          # inverse temperature schedule (linear)
    beta_end: float = 200.0
    energy_scale: float = 20.0       # s in the loss
    loss: str = "exp"                # exp | mse
    logit_reduction: str = "sum"     # sum (w_sum, as in the paper) | mean
    lr: float = 5e-4
    grad_steps: int = 2              # optimiser steps per iteration
    replay_size: int = 16            # elite sequences from the replay buffer added to every batch
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    refine: bool = True              # VQE-refine the angles of the best sequence at the end
    seed: int = 0
    threads: int | None = None       # torch CPU threads (None = PyTorch default)
    verbose: bool = False


@dataclass
class GQEResult:
    energy: float                    # best sampled (unrefined) energy
    sequence: list
    e_hf: float
    e_fci: float
    refined_energy: float | None = None
    refined_parameters: np.ndarray | None = None
    history: list = field(default_factory=list)  # (iteration, beta, mean E, min E, best E, loss)
    n_circuits: int = 0
    wall_time: float = 0.0
    operators: list = field(default_factory=list)

    @property
    def error(self) -> float:
        return self.energy - self.e_fci

    @property
    def refined_error(self) -> float | None:
        return None if self.refined_energy is None else self.refined_energy - self.e_fci

    def __str__(self) -> str:
        s = (f"E_GQE = {self.energy:.10f} Ha | E_FCI = {self.e_fci:.10f} Ha | "
             f"error = {self.error:.3e} Ha ({self.error * HARTREE_TO_KCAL:.4f} kcal/mol) | "
             f"circuits sampled = {self.n_circuits} | {self.wall_time:.1f}s")
        if self.refined_energy is not None:
            s += (f"\nE_GQE+refine = {self.refined_energy:.10f} Ha | error = {self.refined_error:.3e} Ha"
                  f" | chemically accurate: {abs(self.refined_error) < CHEMICAL_ACCURACY}")
        return s


class GQE:
    def __init__(self, problem, config: GQEConfig | None = None, **overrides):
        self.problem = problem
        self.cfg = config or GQEConfig()
        for k, v in overrides.items():
            if not hasattr(self.cfg, k):
                raise TypeError(f"unknown GQE option {k!r}")
            setattr(self.cfg, k, v)
        cfg = self.cfg
        if cfg.threads:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed)
        self.rng = torch.Generator().manual_seed(cfg.seed)
        self.pool = OperatorPool.from_problem(problem, cfg.pool, cfg.times, cfg.screen_tol)
        self.seq_len = cfg.seq_len or int(np.clip(2 * len(self.pool.generators), 4, 30))
        self.model = GPT(self.pool.vocab_size, self.seq_len, cfg.n_layer, cfg.n_head, cfg.n_embd)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr)
        self.h = problem.hamiltonian_sparse
        self._cache: dict[tuple, float] = {}

    def energy(self, seq) -> float:
        key = tuple(int(j) for j in seq)
        if key not in self._cache:
            psi = self.pool.state(key)
            self._cache[key] = float(np.real(np.vdot(psi, self.h @ psi)))
        return self._cache[key]

    def _loss(self, seqs: torch.Tensor, energies: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        w = self.model.sequence_logits(seqs)
        w = w.mean(dim=1) if cfg.logit_reduction == "mean" else w.sum(dim=1)
        target = cfg.energy_scale * (energies - self.problem.e_hf)
        if cfg.loss == "exp":
            return torch.mean((torch.exp(-w) - torch.exp(-target)) ** 2)
        if cfg.loss == "mse":
            return torch.mean((w - target) ** 2)
        raise ValueError(f"unknown loss {cfg.loss!r}")

    def run(self) -> GQEResult:
        cfg, p = self.cfg, self.problem
        t0 = time.perf_counter()
        buffer: dict[tuple, float] = {}
        best_seq, best_e = None, np.inf
        history = []
        for it in range(cfg.iterations):
            frac = it / max(1, cfg.iterations - 1)
            beta = cfg.beta_start + frac * (cfg.beta_end - cfg.beta_start)
            self.model.eval()
            seqs = self.model.generate(cfg.batch_size, self.seq_len, beta, self.rng)
            energies = np.array([self.energy(s.tolist()) for s in seqs])
            for s, e in zip(seqs.tolist(), energies):
                buffer[tuple(s)] = e
            k = int(np.argmin(energies))
            if energies[k] < best_e:
                best_e, best_seq = float(energies[k]), seqs[k].tolist()

            train_s, train_e = seqs, energies
            if cfg.replay_size:
                elite = sorted(buffer.items(), key=lambda kv: kv[1])[:cfg.replay_size]
                train_s = torch.cat([seqs, torch.tensor([list(s) for s, _ in elite])])
                train_e = np.concatenate([energies, [e for _, e in elite]])
            e_t = torch.tensor(np.asarray(train_e, dtype=float).tolist(), dtype=torch.float32)
            self.model.train()
            for _ in range(cfg.grad_steps):
                self.optim.zero_grad()
                loss = self._loss(train_s, e_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
            history.append((it, beta, float(energies.mean()), float(energies.min()), best_e, loss.item()))
            if cfg.verbose and (it % 10 == 0 or it == cfg.iterations - 1):
                print(f"iter {it:4d} beta={beta:7.2f}  <E>={energies.mean():.6f}  min E={energies.min():.8f}"
                      f"  best={best_e:.8f}  err={best_e - p.e_fci:.2e}  loss={loss.item():.3e}")

        result = GQEResult(energy=best_e, sequence=best_seq, e_hf=p.e_hf, e_fci=p.e_fci, history=history,
                           n_circuits=len(self._cache), operators=self.pool.describe(best_seq))
        if cfg.refine and best_seq:
            circ, x0 = self.pool.parametrized_circuit(best_seq)
            res = minimize(lambda x: circ.energy_and_gradient(x, self.h), x0, jac=True, method="BFGS",
                           options={"gtol": 1e-8, "maxiter": 2000})
            result.refined_energy, result.refined_parameters = float(res.fun), res.x
        result.wall_time = time.perf_counter() - t0
        return result
