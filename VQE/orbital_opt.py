"""Orbital-optimized VQE (oo-VQE): optimize the molecular orbitals together with the circuit.

The energy is minimized over both the circuit parameters and a rotation of the orbitals (core,
active and virtual). This is CASSCF with VQE as the active-space solver (Mizukami et al., Phys.
Rev. Research 2, 033421 (2020); Sokolov et al., J. Chem. Phys. 152, 124107 (2020)): PySCF's
two-step CASSCF alternates

  1. VQE in the current orbitals (warm-started from the previous circuit parameters), then
  2. an orbital rotation driven by the VQE state's 1- and 2-particle density matrices,

until both are stationary. The density matrices come from the exact statevector through
``qcore.fci_interface``. With an exact solver this is CASSCF, so ``e_casscf`` (run alongside) is
the reference: oo-VQE can reach it, never go below it.

Orbital optimization matters most for ansatze that depend on the orbital basis, such as pair
ansatze (pUCCD, SPA, k-UpCCGSD): for them rotations among the active orbitals are also optimized
(``internal_rotation``). For UCCSD-like ansatze active-active rotations are nearly redundant.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from qcore import fci_interface as fi
from qcore.problem import MolecularProblem

from .ansatz import Ansatz, get_ansatz
from .vqe import VQE, VQEResult


class VQESolver:
    """PySCF ``fcisolver`` interface around VQE (the subset the two-step CASSCF uses)."""

    def __init__(self, ansatz: Ansatz, mapping, optimizer="BFGS", maxiter=2000, seed=0, verbose=False):
        self.ansatz, self.mapping = ansatz, mapping
        self.optimizer, self.maxiter, self.seed, self.verbose = optimizer, maxiter, seed, verbose
        self.circuit = None
        self.theta = None
        self.last: VQEResult | None = None
        self.history: list[float] = []
        self.converged = True
        self.nroots = 1
        self.conv_tol = 1e-10
        self.max_cycle = maxiter
        self._table = None

    def kernel(self, h1, h2, norb, nelec, ci0=None, ecore=0.0, **kwargs):
        na, nb = (nelec, 0) if np.isscalar(nelec) else nelec
        if np.isscalar(nelec):
            na, nb = (nelec + 1) // 2, nelec // 2
        prob = MolecularProblem.from_integrals(h1, h2, ecore, na, nb, self.mapping, e_fci=np.nan)
        if self.circuit is None:  # the circuit depends only on the orbital count and electrons
            self.circuit = self.ansatz.build(prob)
            self._table = fi.sector_table(prob.mapping, norb, na, nb)
        vqe = VQE(prob, self.ansatz, optimizer=self.optimizer, maxiter=self.maxiter, seed=self.seed)
        vqe.circuit = self.circuit
        res = vqe.run(initial_parameters=self.theta)
        self.theta, self.last = res.parameters, res
        self.converged = res.converged
        self.history.append(res.energy)
        if self.verbose:
            print(f"    VQE in current orbitals: E = {res.energy:.10f}  ({res.n_iterations} iterations)")
        ci = fi.to_civec(res.state, self._table)
        if np.max(np.abs(ci.imag)) < 1e-12:
            ci = np.ascontiguousarray(ci.real)
        return res.energy, ci

    def make_rdm1(self, ci, norb, nelec, **kw):
        return fi.make_rdm1(ci, norb, nelec)

    def make_rdm1s(self, ci, norb, nelec, **kw):
        return fi.make_rdm1s(ci, norb, nelec)

    def make_rdm12(self, ci, norb, nelec, **kw):
        return fi.make_rdm12(ci, norb, nelec)

    def spin_square(self, ci, norb, nelec):
        s2 = self.last.spin_squared if self.last else 0.0
        s = 0.5 * (np.sqrt(4 * s2 + 1) - 1)
        return s2, 2 * s + 1

    def dump_flags(self, verbose=None):
        pass


@dataclass
class OOVQEResult:
    energy: float            # oo-VQE energy at the optimized orbitals
    e_casscf: float          # exact solver in the same active space, optimized orbitals (reference)
    e_initial: float         # VQE energy in the starting (Hartree-Fock) orbitals
    e_fci: float             # exact energy in the starting orbitals' active space
    e_hf: float
    vqe: VQEResult           # last inner VQE run
    macro_iterations: int
    converged: bool
    mo_coeff: np.ndarray
    wall_time: float
    vqe_energies: list = field(default_factory=list)

    @property
    def error(self) -> float:
        """Error against CASSCF, the exact answer for orbital-optimized methods."""
        return self.energy - self.e_casscf

    def __str__(self) -> str:
        return (f"E_ooVQE = {self.energy:.10f} Ha | E_CASSCF = {self.e_casscf:.10f} Ha | "
                f"error vs CASSCF = {self.error:.3e} Ha | E_VQE(HF orbitals) = {self.e_initial:.10f} | "
                f"E_FCI(HF orbitals) = {self.e_fci:.10f} | params = {self.vqe.circuit.get('n_params')} | "
                f"macro iterations = {self.macro_iterations} | {self.wall_time:.1f}s")


class OrbitalOptimizedVQE:
    def __init__(self, problem: MolecularProblem, ansatz: Ansatz | str = "uccsd",
                 internal_rotation: bool | None = None, initial_orbitals: str | None = None,
                 optimizer: str = "BFGS", conv_tol: float = 1e-8, max_macro: int = 60, max_micro: int = 30,
                 seed: int = 0, verbose: bool = False):
        """``internal_rotation``: also rotate active orbitals among themselves; default True for
        ansatze that set ``orbital_dependent`` (pair ansatze, SPA, gate fabric, LUCJ).
        ``initial_orbitals``: "hf" (canonical) or "boys" / "pm" (Boys / Pipek-Mezey localization of
        the active occupied and the active virtual orbitals separately, which leaves the HF
        determinant unchanged); default "boys" for orbital-dependent ansatze, else "hf"."""
        if problem.mf is None:
            raise ValueError("orbital optimization needs a problem built from a molecule (not from_integrals)")
        self.problem = problem
        self.ansatz = get_ansatz(ansatz) if isinstance(ansatz, str) else ansatz
        dependent = getattr(self.ansatz, "orbital_dependent", False)
        self.internal_rotation = dependent if internal_rotation is None else internal_rotation
        self.initial_orbitals = initial_orbitals or ("boys" if dependent else "hf")
        if self.initial_orbitals not in ("hf", "boys", "pm"):
            raise ValueError(f"initial_orbitals must be hf, boys or pm, got {self.initial_orbitals!r}")
        self.optimizer, self.conv_tol = optimizer, conv_tol
        self.max_macro, self.max_micro = max_macro, max_micro
        self.seed, self.verbose = seed, verbose

    def _casscf(self):
        from pyscf import lo, mcscf
        p = self.problem
        mc = mcscf.CASSCF(p.mf, p.n_orbitals, (p.n_alpha, p.n_beta))
        mc.verbose = 0
        mc.conv_tol = self.conv_tol
        mc.max_cycle_macro = self.max_macro
        mc.internal_rotation = self.internal_rotation
        mo = np.array(getattr(p, "mo_coeff", p.mf.mo_coeff), copy=True)  # incl. MP2 / AVAS selections
        if self.initial_orbitals != "hf":
            nc = mc.ncore
            n_docc = min(p.n_alpha, p.n_beta)  # doubly occupied active orbitals; open shells stay put
            blocks = [range(nc, nc + n_docc), range(nc + max(p.n_alpha, p.n_beta), nc + p.n_orbitals)]
            for blk in blocks:
                blk = list(blk)
                if len(blk) > 1:
                    loc = (lo.Boys if self.initial_orbitals == "boys" else lo.PM)(p.mol, mo[:, blk])
                    loc.verbose = 0
                    mo[:, blk] = loc.kernel()
        return mc, mo

    @staticmethod
    def _rdm_energy(mc, mo, dm1, dm2) -> float:
        from pyscf import ao2mo
        h1, ecore = mc.get_h1eff(mo)
        h2 = ao2mo.restore(1, mc.get_h2eff(mo), mc.ncas)
        return float(ecore + np.einsum("pq,pq", h1, dm1) + 0.5 * np.einsum("pqrs,pqrs", h2, dm2))

    def _orbital_step(self, mc, mo, dm1, dm2):
        """Lower the energy at fixed density matrices by rotating the orbitals: preconditioned
        gradient steps from PySCF's analytic orbital gradient, each accepted only if the energy
        decreases (backtracking line search). Returns new orbitals and the final gradient norm."""
        from pyscf.mcscf import mc1step
        gnorm = np.inf
        for _ in range(self.max_micro):
            g, _, _, h_diag = mc1step.gen_g_hop(mc, mo, 1, dm1, dm2, mc.ao2mo(mo))
            gnorm = float(np.linalg.norm(g))
            if gnorm < 1e-6:
                break
            dx = -g / np.maximum(np.abs(h_diag), 0.05)
            step = np.linalg.norm(dx)
            if step > 0.5:
                dx *= 0.5 / step
            e0 = self._rdm_energy(mc, mo, dm1, dm2)
            slope = float(np.dot(g, dx))
            alpha = 1.0
            while alpha > 1e-6:
                trial = mo @ mc.update_rotate_matrix(alpha * dx)
                if self._rdm_energy(mc, trial, dm1, dm2) < e0 + 1e-4 * alpha * slope:
                    mo = trial
                    break
                alpha *= 0.5
            else:
                break  # no descent possible along this direction
        return mo, gnorm

    def run(self) -> OOVQEResult:
        from pyscf import ao2mo
        t0 = time.perf_counter()
        p = self.problem
        ref, mo0 = self._casscf()
        e_casscf = float(ref.kernel(mo0.copy())[0])
        mc, mo = self._casscf()
        solver = VQESolver(self.ansatz, p.mapping, optimizer=self.optimizer, seed=self.seed)
        nelec = (p.n_alpha, p.n_beta)
        e_prev, converged, energies = np.inf, False, []
        for it in range(self.max_macro):
            h1, ecore = mc.get_h1eff(mo)
            e, ci = solver.kernel(h1, mc.get_h2eff(mo), mc.ncas, nelec, ecore=ecore)
            energies.append(e)
            dm1, dm2 = fi.make_rdm12(ci, mc.ncas, nelec)
            mo, gnorm = self._orbital_step(mc, mo, dm1, dm2)
            if self.verbose:
                print(f"  macro {it + 1:3d}: E = {e:.10f}  |g_orb| = {gnorm:.2e}")
            if abs(e_prev - e) < self.conv_tol and gnorm < 1e-4:
                converged = True
                break
            e_prev = e
        # final VQE in the final orbitals
        h1, ecore = mc.get_h1eff(mo)
        e, _ = solver.kernel(h1, mc.get_h2eff(mo), mc.ncas, nelec, ecore=ecore)
        energies.append(e)
        return OOVQEResult(energy=e, e_casscf=e_casscf, e_initial=energies[0], e_fci=p.e_fci,
                           e_hf=p.e_hf, vqe=solver.last, macro_iterations=len(energies) - 1,
                           converged=converged, mo_coeff=mo, wall_time=time.perf_counter() - t0,
                           vqe_energies=energies)
