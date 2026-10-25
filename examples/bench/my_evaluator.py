"""Starter implementation for the SCC performance challenge. Copy this file and make it fast.

    python -m bench run lih-12q --impl examples/bench/my_evaluator.py:MyEvaluator
    python -m bench vqe h2o-16q --impl examples/bench/my_evaluator.py:MyEvaluator

It computes exactly what the reference does (full 2^n statevector, adjoint gradient), but with
the three hot kernels pulled out into methods you can replace one at a time: ``rotate``
(an excitation applied to a state), ``apply_h`` (H psi) and ``overlap`` (the inner products).
Every change must keep passing the correctness gate; see the challenge page for the rules.
"""
from __future__ import annotations

import numpy as np

from bench.interface import ReferenceEvaluator


class MyEvaluator:
    def __init__(self, case, comm=None):
        # Setup is timed separately from the evaluations: build tables, allocate buffers or copy
        # data to a GPU here. The case is defined in bench/cases.py (integrals h1, h2, e_core,
        # n_alpha, n_beta, hf_occupation, excitations).
        self.case, self.comm = case, comm
        ref = ReferenceEvaluator(case)  # TODO: replace with your own data structures
        self.circuit = ref.circuit
        self.hamiltonian = ref.hamiltonian
        self.idx = self.circuit.idx
        # each excitation k is one block rotation exp(theta_k G_k) acting on pairs (c, c ^ x)
        self.ops = self.circuit.ops

    # ------------------------------------------------------------ the three hot kernels
    def generator(self, op, psi):
        """G psi: gather psi[c ^ x] and multiply by a parity sign (TODO: fuse, compile, offload)."""
        return op.apply_generator(psi, self.idx)

    def rotate(self, op, psi, angle, g_psi=None):
        """exp(angle * G) psi, an exact 2x2 rotation on each amplitude pair."""
        g_psi = self.generator(op, psi) if g_psi is None else g_psi
        return op.rotate(psi, angle, g_psi, self.idx)

    def apply_h(self, psi):
        """H psi (TODO: sector-restricted, matrix-free or GPU sparse)."""
        return self.hamiltonian @ psi

    def overlap(self, a, b):
        """<a|b> (TODO: fuse into the kernels that produce a and b; MPI allreduce if distributed)."""
        return np.vdot(a, b)

    # ------------------------------------------------------------ energy + adjoint gradient
    def energy_and_gradient(self, theta):
        theta = np.asarray(theta, float)
        phi = self.circuit.initial_vector()  # the Hartree-Fock determinant
        for op in self.ops:
            a = op.angle(theta)
            if a != 0.0:
                phi = self.rotate(op, phi, a)
        lam = self.apply_h(phi)
        energy = float(np.real(self.overlap(phi, lam)))
        grad = np.zeros(theta.size)
        for op in reversed(self.ops):  # walk the circuit backwards, undoing one rotation at a time
            a = op.angle(theta)
            g_phi = self.generator(op, phi)
            grad[op.param] += 2.0 * op.coeff * np.real(self.overlap(lam, g_phi))
            if a != 0.0:
                phi = self.rotate(op, phi, -a, g_phi)
                lam = self.rotate(op, lam, -a)
        return energy, grad
