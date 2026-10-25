"""Self-verifying stress workloads.

Every workload is built from a deterministic spec, so every worker on every core
(and every node) computes on identical inputs.  Each ``step()`` returns

    (work, checksum, error)

``work`` is in the workload's unit (FLOP, amplitude updates, evaluations); ``checksum``
is an exact bit-level fingerprint of the result (identical across steps, and across
cores running the same code path); ``error`` is None or a description of a failed
correctness check.  Checks are exact or physics-based, so a wrong answer from a
marginal CPU, a bad core, overheating or memory errors is detected rather than
silently averaged into a benchmark number.
"""
from __future__ import annotations

import numpy as np

UNITS = {"dgemm": "GFLOP", "statevector": "Gamp", "vqe": "evals", "transformer": "GFLOP", "triad": "GB"}


def _xor_fingerprint(*arrays) -> int:
    fp = np.uint64(0)
    for a in arrays:
        a = np.ascontiguousarray(a)
        fp ^= np.bitwise_xor.reduce(a.view(np.uint64).reshape(-1))
        fp ^= np.uint64(a.view(np.uint64).sum(dtype=np.uint64))  # wraps; still exact
    return int(fp)


class DGEMM:
    """C = A @ B with small-integer matrices: the float64 result is exact regardless of
    summation order or SIMD path, so it is verified exactly (Freivalds' check in integer
    arithmetic, independent of BLAS) on every step."""

    name = "dgemm"

    def __init__(self, n: int = 1024, seed: int = 1234):
        rng = np.random.default_rng(seed)
        self.n = n
        self.ai = rng.integers(-8, 9, (n, n), dtype=np.int64)
        self.bi = rng.integers(-8, 9, (n, n), dtype=np.int64)
        self.a, self.b = self.ai.astype(np.float64), self.bi.astype(np.float64)
        self.r = rng.integers(-(1 << 20), 1 << 20, (n, 2), dtype=np.int64)
        self.abr = self.ai @ (self.bi @ self.r)  # integer reference, no BLAS involved

    def check(self, c) -> str | None:
        if not np.all(np.abs(c) <= 64 * self.n) or not np.array_equal(c, np.rint(c)):
            return "non-integer or out-of-range entries in A@B"
        if not np.array_equal(c.astype(np.int64) @ self.r, self.abr):
            return "A@B failed exact Freivalds check"
        return None

    def step(self):
        c = self.a @ self.b
        return 2.0 * self.n ** 3 / 1e9, _xor_fingerprint(c), self.check(c)


class Statevector:
    """Random Pauli-rotation circuit on n qubits applied forward then inverted: the state must
    return to the (dense, random-phase) initial state to 1e-10 in norm.  Each rotation is a full-state gather over a
    pseudo-random permutation, so the access pattern stresses caches/DRAM/TLB."""

    name = "statevector"

    def __init__(self, n_qubits: int = 20, n_rotations: int = 16, seed: int = 4321):
        from qcore.pauli import parity_sign
        self._sign = parity_sign
        rng = np.random.default_rng(seed)
        self.n = n_qubits
        dim = 1 << n_qubits
        self.idx = np.arange(dim, dtype=np.uint64)
        self.ops = []
        for _ in range(n_rotations):
            x = int(rng.integers(1, dim))
            z = int(rng.integers(0, dim))
            self.ops.append((np.uint64(x), np.uint64(z), (1j) ** (bin(x & z).count("1") % 4),
                             float(rng.uniform(-np.pi, np.pi))))
        # dense initial state so every amplitude is non-trivial
        self.psi0 = np.exp(1j * rng.uniform(0, 2 * np.pi, dim)) / np.sqrt(dim)

    def _rot(self, psi, x, z, phase, a):
        src = self.idx ^ x
        return np.cos(a) * psi + (1j * np.sin(a) * phase) * (self._sign(src & z) * psi[src])

    def step(self):
        psi = self.psi0
        for x, z, ph, a in self.ops:
            psi = self._rot(psi, x, z, ph, a)
        fp = _xor_fingerprint(psi)
        for x, z, ph, a in reversed(self.ops):
            psi = self._rot(psi, x, z, ph, -a)
        # first-order sensitive (fidelity would only see errors squared)
        dev = float(np.linalg.norm(psi - self.psi0))
        err = None if dev < 1e-10 else f"U^dag U |psi> deviates from |psi> by {dev:.3e}"
        work = 2 * len(self.ops) * (1 << self.n) / 1e9
        return work, fp, err


class VQEEval:
    """One exact UCCSD energy + adjoint gradient (the production VQE code path) at fixed
    parameters, checked against a reference computed once and against E >= E_FCI."""

    name = "vqe"

    def __init__(self, payload: dict):
        self.h = payload["hamiltonian"]
        self.circ = payload["circuit"]
        self.theta = payload["theta"]
        self.e_ref, self.g_ref, self.e_fci = payload["energy"], payload["gradient"], payload["e_fci"]

    def step(self):
        e, g = self.circ.energy_and_gradient(self.theta, self.h)
        err = None
        if not (abs(e - self.e_ref) < 1e-9 and np.max(np.abs(g - self.g_ref)) < 1e-8):
            err = f"energy/gradient mismatch: dE={e - self.e_ref:.3e}"
        elif e < self.e_fci - 1e-9:
            err = f"variational principle violated: E={e:.12f} < E_FCI={self.e_fci:.12f}"
        return 1.0, _xor_fingerprint(np.array([e]), g), err


def vqe_payload(molecule: str = "H2O", seed: int = 7) -> dict:
    """Built once in the parent (runs PySCF), then shipped to workers."""
    from qcore import MolecularProblem
    from VQE.ansatz import UCCSD
    p = MolecularProblem.from_library(molecule)
    circ = UCCSD().build(p)
    theta = 0.05 * np.random.default_rng(seed).standard_normal(circ.n_params)
    h = p.hamiltonian_sparse
    e, g = circ.energy_and_gradient(theta, h)
    return {"hamiltonian": h, "circuit": circ, "theta": theta, "energy": e, "gradient": g,
            "e_fci": p.e_fci, "molecule": p.name, "n_qubits": p.n_qubits}


class Transformer:
    """Forward + backward pass of the GQE GPT (dense matmul path through PyTorch).
    Verified bitwise against the worker's first step; finite-loss check every step."""

    name = "transformer"

    def __init__(self, n_embd: int = 256, n_layer: int = 4, n_head: int = 8, seq: int = 64,
                 batch: int = 8, vocab: int = 512, seed: int = 99):
        import torch
        from GQE.model import GPT
        torch.manual_seed(seed)
        self.torch = torch
        self.model = GPT(vocab, seq, n_layer=n_layer, n_head=n_head, n_embd=n_embd)
        g = torch.Generator().manual_seed(seed)
        self.tokens = torch.randint(0, vocab, (batch, seq), generator=g)
        n_params = sum(p.numel() for p in self.model.parameters())
        # ~6 FLOP per parameter per token (fwd + bwd) + attention scores (12 L T^2 d per sequence)
        self.flop = 6.0 * n_params * batch * seq + 12.0 * n_layer * seq * seq * n_embd * batch

    def step(self):
        torch = self.torch
        self.model.zero_grad(set_to_none=True)
        logits = self.model(self.tokens)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), self.tokens.reshape(-1))
        loss.backward()
        val = float(loss.item())
        gsum = np.array([float(p.grad.double().sum()) for p in self.model.parameters()])
        err = None if np.isfinite(val) and np.all(np.isfinite(gsum)) else "non-finite loss/gradient"
        return self.flop / 1e9, _xor_fingerprint(np.array([val]), gsum), err


class Triad:
    """STREAM-style triad a = b + s*c on arrays much larger than cache (bandwidth ceiling).
    NumPy executes it as two passes (a = s*c; a += b), i.e. 5 x 8 bytes of traffic per element,
    which is what is counted.  Result is verified exactly on a strided sample."""

    name = "triad"

    def __init__(self, n_elements: int = 1 << 24, seed: int = 5):
        rng = np.random.default_rng(seed)
        self.b = rng.random(n_elements)
        self.c = rng.random(n_elements)
        self.a = np.empty(n_elements)
        self.s = 3.0
        self.sample = slice(0, n_elements, max(1, n_elements // 4096))

    def step(self):
        np.multiply(self.c, self.s, out=self.a)
        np.add(self.a, self.b, out=self.a)
        ref = self.b[self.sample] + self.s * self.c[self.sample]
        err = None if np.array_equal(self.a[self.sample], ref) else "triad result mismatch"
        return 5 * 8 * self.a.size / 1e9, _xor_fingerprint(self.a[self.sample]), err


def build(name: str, params: dict):
    if name == "dgemm":
        return DGEMM(**params)
    if name == "statevector":
        return Statevector(**params)
    if name == "vqe":
        return VQEEval(params)
    if name == "transformer":
        return Transformer(**params)
    if name == "triad":
        return Triad(**params)
    raise ValueError(f"unknown workload {name!r}")
