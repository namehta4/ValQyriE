"""Exact CPU statevector circuits with adjoint-method gradients.

A circuit acts on an initial computational-basis state (or a supplied vector)
with a sequence of operations:

* ``Rotation``  U = exp(a G), a = coeff * theta[param] (or fixed when ``param is None``),
  G = i sum_k g_k sigma_k a block of commuting Pauli strings sharing one X-mask (a whole
  fermionic excitation is one block).  ``PauliRotation`` = exp(i a sigma) is the one-string
  case, e.g. RY(t) = exp(-i t Y / 2) has coeff = -1/2.
* ``Gate``           fixed unitary on a few qubits (CNOT, CZ, H, ...).

Gradients use the adjoint (reverse-mode) method: exact to machine precision
at the cost of ~3 circuit evaluations, independent of the parameter count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pauli import PauliSum, parity_sign, popcount


class _DiagCache:
    """Byte-budgeted cache of block diagonals (avoids recomputing popcounts every call)."""

    def __init__(self, budget_bytes: int = 1 << 29):
        self.budget = budget_bytes
        self.used = 0

    def admit(self, nbytes: int) -> bool:
        if self.used + nbytes > self.budget:
            return False
        self.used += nbytes
        return True


DIAG_CACHE = _DiagCache()


class Rotation:
    """U = exp(a G),  G = i * sum_k g_k sigma(x, z_k)  (all strings share one X-mask x).

    Because all strings share x, G = D X^x with D diagonal, i.e. (G psi)[c] = d(c) psi[c ^ x].
    Anti-Hermiticity gives G^2 = -|d|^2 (diagonal), so exp(aG) is an exact 2x2 rotation on each
    pair (c, c^x):  exp(aG) psi = cos(a w) psi + sin(a w)/w * G psi,  w = |d|.
    A single Pauli rotation exp(i a sigma) is the special case of one string with g = 1.
    The angle is a = coeff * theta[param] (or a = coeff when ``param`` is None).
    """

    __slots__ = ("x", "zs", "gs", "coeff", "param", "_x", "_d", "_w", "_binary")

    def __init__(self, x: int, zs, gs, coeff: float, param: int | None = None):
        self.x = int(x)
        self.zs = [int(z) for z in zs]
        self.gs = [float(g) for g in gs]
        self.coeff = float(coeff)
        self.param = param
        self._x = np.uint64(self.x)
        self._d = None

    def angle(self, theta) -> float:
        return self.coeff * theta[self.param] if self.param is not None else self.coeff

    def _diag(self, idx: np.ndarray):
        if self._d is not None and self._d.size == idx.size:
            return self._d, self._w, self._binary
        src = idx ^ self._x
        d = np.zeros(idx.size, dtype=np.complex128)
        for z, g in zip(self.zs, self.gs):
            ph = 1j * g * (1j) ** (bin(self.x & z).count("1") % 4)
            d += ph * parity_sign(src & np.uint64(z)) if z else ph
        if np.all(np.abs(d.imag) < 1e-14):
            d = np.ascontiguousarray(d.real)
        w = np.abs(d)
        binary = bool(np.all((np.abs(w) < 1e-12) | (np.abs(w - 1) < 1e-12)))
        w2 = (w > 0.5).astype(float) if binary else w
        if DIAG_CACHE.admit(d.nbytes + w2.nbytes):
            self._d, self._w, self._binary = d, w2, binary
        return d, w2, binary

    def apply_generator(self, psi: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """G psi."""
        d, _, _ = self._diag(idx)
        return d * (psi[idx ^ self._x] if self.x else psi)

    def rotate(self, psi: np.ndarray, a: float, g_psi: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """exp(aG) psi given g_psi = G psi."""
        d, w, binary = self._diag(idx)
        if self.x == 0:  # diagonal generator: exp(a d) elementwise
            return np.exp(a * d) * psi
        if binary:  # w in {0, 1}
            if w.all():
                return np.cos(a) * psi + np.sin(a) * g_psi
            return psi + ((np.cos(a) - 1.0) * w) * psi + np.sin(a) * g_psi
        aw = a * w
        safe = np.where(w > 0, w, 1.0)
        return np.cos(aw) * psi + np.where(w > 0, np.sin(aw) / safe, a) * g_psi


def PauliRotation(x: int, z: int, coeff: float, param: int | None = None) -> Rotation:
    """exp(i * coeff * theta[param] * sigma(x, z))."""
    return Rotation(x, [z], [1.0], coeff, param)


@dataclass
class Gate:
    qubits: tuple[int, ...]
    matrix: np.ndarray  # index = sum_j bit(qubits[j]) * 2^(k-1-j)  (qubits[0] most significant)
    name: str = "U"

    def apply(self, psi: np.ndarray, n: int, dagger: bool = False) -> np.ndarray:
        k = len(self.qubits)
        m = self.matrix.conj().T if dagger else self.matrix
        t = psi.reshape((2,) * n)
        axes = [n - 1 - q for q in self.qubits]
        t = np.moveaxis(t, axes, range(k))
        shp = t.shape
        t = (m @ t.reshape(1 << k, -1)).reshape(shp)
        return np.moveaxis(t, range(k), axes).reshape(-1)


GATES = {
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "cx": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex),
    "cz": np.diag([1, 1, 1, -1]).astype(complex),
}


class Generator:
    """Anti-Hermitian generator G = i * sum_k g_k sigma_k with mutually commuting sigma_k.

    exp(theta * G) = prod_k exp(i theta g_k sigma_k) exactly.
    """

    def __init__(self, x, z, g, label: str = ""):
        self.x = [int(v) for v in x]
        self.z = [int(v) for v in z]
        self.g = [float(v) for v in g]
        self.label = label
        self.commuting = all(
            (bin(self.x[a] & self.z[b]).count("1") + bin(self.z[a] & self.x[b]).count("1")) % 2 == 0
            for a in range(len(self.x)) for b in range(a))

    @classmethod
    def from_pauli_sum(cls, op: PauliSum, label: str = "", tol: float = 1e-12) -> "Generator":
        op = op.chop(tol)
        if not op.is_antihermitian():
            raise ValueError("generator must be anti-Hermitian")
        h = op.hermitian_coeffs  # = i * g
        order = np.lexsort((op.z, op.x))
        return cls(op.x[order], op.z[order], np.imag(h)[order], label)

    def to_pauli_sum(self, n_qubits: int) -> PauliSum:
        return PauliSum.from_hermitian_terms(n_qubits, self.x, self.z, 1j * np.array(self.g))

    def __len__(self) -> int:
        return len(self.g)

    def __repr__(self) -> str:
        return f"Generator({self.label or '?'}, {len(self)} Pauli strings)"


class Circuit:
    def __init__(self, n_qubits: int, initial_state: int | np.ndarray = 0):
        self.n_qubits = n_qubits
        self.initial_state = initial_state
        self.ops: list[Rotation | Gate] = []
        self.n_params = 0
        self.param_labels: list[str] = []
        self._idx = None

    # ---------------------------------------------------------- building
    def new_parameter(self, label: str = "") -> int:
        self.n_params += 1
        self.param_labels.append(label or f"t{self.n_params - 1}")
        return self.n_params - 1

    def add_rotation(self, x: int, z: int, coeff: float, param: int | None = None):
        """exp(i * coeff * theta[param] * sigma(x, z))."""
        self.ops.append(PauliRotation(int(x), int(z), float(coeff), param))

    def add_pauli_rotation(self, label: str, coeff: float, param: int | None = None):
        p = PauliSum.from_string(self.n_qubits, label)
        self.add_rotation(int(p.x[0]), int(p.z[0]), coeff, param)

    def add_generator(self, gen: Generator, param: int | None, scale: float = 1.0):
        """Append exp(scale * theta[param] * G) (or exp(scale * G) if param is None).

        Exact for generators with mutually commuting Pauli strings: strings are grouped by
        X-mask and each group is applied as one block rotation.
        """
        if not gen.commuting:
            raise ValueError(f"{gen} has non-commuting Pauli strings; exponential would be approximate")
        groups: dict[int, tuple[list, list]] = {}
        for x, z, g in zip(gen.x, gen.z, gen.g):
            zs, gs = groups.setdefault(x, ([], []))
            zs.append(z)
            gs.append(g)
        for x, (zs, gs) in groups.items():
            self.ops.append(Rotation(x, zs, gs, scale, param))

    def add_gate(self, name_or_matrix, qubits, name: str | None = None):
        m = GATES[name_or_matrix] if isinstance(name_or_matrix, str) else np.asarray(name_or_matrix, complex)
        self.ops.append(Gate(tuple(qubits), m, name or str(name_or_matrix)))

    # ---------------------------------------------------------- simulation
    @property
    def idx(self) -> np.ndarray:
        if self._idx is None or self._idx.size != (1 << self.n_qubits):
            self._idx = np.arange(1 << self.n_qubits, dtype=np.uint64)
        return self._idx

    def initial_vector(self) -> np.ndarray:
        if isinstance(self.initial_state, (int, np.integer)):
            psi = np.zeros(1 << self.n_qubits, dtype=np.complex128)
            psi[int(self.initial_state)] = 1.0
            return psi
        return np.asarray(self.initial_state, dtype=np.complex128).copy()

    def state(self, theta=None, psi: np.ndarray | None = None) -> np.ndarray:
        theta = np.zeros(self.n_params) if theta is None else np.asarray(theta, float)
        psi = self.initial_vector() if psi is None else psi.astype(np.complex128, copy=True)
        idx, n = self.idx, self.n_qubits
        for op in self.ops:
            if isinstance(op, Rotation):
                a = op.angle(theta)
                if a != 0.0:
                    psi = op.rotate(psi, a, op.apply_generator(psi, idx), idx)
            else:
                psi = op.apply(psi, n)
        return psi

    def energy(self, theta, hamiltonian, diag=None) -> float:
        psi = self.state(theta)
        hpsi = hamiltonian @ psi
        if diag is not None:
            hpsi = hpsi + diag * psi
        return float(np.real(np.vdot(psi, hpsi)))

    def energy_and_gradient(self, theta, hamiltonian, diag=None) -> tuple[float, np.ndarray]:
        """<psi|H|psi> and its exact gradient (adjoint method).

        ``hamiltonian`` is anything supporting ``@`` (sparse matrix / LinearOperator);
        ``diag`` is an optional extra diagonal (e.g. symmetry penalties).
        """
        theta = np.asarray(theta, float)
        idx, n = self.idx, self.n_qubits
        phi = self.state(theta)
        lam = hamiltonian @ phi
        if diag is not None:
            lam = lam + diag * phi
        energy = float(np.real(np.vdot(phi, lam)))
        grad = np.zeros(self.n_params)
        for op in reversed(self.ops):
            if isinstance(op, Rotation):
                a = op.angle(theta)
                g_phi = op.apply_generator(phi, idx)
                if op.param is not None:
                    # d/dtheta exp(c theta G) = c G U  ->  2 c Re<lam|G|phi>
                    grad[op.param] += 2.0 * op.coeff * np.real(np.vdot(lam, g_phi))
                if a != 0.0:
                    phi = op.rotate(phi, -a, g_phi, idx)  # U^dag = exp(-aG); reuses G phi
                    lam = op.rotate(lam, -a, op.apply_generator(lam, idx), idx)
            else:
                phi = op.apply(phi, n, dagger=True)
                lam = op.apply(lam, n, dagger=True)
        return energy, grad

    def depth_summary(self) -> dict:
        strings = [(op.x, z) for op in self.ops if isinstance(op, Rotation) for z in op.zs]
        weights = [bin(x | z).count("1") for x, z in strings]
        # CNOT count of a standard CNOT-ladder implementation of exp(i a P): 2 (w - 1)
        return {"n_params": self.n_params, "pauli_rotations": len(strings),
                "fixed_gates": sum(isinstance(op, Gate) for op in self.ops),
                "max_pauli_weight": max(weights, default=0),
                "cnot_estimate": sum(2 * (w - 1) for w in weights if w > 1)
                + sum(1 for op in self.ops if isinstance(op, Gate) and len(op.qubits) == 2)}


def expectation(psi: np.ndarray, op) -> float:
    return float(np.real(np.vdot(psi, op @ psi)))


__all__ = ["Circuit", "Gate", "Generator", "PauliRotation", "Rotation", "GATES", "expectation",
           "popcount"]
