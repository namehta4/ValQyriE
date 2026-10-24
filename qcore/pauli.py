"""Vectorised Pauli-operator algebra in symplectic (X^x Z^z) form.

Every operator is stored as  O = sum_k c_k  X^{x_k} Z^{z_k}  where x_k, z_k are
bit-masks over qubits (bit i <-> qubit i) and X^x Z^z means the tensor product
of X^{x_i} Z^{z_i} on each qubit i.  In this form the product rule carries only
a sign,

    (X^{x1} Z^{z1}) (X^{x2} Z^{z2}) = (-1)^{|z1 & x2|} X^{x1^x2} Z^{z1^z2},

which makes products exact and cheap.  The usual Hermitian Pauli string with Y's
is related by  X^x Z^z = (-i)^{|x & z|} sigma(x, z)  (since Y = iXZ).

Basis convention: computational basis index b has qubit i in state bit i of b
(little-endian).  The statevector entry psi[b] is the amplitude of |b>.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp

MAX_QUBITS = 32  # x, z packed into one uint64 key for deduplication


if hasattr(np, "bitwise_count"):  # NumPy >= 2.0
    def popcount(a: np.ndarray) -> np.ndarray:
        return np.bitwise_count(np.asarray(a, dtype=np.uint64)).astype(np.int64)
else:  # NumPy 1.x (e.g. Intel macOS, where the last PyTorch wheels require NumPy < 2)
    _BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)

    def popcount(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.uint64)
        flat = np.ascontiguousarray(a.reshape(-1))
        return _BYTE_POPCOUNT[flat.view(np.uint8)].reshape(-1, 8).sum(axis=1).reshape(a.shape)


def parity_sign(a: np.ndarray) -> np.ndarray:
    """(-1)^{popcount(a)} as float array."""
    return 1.0 - 2.0 * (popcount(a) & 1)


def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform: out[c] = sum_z a[z] (-1)^{|c & z|}."""
    n = a.size
    out = np.array(a, dtype=np.complex128, copy=True)
    h = 1
    while h < n:
        v = out.reshape(-1, 2, h)
        lo, hi = v[:, 0, :].copy(), v[:, 1, :].copy()
        v[:, 0, :] = lo + hi
        v[:, 1, :] = lo - hi
        h *= 2
    return out


class PauliSum:
    """Linear combination of Pauli strings on ``n_qubits`` qubits."""

    __slots__ = ("n_qubits", "x", "z", "c")

    def __init__(self, n_qubits: int, x=(), z=(), c=(), simplify: bool = True):
        if n_qubits > MAX_QUBITS:
            raise ValueError(f"at most {MAX_QUBITS} qubits are supported")
        self.n_qubits = int(n_qubits)
        self.x = np.asarray(x, dtype=np.uint64).reshape(-1)
        self.z = np.asarray(z, dtype=np.uint64).reshape(-1)
        self.c = np.asarray(c, dtype=np.complex128).reshape(-1)
        if simplify:
            self._simplify()

    # ------------------------------------------------------------------ build
    @classmethod
    def zero(cls, n_qubits: int) -> "PauliSum":
        return cls(n_qubits)

    @classmethod
    def identity(cls, n_qubits: int, coeff: complex = 1.0) -> "PauliSum":
        return cls(n_qubits, [0], [0], [coeff])

    @classmethod
    def from_string(cls, n_qubits: int, label: str, coeff: complex = 1.0) -> "PauliSum":
        """Hermitian Pauli string, e.g. ``"X0 Y1 Z3"`` (``""`` is identity)."""
        x = z = 0
        for tok in label.split():
            op, q = tok[0].upper(), int(tok[1:])
            if op in "XY":
                x |= 1 << q
            if op in "ZY":
                z |= 1 << q
        ny = bin(x & z).count("1")
        # sigma = i^{ny} X^x Z^z
        return cls(n_qubits, [x], [z], [coeff * (1j) ** ny])

    @classmethod
    def from_hermitian_terms(cls, n_qubits: int, x, z, coeffs) -> "PauliSum":
        """Build from Hermitian Pauli strings sigma(x, z) with given coefficients."""
        x = np.asarray(x, dtype=np.uint64)
        z = np.asarray(z, dtype=np.uint64)
        ph = (1j) ** (popcount(x & z) % 4)
        return cls(n_qubits, x, z, np.asarray(coeffs) * ph)

    # ------------------------------------------------------------- utilities
    def _simplify(self, tol: float = 0.0) -> None:
        if self.c.size == 0:
            return
        key = (self.x << np.uint64(32)) | self.z
        uniq, inv = np.unique(key, return_inverse=True)
        c = np.zeros(uniq.size, dtype=np.complex128)
        np.add.at(c, inv, self.c)
        keep = np.abs(c) > tol
        self.x = (uniq >> np.uint64(32))[keep]
        self.z = (uniq & np.uint64(0xFFFFFFFF))[keep]
        self.c = c[keep]

    def chop(self, tol: float = 1e-12) -> "PauliSum":
        out = self.copy()
        out._simplify(tol)
        return out

    def copy(self) -> "PauliSum":
        return PauliSum(self.n_qubits, self.x.copy(), self.z.copy(), self.c.copy(), simplify=False)

    def __len__(self) -> int:
        return self.c.size

    @property
    def hermitian_coeffs(self) -> np.ndarray:
        """Coefficients w.r.t. Hermitian Pauli strings sigma(x, z)."""
        return self.c * (-1j) ** (popcount(self.x & self.z) % 4)

    def terms(self) -> list[tuple[str, complex]]:
        """Human-readable list of (label, coefficient) with Hermitian strings."""
        out = []
        for x, z, h in zip(self.x, self.z, self.hermitian_coeffs):
            ops = []
            for q in range(self.n_qubits):
                bx, bz = (int(x) >> q) & 1, (int(z) >> q) & 1
                if bx or bz:
                    ops.append(("Y" if bx and bz else "X" if bx else "Z") + str(q))
            out.append((" ".join(ops), complex(h)))
        return out

    def constant(self) -> complex:
        m = (self.x == 0) & (self.z == 0)
        return complex(self.c[m].sum())

    # -------------------------------------------------------------- algebra
    def _coerce(self, other) -> "PauliSum":
        if isinstance(other, PauliSum):
            if other.n_qubits != self.n_qubits:
                raise ValueError("qubit-count mismatch")
            return other
        return PauliSum.identity(self.n_qubits, complex(other))

    def __add__(self, other) -> "PauliSum":
        o = self._coerce(other)
        return PauliSum(self.n_qubits, np.concatenate([self.x, o.x]),
                        np.concatenate([self.z, o.z]), np.concatenate([self.c, o.c]))

    __radd__ = __add__

    def __neg__(self) -> "PauliSum":
        return PauliSum(self.n_qubits, self.x, self.z, -self.c, simplify=False)

    def __sub__(self, other) -> "PauliSum":
        return self + (-self._coerce(other))

    def __rsub__(self, other) -> "PauliSum":
        return (-self) + other

    def __mul__(self, other) -> "PauliSum":
        if not isinstance(other, PauliSum):
            return PauliSum(self.n_qubits, self.x, self.z, self.c * complex(other), simplify=False)
        o = self._coerce(other)
        x = self.x[:, None] ^ o.x[None, :]
        z = self.z[:, None] ^ o.z[None, :]
        sign = parity_sign(self.z[:, None] & o.x[None, :])
        c = self.c[:, None] * o.c[None, :] * sign
        return PauliSum(self.n_qubits, x, z, c)

    def __rmul__(self, other) -> "PauliSum":
        return PauliSum(self.n_qubits, self.x, self.z, self.c * complex(other), simplify=False)

    def __truediv__(self, scalar) -> "PauliSum":
        return self * (1.0 / scalar)

    def dagger(self) -> "PauliSum":
        # (X^x Z^z)^dag = Z^z X^x = (-1)^{|x & z|} X^x Z^z
        return PauliSum(self.n_qubits, self.x, self.z,
                        np.conj(self.c) * parity_sign(self.x & self.z), simplify=False)

    def commutator(self, other: "PauliSum") -> "PauliSum":
        return self * other - other * self

    def is_hermitian(self, tol: float = 1e-10) -> bool:
        return len((self - self.dagger()).chop(tol)) == 0

    def is_antihermitian(self, tol: float = 1e-10) -> bool:
        return len((self + self.dagger()).chop(tol)) == 0

    def norm1(self) -> float:
        return float(np.abs(self.c).sum())

    # ---------------------------------------------------------- statevector
    def apply(self, psi: np.ndarray) -> np.ndarray:
        """Return O|psi> (matrix-free, O(n_terms * 2^n))."""
        idx = np.arange(psi.size, dtype=np.uint64)
        out = np.zeros_like(psi, dtype=np.complex128)
        for x, z, c in zip(self.x, self.z, self.c):
            src = idx ^ x
            out += c * parity_sign(src & z) * psi[src]
        return out

    def to_sparse(self, tol: float = 1e-12) -> sp.csr_matrix:
        """Sparse CSR matrix in the computational basis.

        Terms are grouped by X-mask; for each group the row values
        d(r) = sum_k c_k (-1)^{|(r^x) & z_k|} are produced with a fast
        Walsh-Hadamard transform when that is cheaper than direct summation.
        Entries cancelling to below ``tol`` are dropped, so number-conserving
        Hamiltonians end up with their genuine (much smaller) sparsity.
        """
        n = self.n_qubits
        dim = 1 << n
        idx = np.arange(dim, dtype=np.uint64)
        rows, cols, vals = [], [], []
        order = np.argsort(self.x, kind="stable")
        xs, zs, cs = self.x[order], self.z[order], self.c[order]
        bounds = np.flatnonzero(np.diff(xs)) + 1
        for sl in np.split(np.arange(xs.size), bounds):
            if sl.size == 0:
                continue
            x = xs[sl[0]]
            col = idx ^ x
            if sl.size > n:
                f = np.zeros(dim, dtype=np.complex128)
                np.add.at(f, zs[sl].astype(np.int64), cs[sl])
                d = fwht(f)[col.astype(np.int64)]
            else:
                d = np.zeros(dim, dtype=np.complex128)
                for z, c in zip(zs[sl], cs[sl]):
                    d += c * parity_sign(col & z)
            keep = np.abs(d) > tol
            rows.append(idx[keep].astype(np.int64))
            cols.append(col[keep].astype(np.int64))
            vals.append(d[keep])
        if not rows:
            return sp.csr_matrix((dim, dim), dtype=np.complex128)
        m = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(dim, dim))
        return m.tocsr()

    def to_dense(self) -> np.ndarray:
        return self.to_sparse().toarray()

    def __repr__(self) -> str:
        head = ", ".join(f"{h.real:+.6f}*[{l}]" for l, h in self.terms()[:6])
        more = "" if len(self) <= 6 else f", ... ({len(self)} terms)"
        return f"PauliSum({self.n_qubits}q: {head}{more})"


def hermitian_strings_commute(x1: int, z1: int, x2: int, z2: int) -> bool:
    return (bin(x1 & z2).count("1") + bin(z1 & x2).count("1")) % 2 == 0


def sum_ops(ops: Iterable[PauliSum], n_qubits: int) -> PauliSum:
    ops = list(ops)
    if not ops:
        return PauliSum.zero(n_qubits)
    return PauliSum(n_qubits, np.concatenate([o.x for o in ops]),
                    np.concatenate([o.z for o in ops]), np.concatenate([o.c for o in ops]))
