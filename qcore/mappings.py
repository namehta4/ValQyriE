"""Fermion-to-qubit mappings as linear (GF(2)) encodings.

A linear encoding stores qubit values q = beta . n (mod 2) for occupation
vector n.  Jordan-Wigner (beta = I), parity (lower-triangular ones) and
Bravyi-Kitaev (Fenwick tree) are all of this form [Seeley, Richard & Love,
J. Chem. Phys. 137, 224109 (2012)].  From beta alone the Majorana operators are

    c_p = (-1)^{sum_{j<p} n_j} flip(n_p)          -> Z^{parity set} X^{update set}
    d_p = i c_p (-1)^{n_p}

and a_p^dag = (c_p - i d_p)/2,  a_p = (c_p + i d_p)/2.

Spin-orbital ordering used throughout the package is interleaved:
mode 2k = spatial orbital k spin alpha, mode 2k+1 = spatial orbital k spin beta.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .pauli import PauliSum, popcount


def _gf2_inv(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    a = np.concatenate([m.astype(np.uint8) % 2, np.eye(n, dtype=np.uint8)], axis=1)
    for col in range(n):
        piv = next((r for r in range(col, n) if a[r, col]), None)
        if piv is None:
            raise ValueError("encoding matrix is singular over GF(2)")
        a[[col, piv]] = a[[piv, col]]
        for r in range(n):
            if r != col and a[r, col]:
                a[r] ^= a[col]
    return a[:, n:]


def _mask(bits) -> int:
    return int(sum(1 << i for i, b in enumerate(bits) if b))


class FermionQubitMapping:
    """Linear fermion-to-qubit encoding defined by a binary matrix ``beta``."""

    name = "linear"

    def __init__(self, n_modes: int, beta: np.ndarray | None = None):
        self.n_modes = self.n_qubits = int(n_modes)
        self.beta = (np.eye(n_modes, dtype=np.uint8) if beta is None
                     else np.asarray(beta, dtype=np.uint8) % 2)
        self.beta_inv = _gf2_inv(self.beta)
        n = n_modes
        # occupation n_p = parity(q & occ_mask[p])
        self.occ_masks = [_mask(self.beta_inv[p]) for p in range(n)]
        self.update_masks = [_mask(self.beta[:, p]) for p in range(n)]
        self.parity_masks = [_mask(self.beta_inv[:p].sum(axis=0) % 2) for p in range(n)]
        self._ops: dict = {}

    # ------------------------------------------------------------ operators
    def majoranas(self, p: int) -> tuple[PauliSum, PauliSum]:
        n = self.n_qubits
        flip = PauliSum(n, [self.update_masks[p]], [0], [1.0])
        par = PauliSum(n, [0], [self.parity_masks[p]], [1.0])
        c = par * flip
        d = 1j * c * PauliSum(n, [0], [self.occ_masks[p]], [1.0])
        return c, d

    def creation(self, p: int) -> PauliSum:
        key = ("cr", p)
        if key not in self._ops:
            c, d = self.majoranas(p)
            self._ops[key] = (c - 1j * d) * 0.5
        return self._ops[key]

    def annihilation(self, p: int) -> PauliSum:
        key = ("an", p)
        if key not in self._ops:
            c, d = self.majoranas(p)
            self._ops[key] = (c + 1j * d) * 0.5
        return self._ops[key]

    def hopping(self, p: int, q: int) -> PauliSum:
        """E_pq = a_p^dag a_q."""
        key = ("E", p, q)
        if key not in self._ops:
            self._ops[key] = self.creation(p) * self.annihilation(q)
        return self._ops[key]

    def number(self, p: int) -> PauliSum:
        return self.hopping(p, p)

    def ladder_product(self, ops: list[tuple[int, int]]) -> PauliSum:
        """Product of ladder operators [(mode, 1=create / 0=annihilate), ...]."""
        out = PauliSum.identity(self.n_qubits)
        for p, dag in ops:
            out = out * (self.creation(p) if dag else self.annihilation(p))
        return out

    # ---------------------------------------------------------------- states
    def encode(self, occupation) -> int:
        """Computational-basis index for an occupation-number vector."""
        n = np.asarray(occupation, dtype=np.uint8) % 2
        q = (self.beta @ n) % 2
        return _mask(q)

    def occupations(self, indices: np.ndarray) -> np.ndarray:
        """Occupation numbers (len(indices) x n_modes) of computational basis states."""
        idx = np.asarray(indices, dtype=np.uint64)
        return np.stack([popcount(idx & np.uint64(m)) & 1 for m in self.occ_masks], axis=1)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_modes={self.n_modes})"


class JordanWigner(FermionQubitMapping):
    name = "jordan_wigner"

    def __init__(self, n_modes: int):
        super().__init__(n_modes, np.eye(n_modes, dtype=np.uint8))


class Parity(FermionQubitMapping):
    name = "parity"

    def __init__(self, n_modes: int):
        super().__init__(n_modes, np.tril(np.ones((n_modes, n_modes), dtype=np.uint8)))


class BravyiKitaev(FermionQubitMapping):
    """Fenwick-tree Bravyi-Kitaev: qubit j stores parity of modes (j & (j+1)) .. j."""

    name = "bravyi_kitaev"

    def __init__(self, n_modes: int):
        beta = np.zeros((n_modes, n_modes), dtype=np.uint8)
        for j in range(n_modes):
            beta[j, (j & (j + 1)):j + 1] = 1
        super().__init__(n_modes, beta)


MAPPINGS = {
    "jw": JordanWigner, "jordan_wigner": JordanWigner, "jordan-wigner": JordanWigner,
    "bk": BravyiKitaev, "bravyi_kitaev": BravyiKitaev, "bravyi-kitaev": BravyiKitaev,
    "parity": Parity,
}


@lru_cache(maxsize=None)
def get_mapping(name: str, n_modes: int) -> FermionQubitMapping:
    try:
        return MAPPINGS[name.lower()](n_modes)
    except KeyError:
        raise ValueError(f"unknown mapping {name!r}; choose from {sorted(set(MAPPINGS))}") from None
