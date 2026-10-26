"""Benchmark cases: a language-neutral definition of the problem every implementation solves.

A case is a UCCSD circuit on the Hartree-Fock state of a molecular Hamiltonian in an active
space, under the Jordan-Wigner encoding:

* Spin orbitals are interleaved: qubit 2p = spatial orbital p, spin up (alpha); qubit 2p + 1 =
  orbital p, spin down (beta). Basis state |b>: bit j of the integer b = occupation of qubit j.
* H = e_core + sum_pq,s h1[p,q] a+_ps a_qs
        + 1/2 sum_pqrs,st h2[p,q,r,s] a+_ps a+_rt a_st a_qs      (h2 in chemists' notation (pq|rs))
* Fermionic operators: a+_j = Z_0 ... Z_{j-1} (X_j - i Y_j)/2 (Jordan-Wigner).
* Initial state: the Hartree-Fock determinant |HF> (``hf_occupation``; the integer ``hf_index``).
* Excitation k = (occupied, virtual) spin-orbital tuples, one or two of each:
  T_k = a+_{v1} [a+_{v2}] [a_{o2}] a_{o1},  G_k = T_k - T_k^dagger  (anti-Hermitian).
* Circuit: |psi(theta)> = exp(theta_{K-1} G_{K-1}) ... exp(theta_1 G_1) exp(theta_0 G_0) |HF>,
  i.e. excitation 0 acts first. Each exponential is exact (no Trotter error inside it).
* Task: E(theta) = <psi|H|psi> and its gradient dE/dtheta_k, in double precision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# the cases as files (see build_case)
DATA_DIR = Path(__file__).parent / "data"

# name -> how to build it; "role" says where it is used in the challenge
CASES = {
    "lih-12q": {"role": "development", "molecule": "LiH", "basis": "sto-3g", "active": "default",
                "description": "LiH / STO-3G, all electrons (4e, 6o)"},
    "h2o-16q": {"role": "development", "molecule": "H2O", "basis": "6-31g", "active": (8, 8),
                "description": "H2O / 6-31G, (8e, 8o)"},
    "h2o-20q": {"role": "development", "molecule": "H2O", "basis": "6-31g", "active": (8, 10),
                "description": "H2O / 6-31G, (8e, 10o)"},
    "h2o-24q": {"role": "fixed case (stages 1-6)", "molecule": "H2O", "basis": "6-31g", "active": (8, 12),
                "description": "H2O / 6-31G, O 1s frozen: (8e, 12o), 24 qubits"},
    "anthracene-28q": {"role": "stretch case (stage 5)", "molecule": "Anthracene", "basis": "sto-3g",
                       "active": "default", "fci": False,
                       "description": "anthracene / STO-3G, pi space (14e, 14o) by AVAS, 28 qubits"},
}


@dataclass
class CaseData:
    name: str
    description: str
    n_orbitals: int
    n_alpha: int
    n_beta: int
    h1: np.ndarray
    h2: np.ndarray
    e_core: float
    excitations: list
    hf_occupation: np.ndarray
    e_hf: float
    e_fci: float = float("nan")
    meta: dict = field(default_factory=dict)

    @property
    def n_qubits(self) -> int:
        return 2 * self.n_orbitals

    @property
    def n_params(self) -> int:
        return len(self.excitations)

    @property
    def hf_index(self) -> int:
        return int(sum(1 << j for j, o in enumerate(self.hf_occupation) if o))

    # ------------------------------------------------------------- files
    def save(self, directory) -> Path:
        """Write ``<name>.npz`` (arrays) and ``<name>.json`` (scalars + the definition)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        exc = np.full((self.n_params, 4), -1, dtype=np.int64)  # columns o1, o2, v1, v2 (-1 = unused)
        for k, (occ, virt) in enumerate(self.excitations):
            exc[k, :len(occ)] = occ
            exc[k, 2:2 + len(virt)] = virt
        np.savez(d / f"{self.name}.npz", h1=self.h1, h2=self.h2, excitations=exc,
                 hf_occupation=np.asarray(self.hf_occupation, dtype=np.int8))
        meta = {"name": self.name, "description": self.description, "n_orbitals": self.n_orbitals,
                "n_qubits": self.n_qubits, "n_alpha": self.n_alpha, "n_beta": self.n_beta,
                "n_params": self.n_params, "hf_index": self.hf_index, "e_core": self.e_core,
                "e_hf": self.e_hf, "e_fci": None if np.isnan(self.e_fci) else self.e_fci,
                "arrays": {"h1": "[p, q]", "h2": "[p, q, r, s] = (pq|rs), chemists' notation",
                           "excitations": "[k, (o1, o2, v1, v2)] spin-orbital indices, -1 = unused",
                           "hf_occupation": "[j] occupation of spin orbital / qubit j"},
                "definition": __doc__, **self.meta}
        (d / f"{self.name}.json").write_text(json.dumps(meta, indent=1))
        return d / f"{self.name}.npz"

    @classmethod
    def load(cls, path) -> "CaseData":
        path = Path(path)
        arr = np.load(path.with_suffix(".npz"))
        meta = json.loads(path.with_suffix(".json").read_text())
        excs = []
        for row in arr["excitations"]:
            occ = tuple(int(v) for v in row[:2] if v >= 0)
            virt = tuple(int(v) for v in row[2:] if v >= 0)
            excs.append((occ, virt))
        e_fci = meta.get("e_fci")
        return cls(meta["name"], meta["description"], meta["n_orbitals"], meta["n_alpha"], meta["n_beta"],
                   arr["h1"], arr["h2"], meta["e_core"], excs, arr["hf_occupation"], meta["e_hf"],
                   float("nan") if e_fci is None else e_fci,
                   meta={k: meta[k] for k in ("role", "molecule") if k in meta})


def build_case(name: str, compute_fci: bool | None = None, from_pyscf: bool = False) -> CaseData:
    """A named case, loaded from ``bench/data/<name>.npz``: the integrals the stored reference
    answers were computed from, identical on every machine.

    ``from_pyscf=True`` builds the case afresh instead. That agrees with the stored one only up to
    the sign of each orbital: LAPACK builds (Accelerate, OpenBLAS, MKL) return eigenvectors with
    different signs, and an excitation's parameter changes sign with its orbitals, so a rebuilt case
    need not reproduce the reference points. The exact energy is then computed unless the case says
    otherwise (the 28-qubit case: 11.8M determinants) or ``compute_fci`` overrides it."""
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; choose from {', '.join(CASES)}")
    if not from_pyscf:
        case = CaseData.load(DATA_DIR / f"{name}.npz")
        if compute_fci and np.isnan(case.e_fci):
            from pyscf import fci
            case.e_fci = float(fci.direct_spin1.FCI().kernel(case.h1, case.h2, case.n_orbitals,
                                                             (case.n_alpha, case.n_beta),
                                                             ecore=case.e_core)[0])
        return case
    from qcore import MolecularProblem
    from VQE.ansatz import UCCSD
    spec = CASES[name]
    fci = spec.get("fci", True) if compute_fci is None else compute_fci
    p = MolecularProblem.from_library(spec["molecule"], basis=spec["basis"], active_space=spec["active"],
                                      compute_fci=fci)
    excs = UCCSD().excitation_list(p)
    return CaseData(name, spec["description"], p.n_orbitals, p.n_alpha, p.n_beta, np.asarray(p.h1),
                    np.asarray(p.h2), p.e_core, excs, p.hf_occupation.copy(), p.e_hf, p.e_fci,
                    meta={"role": spec["role"], "molecule": p.name})
