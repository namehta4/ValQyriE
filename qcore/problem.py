"""Molecular electronic-structure problem: PySCF integrals -> qubit Hamiltonian.

The mean-field reference is RHF (closed shell) or ROHF (open shell).  An active
space (frozen doubly occupied core + truncated virtuals) is handled by PySCF's
CASCI machinery, which provides the effective one-body integrals, the core
energy and the exact (FCI-in-active-space) reference energy used to validate
the variational algorithms.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import scipy.sparse.linalg as spla

from .active_space import AutoActiveSpace, parse_active_space, select_active_space
from .hamiltonian import molecular_qubit_hamiltonian, s2_operator
from .mappings import FermionQubitMapping, get_mapping
from .molecules import get_molecule
from .pauli import PauliSum

HARTREE_TO_KCAL = 627.509474
CHEMICAL_ACCURACY = 1.6e-3  # Hartree (1 kcal/mol)


class MolecularProblem:
    """Electronic ground-state problem for a molecule in an active space.

    Parameters
    ----------
    geometry : PySCF atom spec (string or list of (symbol, (x, y, z))) in ``unit``.
    basis, charge, spin : as in PySCF (``spin`` = N_alpha - N_beta).
    active_electrons, active_orbitals : active space; ``None`` = all electrons/orbitals.
    auto_active : ``AutoActiveSpace`` to choose the active orbitals from MP2 occupations
        instead (``qcore.active_space``); overrides ``active_electrons``/``active_orbitals``.
    mapping : "jw", "bk" or "parity".
    """

    def __init__(self, geometry, basis: str = "sto-3g", charge: int = 0, spin: int = 0,
                 active_electrons: int | None = None, active_orbitals: int | None = None,
                 mapping: str = "jw", name: str | None = None, unit: str = "Angstrom",
                 auto_active: AutoActiveSpace | None = None, verbose: bool = False):
        from pyscf import ao2mo, gto, mcscf, scf

        self.name = name or "molecule"
        self.basis = basis
        self.mol = gto.M(atom=geometry, basis=basis, charge=charge, spin=spin, unit=unit,
                         verbose=4 if verbose else 0)
        mf = scf.RHF(self.mol) if spin == 0 else scf.ROHF(self.mol)
        mf.conv_tol = 1e-11
        mf.max_cycle = 200
        mf.kernel()
        if not mf.converged:
            mf = mf.newton()
            mf.kernel()
            if not mf.converged:
                raise RuntimeError(f"SCF did not converge for {self.name}")
        self.mf = mf
        self.e_hf = float(mf.e_tot)

        n_elec = self.mol.nelectron
        n_mo = mf.mo_coeff.shape[1]
        self.active_choice = None
        if auto_active is not None:
            self.active_choice = select_active_space(mf, auto_active)
            active_electrons = self.active_choice.n_electrons
            active_orbitals = len(self.active_choice.orbitals)
        ncas = n_mo if active_orbitals is None else int(active_orbitals)
        nelecas = n_elec if active_electrons is None else int(active_electrons)
        if (nelecas - spin) % 2:
            raise ValueError("active electron count incompatible with spin")
        if (n_elec - nelecas) % 2 or nelecas > 2 * ncas:
            raise ValueError("invalid active space")
        self.n_alpha = (nelecas + spin) // 2
        self.n_beta = (nelecas - spin) // 2
        self.n_orbitals = ncas
        self.n_electrons = nelecas
        self.n_frozen = (n_elec - nelecas) // 2
        self.n_qubits = 2 * ncas

        cas = mcscf.CASCI(mf, ncas, (self.n_alpha, self.n_beta))
        cas.verbose = 0
        if self.active_choice is not None:  # possibly non-contiguous canonical orbitals
            cas.mo_coeff = cas.sort_mo(self.active_choice.orbitals, base=0)
        h1, e_core = cas.get_h1eff()
        self.h1 = np.asarray(h1)
        self.e_core = float(e_core)
        self.h2 = ao2mo.restore(1, cas.get_h2eff(), ncas)
        # exact lowest energy in the active space for this (N_alpha, N_beta) sector --
        # the same sector the qubit algorithms are compared against
        self.e_fci = float(cas.kernel()[0])
        self.mapping: FermionQubitMapping = get_mapping(mapping, self.n_qubits)

    # --------------------------------------------------------- constructors
    @classmethod
    def from_library(cls, name: str, bond_length: float | None = None, basis: str = "sto-3g",
                     mapping: str = "jw", active_space="default", **kw) -> "MolecularProblem":
        """Build a benchmark molecule. ``active_space``: "default" (the library's choice),
        "full", (n_elec, n_orb), or "auto" / ``AutoActiveSpace(...)`` (MP2 occupations)."""
        spec = get_molecule(name)
        r = spec.r_eq if bond_length is None else bond_length
        active_space = parse_active_space(active_space)
        if active_space == "default":
            active_space = spec.active
        elif active_space == "full":
            active_space = None
        if isinstance(active_space, AutoActiveSpace):
            kw["auto_active"], active_space = active_space, None
        ae, ao = (None, None) if active_space is None else active_space
        return cls(spec.geometry(r), basis=basis, charge=spec.charge, spin=spec.spin,
                   active_electrons=ae, active_orbitals=ao, mapping=mapping,
                   name=f"{spec.name}(r={r:.4f})", **kw)

    @classmethod
    def from_integrals(cls, h1, h2, e_core: float, n_alpha: int, n_beta: int, mapping="jw",
                       name: str = "integrals", e_fci: float | None = None) -> "MolecularProblem":
        """Problem defined directly by active-space integrals (h1[p,q], h2 in chemists' notation,
        any PySCF symmetry packing). ``e_fci`` defaults to an exact PySCF FCI; pass ``nan`` to skip
        it. ``e_hf`` is the energy of the lowest determinant (orbitals 0.. occupied)."""
        from pyscf import ao2mo, fci
        self = cls.__new__(cls)
        n = np.asarray(h1).shape[0]
        self.name, self.basis, self.mol, self.mf, self.active_choice = name, "-", None, None, None
        self.h1 = np.asarray(h1, dtype=float)
        self.h2 = ao2mo.restore(1, np.asarray(h2), n)
        self.e_core = float(e_core)
        self.n_alpha, self.n_beta = int(n_alpha), int(n_beta)
        self.n_orbitals, self.n_electrons, self.n_frozen = n, self.n_alpha + self.n_beta, 0
        self.n_qubits = 2 * n
        self.mapping = get_mapping(mapping, self.n_qubits) if isinstance(mapping, str) else mapping
        if e_fci is None:
            e_fci = fci.direct_spin1.kernel(self.h1, self.h2, n, (self.n_alpha, self.n_beta), ecore=self.e_core)[0]
        self.e_fci = float(e_fci)
        d = np.diag(self.h1)
        jj = np.einsum("ppqq->pq", self.h2)
        kk = np.einsum("pqqp->pq", self.h2)
        oa, ob = np.arange(self.n_alpha), np.arange(self.n_beta)
        self.e_hf = float(self.e_core + d[oa].sum() + d[ob].sum()
                          + 0.5 * (jj[np.ix_(oa, oa)] - kk[np.ix_(oa, oa)]).sum()
                          + 0.5 * (jj[np.ix_(ob, ob)] - kk[np.ix_(ob, ob)]).sum() + jj[np.ix_(oa, ob)].sum())
        return self

    # ------------------------------------------------------------- qubit ops
    @cached_property
    def hamiltonian(self) -> PauliSum:
        return molecular_qubit_hamiltonian(self.h1, self.h2, self.e_core, self.mapping)

    @cached_property
    def hamiltonian_sparse(self):
        return self.hamiltonian.to_sparse()

    @cached_property
    def hf_occupation(self) -> np.ndarray:
        occ = np.zeros(self.n_qubits, dtype=np.uint8)
        occ[0:2 * self.n_alpha:2] = 1
        occ[1:2 * self.n_beta:2] = 1
        return occ

    @cached_property
    def hf_index(self) -> int:
        return self.mapping.encode(self.hf_occupation)

    @cached_property
    def occupations(self) -> np.ndarray:
        """Occupation numbers of every computational basis state (2^n x n_modes)."""
        return self.mapping.occupations(np.arange(1 << self.n_qubits))

    @cached_property
    def number_diagonal(self) -> np.ndarray:
        return self.occupations.sum(axis=1).astype(float)

    @cached_property
    def sz_diagonal(self) -> np.ndarray:
        occ = self.occupations
        return 0.5 * (occ[:, 0::2].sum(axis=1) - occ[:, 1::2].sum(axis=1))

    @cached_property
    def sector_indices(self) -> np.ndarray:
        """Basis states with the target N_alpha, N_beta."""
        occ = self.occupations
        return np.flatnonzero((occ[:, 0::2].sum(axis=1) == self.n_alpha)
                              & (occ[:, 1::2].sum(axis=1) == self.n_beta))

    @cached_property
    def s2_sparse(self):
        return s2_operator(self.mapping).to_sparse()

    def exact_ground_state(self) -> tuple[float, np.ndarray]:
        """Lowest eigenpair of H within the (N_alpha, N_beta) sector."""
        idx = self.sector_indices
        hs = self.hamiltonian_sparse[idx][:, idx]
        if idx.size <= 2000:
            w, v = np.linalg.eigh(hs.toarray())
            e, vec = w[0], v[:, 0]
        else:
            w, v = spla.eigsh(hs, k=1, which="SA", tol=1e-12)
            e, vec = w[0], v[:, 0]
        psi = np.zeros(1 << self.n_qubits, dtype=np.complex128)
        psi[idx] = vec
        return float(np.real(e)), psi

    def energy(self, psi: np.ndarray) -> float:
        return float(np.real(np.vdot(psi, self.hamiltonian_sparse @ psi)))

    def spin_squared(self, psi: np.ndarray) -> float:
        return float(np.real(np.vdot(psi, self.s2_sparse @ psi)))

    def summary(self) -> str:
        return (f"{self.name}: basis={self.basis}, mapping={self.mapping.name}, "
                f"active=({self.n_electrons}e,{self.n_orbitals}o), frozen core={self.n_frozen}, "
                f"qubits={self.n_qubits}, Pauli terms={len(self.hamiltonian)}\n"
                f"  E_HF  = {self.e_hf:.10f} Ha\n  E_FCI = {self.e_fci:.10f} Ha (active space)"
                + (f"\n  {self.active_choice.describe()}" if self.active_choice else "")
                + "".join(f"\n  WARNING: {w}" for w in (self.active_choice.warnings if self.active_choice else [])))
