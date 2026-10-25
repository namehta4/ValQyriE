"""Automatic active-space selection from MP2 orbital occupations (opt-in, never the default).

MP2 gives every Hartree-Fock orbital an approximate occupation: the diagonal of the MP2
one-particle density in the canonical HF orbitals. Orbitals that stay close to 2 (filled) or
0 (empty) behave like Hartree-Fock and can be left out; orbitals whose occupation moves away
from 2 or 0 by more than ``threshold`` carry the electron correlation and form the active
space. Singly occupied orbitals (open shells) are always active. When more orbitals pass the
threshold than ``max_qubits`` allows, the most fractional ones are kept. Degenerate orbitals
(e.g. a pair of pi orbitals) are kept or dropped together so the result does not depend on an
arbitrary rotation between them.

The orbitals stay the canonical HF orbitals, so the Hartree-Fock determinant remains the
reference state of the qubit problem. MP2 is a perturbative estimate: it is unreliable when the
occupations are strongly fractional (stretched bonds, near-degenerate states), and the selection
then comes with a warning.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEGENERACY_TOL = 1e-4  # Hartree, orbital-energy difference treated as degenerate
STRONG_DEVIATION = 0.1  # occupation this far from 2 or 0 -> MP2 likely unreliable


@dataclass(frozen=True)
class AutoActiveSpace:
    """Settings for the MP2-occupation selection."""
    threshold: float = 0.01
    max_qubits: int = 16

    def __post_init__(self):
        if not 0 < self.threshold < 1:
            raise ValueError(f"auto active space: threshold must be between 0 and 1, got {self.threshold}")
        if self.max_qubits < 4:
            raise ValueError(f"auto active space: max_qubits must be at least 4, got {self.max_qubits}")


@dataclass
class ActiveSpaceChoice:
    orbitals: list[int]          # 0-based canonical MO indices
    n_electrons: int
    occupations: np.ndarray      # MP2 occupation of every MO
    settings: AutoActiveSpace
    n_above_threshold: int       # orbitals that passed the threshold (before the max_qubits cap)
    warnings: list[str] = field(default_factory=list)

    def describe(self) -> str:
        occ = ", ".join(f"{i}:{self.occupations[i]:.3f}" for i in self.orbitals)
        return (f"auto active space ({self.n_electrons}e, {len(self.orbitals)}o) from MP2 occupations "
                f"[threshold {self.settings.threshold}, max {self.settings.max_qubits} qubits]: "
                f"orbitals {{{occ}}}")


def parse_active_space(value):
    """Normalize the user-facing forms: "default", "full", "auto", [n_e, n_o], an
    AutoActiveSpace, or {"method": "auto", "threshold": ..., "max_qubits": ...}."""
    if isinstance(value, AutoActiveSpace) or value in ("default", "full"):
        return value
    if value == "auto":
        return AutoActiveSpace()
    if isinstance(value, dict):
        opts = dict(value)
        if opts.pop("method", None) != "auto":
            raise ValueError('an active_space object needs "method": "auto"')
        unknown = set(opts) - {"threshold", "max_qubits"}
        if unknown:
            raise ValueError(f"unknown auto active-space key(s) {sorted(unknown)}; allowed: threshold, max_qubits")
        return AutoActiveSpace(**opts)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return tuple(int(v) for v in value)
    raise ValueError(f'active_space must be "default", "full", "auto", [n_electrons, n_orbitals] or '
                     f'{{"method": "auto", ...}}, got {value!r}')


def add_cli_arguments(ap):
    """--active / --auto-threshold / --auto-max-qubits, shared by the VQE and GQE command lines."""
    ap.add_argument("--active", default="default",
                    help='"default" (library choice), "full", "NELEC,NORB" (e.g. 4,4), or "auto" '
                         "(opt-in: pick orbitals from MP2 occupations)")
    ap.add_argument("--auto-threshold", type=float, default=AutoActiveSpace.threshold,
                    help="--active auto: keep orbitals whose occupation is this far from 2 or 0")
    ap.add_argument("--auto-max-qubits", type=int, default=AutoActiveSpace.max_qubits,
                    help="--active auto: upper limit on the qubit count")


def from_cli(args):
    if args.active == "auto":
        return AutoActiveSpace(threshold=args.auto_threshold, max_qubits=args.auto_max_qubits)
    if args.active in ("default", "full"):
        return args.active
    try:
        return tuple(int(v) for v in args.active.split(","))
    except ValueError:
        raise ValueError(f'--active must be "default", "full", "auto" or "NELEC,NORB", got {args.active!r}') from None


def mp2_occupations(mf) -> np.ndarray:
    """Diagonal of the MP2 one-particle density in the canonical HF orbitals (RMP2 for RHF,
    UMP2 for ROHF; alpha + beta)."""
    from pyscf import mp
    pt = mp.MP2(mf)
    pt.verbose = 0
    pt.kernel()
    dm = pt.make_rdm1()
    if isinstance(dm, (tuple, list)) or np.ndim(dm) == 3:
        dm = dm[0] + dm[1]
    return np.real(np.diag(np.asarray(dm)))


def _groups(mo_energy, mo_occ):
    """Runs of consecutive orbitals with the same HF occupation and (near-)equal energy."""
    out = []
    for i in range(len(mo_energy)):
        if out and mo_occ[i] == mo_occ[out[-1][-1]] and abs(mo_energy[i] - mo_energy[out[-1][-1]]) < DEGENERACY_TOL:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def select_active_space(mf, settings: AutoActiveSpace = AutoActiveSpace()) -> ActiveSpaceChoice:
    mo_occ = np.rint(np.asarray(mf.mo_occ)).astype(int)
    mo_energy = np.asarray(mf.mo_energy)
    occ = mp2_occupations(mf)
    max_orb = settings.max_qubits // 2
    dev = np.where(mo_occ == 2, 2.0 - occ, occ)

    groups = _groups(mo_energy, mo_occ)
    somo = [i for i in range(len(mo_occ)) if mo_occ[i] == 1]
    if len(somo) > max_orb:
        raise ValueError(f"auto active space: {len(somo)} singly occupied orbitals need more than "
                         f"max_qubits={settings.max_qubits}")
    chosen = list(somo)
    closed = [g for g in groups if mo_occ[g[0]] != 1]
    score = {tuple(g): float(np.mean(dev[g])) for g in closed}

    def fits(g):
        return len(chosen) + len(g) <= max_orb

    # always correlate at least the most fractional filled and empty orbital groups
    for kind in (2, 0):
        best = max((g for g in closed if mo_occ[g[0]] == kind), key=lambda g: score[tuple(g)], default=None)
        if best is not None and fits(best):
            chosen += best
    above = [g for g in closed if score[tuple(g)] > settings.threshold]
    n_above = len(somo) + sum(len(g) for g in above)
    dropped = 0
    for g in sorted(above, key=lambda g: -score[tuple(g)]):
        if g[0] in chosen:
            continue
        if fits(g):
            chosen += g
        else:
            dropped += len(g)

    orbitals = sorted(chosen)
    n_electrons = int(sum(mo_occ[orbitals]))
    if n_electrons == 0 or n_electrons == 2 * len(orbitals):
        raise ValueError("auto active space: selection has no empty or no filled orbitals; "
                         "raise max_qubits or choose the active space by hand")
    warnings = []
    if dropped:
        warnings.append(f"{dropped} more orbital(s) passed the threshold but did not fit in "
                        f"max_qubits={settings.max_qubits}")
    if np.any(occ < -1e-8) or np.any(occ > 2 + 1e-8):
        warnings.append("MP2 occupations outside [0, 2]: MP2 has broken down for this molecule; "
                        "choose the active space by hand")
    closed_dev = dev[mo_occ != 1]
    if closed_dev.size and closed_dev.max() > STRONG_DEVIATION:
        warnings.append(f"strongly fractional MP2 occupations (up to {closed_dev.max():.2f} away from 2 or 0), "
                        "typical of stretched bonds; MP2 is unreliable here, so compare with a manual "
                        "active space")
    return ActiveSpaceChoice(orbitals, n_electrons, occ, settings, n_above, warnings)
