"""Library of benchmark molecules with experimental equilibrium geometries.

Bond lengths in Angstrom (NIST CCCBDB experimental r_e unless noted).  Every
entry takes an optional ``bond_length`` so dissociation curves can be scanned.
Default active spaces keep the qubit count at <= 14 in STO-3G so that VQE/GQE
run comfortably on a laptop CPU; pass ``active_space="full"`` or explicit
``active_electrons`` / ``active_orbitals`` to change them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MoleculeSpec:
    name: str
    geometry: Callable[[float], list]
    r_eq: float
    charge: int = 0
    spin: int = 0  # 2S = N_alpha - N_beta
    active: tuple[int, int] | None = None  # (electrons, spatial orbitals); None = full
    note: str = ""
    aliases: tuple[str, ...] = field(default_factory=tuple)


def _diatomic(a: str, b: str):
    return lambda r: [(a, (0.0, 0.0, 0.0)), (b, (0.0, 0.0, r))]


def _linear_chain(n: int):
    return lambda r: [("H", (0.0, 0.0, i * r)) for i in range(n)]


def _ring(n: int):
    def geo(r):
        rad = r / (2 * np.sin(np.pi / n))
        return [("H", (rad * np.cos(2 * np.pi * k / n), rad * np.sin(2 * np.pi * k / n), 0.0))
                for k in range(n)]
    return geo


def _bent_xh2(x: str, angle_deg: float):
    def geo(r):
        h = np.deg2rad(angle_deg) / 2
        return [(x, (0.0, 0.0, 0.0)),
                ("H", (r * np.sin(h), 0.0, r * np.cos(h))),
                ("H", (-r * np.sin(h), 0.0, r * np.cos(h)))]
    return geo


def _linear_xh2(x: str):
    return lambda r: [(x, (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, r)), ("H", (0.0, 0.0, -r))]


def _pyramidal_xh3(x: str, angle_deg: float):
    def geo(r):
        # H atoms on a cone around z with H-X-H angle = angle_deg
        a = np.deg2rad(angle_deg)
        cos_t = np.sqrt((1 + 2 * np.cos(a)) / 3)  # angle between bond and C3 axis
        sin_t = np.sqrt(1 - cos_t ** 2)
        out = [(x, (0.0, 0.0, 0.0))]
        for k in range(3):
            phi = 2 * np.pi * k / 3
            out.append(("H", (r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), -r * cos_t)))
        return out
    return geo


def _tetrahedral_xh4(x: str):
    def geo(r):
        s = r / np.sqrt(3)
        return [(x, (0.0, 0.0, 0.0)), ("H", (s, s, s)), ("H", (-s, -s, s)),
                ("H", (-s, s, -s)), ("H", (s, -s, -s))]
    return geo


LIBRARY: dict[str, MoleculeSpec] = {s.name.lower(): s for s in [
    MoleculeSpec("H2", _diatomic("H", "H"), 0.7414),
    MoleculeSpec("HeH+", _diatomic("He", "H"), 0.7743, charge=1, note="r_e from ab initio (Kolos & Peek)"),
    MoleculeSpec("LiH", _diatomic("Li", "H"), 1.5949),
    MoleculeSpec("BeH2", _linear_xh2("Be"), 1.3264, active=(4, 6), note="Be 1s frozen"),
    MoleculeSpec("H2O", _bent_xh2("O", 104.48), 0.9578, active=(8, 6),
                 note="O 1s frozen, highest virtual dropped"),
    MoleculeSpec("NH3", _pyramidal_xh3("N", 106.67), 1.0124, active=(8, 7), note="N 1s frozen"),
    MoleculeSpec("CH4", _tetrahedral_xh4("C"), 1.0870, active=(8, 7),
                 note="C 1s frozen, highest virtual dropped"),
    MoleculeSpec("HF", _diatomic("F", "H"), 0.9168, active=(8, 5), note="F 1s frozen"),
    MoleculeSpec("N2", _diatomic("N", "N"), 1.0977, active=(6, 6), note="2p valence space"),
    MoleculeSpec("CO", _diatomic("C", "O"), 1.1283, active=(6, 6)),
    MoleculeSpec("O2", _diatomic("O", "O"), 1.2075, spin=2, active=(8, 6),
                 note="triplet ground state (X 3Sigma_g^-)"),
    MoleculeSpec("H4", _linear_chain(4), 1.0, note="linear chain, r = H-H spacing"),
    MoleculeSpec("H6", _linear_chain(6), 1.0, note="linear chain, r = H-H spacing"),
    MoleculeSpec("H4-ring", _ring(4), 1.0, note="square H4, r = side length"),
]}


def get_molecule(name: str) -> MoleculeSpec:
    try:
        return LIBRARY[name.lower()]
    except KeyError:
        raise ValueError(f"unknown molecule {name!r}; available: "
                         f"{', '.join(s.name for s in LIBRARY.values())}") from None


def list_molecules() -> list[str]:
    return [s.name for s in LIBRARY.values()]
