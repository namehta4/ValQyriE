"""Correctness gate: a result counts only if it reproduces the stored reference answers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ENERGY_TOL = 1e-10    # Ha
GRADIENT_TOL = 1e-8   # Ha per radian, max over components
VQE_ENERGY_TOL = 1e-7  # final energy of a full optimization (paths differ by rounding)

REFERENCE_DIR = Path(__file__).parent / "references"


def load_reference(case_name: str, path=None) -> dict | None:
    p = Path(path) if path else REFERENCE_DIR / f"{case_name}.json"
    if not p.exists():
        return None
    ref = json.loads(p.read_text())
    for pt in ref["points"]:
        pt["theta"] = np.asarray(pt["theta"], float)
        pt["gradient"] = np.asarray(pt["gradient"], float)
    return ref


def check_point(energy, gradient, point, e_fci=None) -> dict:
    gradient = np.asarray(gradient, float)
    out = {"energy_error": abs(float(energy) - point["energy"]),
           "gradient_error": float(np.max(np.abs(gradient - point["gradient"]))) if gradient.shape == point["gradient"].shape else float("inf"),
           "finite": bool(np.isfinite(energy) and np.all(np.isfinite(gradient)))}
    reasons = []
    if not out["finite"]:
        reasons.append("non-finite energy or gradient")
    if out["energy_error"] > ENERGY_TOL:
        reasons.append(f"energy off by {out['energy_error']:.2e} Ha (limit {ENERGY_TOL:.0e})")
    if out["gradient_error"] > GRADIENT_TOL:
        reasons.append(f"gradient off by {out['gradient_error']:.2e} (limit {GRADIENT_TOL:.0e})")
    if e_fci is not None and np.isfinite(e_fci) and energy < e_fci - ENERGY_TOL:
        reasons.append(f"energy {energy:.12f} below the exact ground state {e_fci:.12f}")
    out["passed"] = not reasons
    out["reasons"] = reasons
    return out
