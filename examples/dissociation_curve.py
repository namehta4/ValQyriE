"""Potential-energy curve: HF, FCI, VQE (chosen ansatz) and optionally GQE.

    python examples/dissociation_curve.py --molecule H2 --ansatz uccsd
    python examples/dissociation_curve.py --molecule LiH --points 1.0 1.6 2.4 3.2 --gqe
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from qcore import MolecularProblem  # noqa: E402
from VQE import VQE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--molecule", default="H2")
ap.add_argument("--ansatz", default="uccsd")
ap.add_argument("--points", nargs="+", type=float, default=list(np.round(np.linspace(0.5, 2.5, 9), 3)))
ap.add_argument("--gqe", action="store_true")
a = ap.parse_args()

print(f"{'r (A)':>7}{'E_HF':>16}{'E_FCI':>16}{'E_VQE':>16}{'VQE err':>11}" + (f"{'GQE err':>11}" if a.gqe else ""))
prev = None
for r in a.points:
    p = MolecularProblem.from_library(a.molecule, bond_length=r)
    res = VQE(p, a.ansatz).run(initial_parameters=prev)
    prev = res.parameters  # warm start along the curve
    line = f"{r:>7.3f}{p.e_hf:>16.8f}{p.e_fci:>16.8f}{res.energy:>16.8f}{res.error:>11.2e}"
    if a.gqe:
        from GQE import GQE
        line += f"{GQE(p).run().refined_error:>11.2e}"
    print(line, flush=True)
