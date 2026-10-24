"""Command line:  python -m VQE --molecule LiH --ansatz uccsd --mapping jw"""
from __future__ import annotations

import argparse
import json
import sys

from qcore import MolecularProblem, list_molecules

from .adapt import AdaptVQE
from .ansatz import ANSATZE, get_ansatz
from .vqe import VQE


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m VQE", description="CPU statevector VQE / ADAPT-VQE")
    ap.add_argument("--molecule", default="H2", help=f"one of: {', '.join(list_molecules())}")
    ap.add_argument("--bond-length", type=float, default=None, help="Angstrom (default: experimental r_e)")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--mapping", default="jw", choices=["jw", "bk", "parity"])
    ap.add_argument("--active", default="default",
                    help='"default", "full", or "NELEC,NORB" (e.g. 4,4)')
    ap.add_argument("--ansatz", default="uccsd", help=f"{sorted(ANSATZE)} or 'adapt'")
    ap.add_argument("--pool", default="fermionic", help="ADAPT pool: fermionic|qeb|qubit|generalized")
    ap.add_argument("--layers", type=int, default=None, help="HEA layers / k for k-UpCCGSD")
    ap.add_argument("--reps", type=int, default=None, help="UCC repetitions")
    ap.add_argument("--optimizer", default="BFGS")
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--number-penalty", type=float, default=0.0)
    ap.add_argument("--spin-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print a JSON summary")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    active = a.active if a.active in ("default", "full") else tuple(int(v) for v in a.active.split(","))
    prob = MolecularProblem.from_library(a.molecule, bond_length=a.bond_length, basis=a.basis,
                                         mapping=a.mapping, active_space=active)
    print(prob.summary())

    if a.ansatz.lower() == "adapt":
        res = AdaptVQE(prob, pool=a.pool, optimizer=a.optimizer, verbose=a.verbose).run()
        name = f"ADAPT-VQE[{a.pool}]"
    else:
        kw = {}
        if a.layers is not None:
            kw["k" if a.ansatz.lower() in ("kupccgsd", "upccgsd") else "layers"] = a.layers
        if a.reps is not None:
            kw["reps"] = a.reps
        res = VQE(prob, get_ansatz(a.ansatz, **kw), optimizer=a.optimizer, maxiter=a.maxiter,
                  number_penalty=a.number_penalty, spin_penalty=a.spin_penalty, seed=a.seed,
                  verbose=a.verbose).run()
        name = a.ansatz
    print(f"[{name}] {res}")
    if a.json:
        print(json.dumps({"molecule": prob.name, "ansatz": name, "mapping": a.mapping, "basis": a.basis,
                          "n_qubits": prob.n_qubits, "energy": res.energy, "e_hf": res.e_hf,
                          "e_fci": res.e_fci, "error": res.error, "circuit": res.circuit,
                          "iterations": res.n_iterations, "wall_time": res.wall_time}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(f"error: {exc}")
