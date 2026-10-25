"""Command line:  python -m VQE --molecule LiH --ansatz uccsd --mapping jw"""
from __future__ import annotations

import argparse
import json
import sys

from qcore import MolecularProblem, active_space, list_molecules
from qcore.excitations import SCREEN_TOL

from .adapt import AdaptVQE
from .orbital_opt import OrbitalOptimizedVQE
from .ansatz import ANSATZE, get_ansatz
from .vqe import VQE


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m VQE", description="CPU statevector VQE / ADAPT-VQE")
    ap.add_argument("--molecule", default="H2", help=f"one of: {', '.join(list_molecules())}")
    ap.add_argument("--bond-length", type=float, default=None, help="Angstrom (default: experimental r_e)")
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--mapping", default="jw", choices=["jw", "bk", "parity"])
    active_space.add_cli_arguments(ap)
    ap.add_argument("--ansatz", default="uccsd", help=f"{sorted(ANSATZE)} or 'adapt'")
    ap.add_argument("--pool", default="fermionic", help="ADAPT pool: fermionic|qeb|qubit|generalized|ceo")
    ap.add_argument("--tetris", action="store_true",
                    help="ADAPT: add every operator on disjoint qubits per iteration (TETRIS-ADAPT)")
    ap.add_argument("--layers", type=int, default=None, help="layers of hea / fabric / lucj, k for k-UpCCGSD")
    ap.add_argument("--interactions", default=None, choices=["square", "heavy-hex", "all"],
                    help="lucj: which density-density terms (default square)")
    ap.add_argument("--orbital-opt", action="store_true",
                    help="optimize the orbitals together with the circuit (oo-VQE, reference: CASSCF)")
    ap.add_argument("--initial-orbitals", default=None, choices=["hf", "boys", "pm"],
                    help="--orbital-opt starting orbitals (default boys for pair ansatze, else hf)")
    ap.add_argument("--reps", type=int, default=None, help="UCC repetitions")
    ap.add_argument("--screen", action="store_true",
                    help="drop symmetry-forbidden doubles (uccsd, uccd, quccsd)")
    ap.add_argument("--optimizer", default="BFGS")
    ap.add_argument("--maxiter", type=int, default=1000)
    ap.add_argument("--number-penalty", type=float, default=0.0)
    ap.add_argument("--spin-penalty", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", action="store_true", help="print a JSON summary")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    active = active_space.from_cli(a)
    prob = MolecularProblem.from_library(a.molecule, bond_length=a.bond_length, basis=a.basis,
                                         mapping=a.mapping, active_space=active)
    print(prob.summary())

    if a.ansatz.lower() == "adapt":
        res = AdaptVQE(prob, pool=a.pool, optimizer=a.optimizer, tetris=a.tetris, verbose=a.verbose).run()
        name = f"{'TETRIS-' if a.tetris else ''}ADAPT-VQE[{a.pool}]"
    else:
        kw = {}
        if a.layers is not None:
            kw["k" if a.ansatz.lower() in ("kupccgsd", "upccgsd") else "layers"] = a.layers
        if a.reps is not None:
            kw["reps"] = a.reps
        if a.interactions is not None:
            kw["interactions"] = a.interactions
        if a.screen:
            if a.ansatz.lower() not in ("uccsd", "uccd", "quccsd"):
                raise ValueError(f"--screen applies to uccsd, uccd and quccsd, not {a.ansatz}")
            kw["screen_tol"] = SCREEN_TOL
        ansatz = get_ansatz(a.ansatz, **kw)
        if a.orbital_opt:
            res = OrbitalOptimizedVQE(prob, ansatz, initial_orbitals=a.initial_orbitals, optimizer=a.optimizer,
                                      seed=a.seed, verbose=a.verbose).run()
            print(f"[oo-{a.ansatz}] {res}")
            if a.json:
                print(json.dumps({"molecule": prob.name, "ansatz": f"oo-{a.ansatz}", "mapping": a.mapping,
                                  "basis": a.basis, "n_qubits": prob.n_qubits, "energy": res.energy,
                                  "e_casscf": res.e_casscf, "error_vs_casscf": res.error, "e_fci": res.e_fci,
                                  "e_hf": res.e_hf, "circuit": res.vqe.circuit,
                                  "macro_iterations": res.macro_iterations, "wall_time": res.wall_time}, indent=2))
            return
        res = VQE(prob, ansatz, optimizer=a.optimizer, maxiter=a.maxiter,
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
