"""Benchmark VQE ansatze (and optionally GQE) across the molecule library.

    python examples/benchmark.py                      # UCCSD on all molecules
    python examples/benchmark.py --ansatz uccsd adapt kupccgsd --molecules H2 LiH H2O
    python examples/benchmark.py --ansatz adapt-qeb adapt-ceo adapt-ceo-tetris   # ADAPT pools / TETRIS
    python examples/benchmark.py --gqe --molecules H2 LiH
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcore import CHEMICAL_ACCURACY, MolecularProblem, list_molecules  # noqa: E402
from VQE import VQE, AdaptVQE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--molecules", nargs="+", default=[m for m in list_molecules() if m != "H6"])
ap.add_argument("--ansatz", nargs="+", default=["uccsd"])
ap.add_argument("--mapping", default="jw")
ap.add_argument("--gqe", action="store_true")
ap.add_argument("--csv", default=None)
a = ap.parse_args()

rows = []
print(f"{'molecule':<16}{'qubits':>7}{'method':>18}{'E (Ha)':>17}{'E_FCI (Ha)':>17}{'error':>11}"
      f"{'params':>8}{'chem.acc':>9}{'time':>8}")
for mol in a.molecules:
    t = time.perf_counter()
    p = MolecularProblem.from_library(mol, mapping=a.mapping)
    _ = p.hamiltonian_sparse
    methods = list(a.ansatz) + (["gqe"] if a.gqe else [])
    for m in methods:
        if m.startswith("adapt"):  # adapt[-POOL][-tetris]
            parts = m.split("-")[1:]
            tetris = "tetris" in parts
            pool = next((x for x in parts if x != "tetris"), "fermionic")
            r = AdaptVQE(p, pool=pool, tetris=tetris).run()
            e, n = r.energy, r.circuit["n_params"]
        elif m == "gqe":
            from GQE import GQE
            r = GQE(p).run()
            e, n = r.refined_energy, len(r.sequence)
        else:
            r = VQE(p, m).run()
            e, n = r.energy, r.circuit["n_params"]
        err = e - p.e_fci
        dt = time.perf_counter() - t
        t = time.perf_counter()
        rows.append(dict(molecule=p.name, qubits=p.n_qubits, method=m, energy=e, e_fci=p.e_fci,
                         e_hf=p.e_hf, error=err, params=n, time_s=dt))
        print(f"{p.name:<16}{p.n_qubits:>7}{m:>18}{e:>17.10f}{p.e_fci:>17.10f}{err:>11.2e}{n:>8}"
              f"{'yes' if abs(err) < CHEMICAL_ACCURACY else 'NO':>9}{dt:>7.1f}s", flush=True)

if a.csv:
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
