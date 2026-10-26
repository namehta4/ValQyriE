"""Command line:  python -m GQE --molecule LiH --iterations 300"""
from __future__ import annotations

import argparse
import json
import sys

from qcore import MolecularProblem, active_space, list_molecules

from .gqe import GQE, GQEConfig


def main(argv=None):
    d = GQEConfig()
    ap = argparse.ArgumentParser(prog="python -m GQE", description="CPU Generative Quantum Eigensolver (GPT-QE)")
    ap.add_argument("--molecule", default="H2", help=f"one of: {', '.join(list_molecules())}")
    ap.add_argument("--bond-length", type=float, default=None)
    ap.add_argument("--basis", default="sto-3g")
    ap.add_argument("--mapping", default="jw", choices=["jw", "bk", "parity"])
    active_space.add_cli_arguments(ap)
    ap.add_argument("--pool", default=d.pool, choices=["uccsd", "generalized", "qeb", "qubit"])
    ap.add_argument("--no-screen", action="store_true", help="keep symmetry-forbidden doubles in the pool")
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--iterations", type=int, default=d.iterations)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--beta-start", type=float, default=d.beta_start)
    ap.add_argument("--beta-end", type=float, default=d.beta_end)
    ap.add_argument("--energy-scale", type=float, default=d.energy_scale)
    ap.add_argument("--loss", default=d.loss, choices=["exp", "mse"])
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--no-refine", action="store_true")
    ap.add_argument("--seed", type=int, default=d.seed)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    active = active_space.from_cli(a)
    prob = MolecularProblem.from_library(a.molecule, bond_length=a.bond_length, basis=a.basis,
                                         mapping=a.mapping, active_space=active)
    print(prob.summary())
    cfg = GQEConfig(pool=a.pool, screen_tol=None if a.no_screen else d.screen_tol, seq_len=a.seq_len,
                    iterations=a.iterations, batch_size=a.batch_size, beta_start=a.beta_start,
                    beta_end=a.beta_end, energy_scale=a.energy_scale, loss=a.loss, lr=a.lr,
                    refine=not a.no_refine, seed=a.seed, threads=a.threads, verbose=a.verbose)
    gqe = GQE(prob, cfg)
    print(f"pool: {len(gqe.pool.generators)} generators x {2 * len(cfg.times)} times = "
          f"{gqe.pool.vocab_size} tokens, sequence length {gqe.seq_len}")
    res = gqe.run()
    print(res)
    if a.verbose:
        print("best circuit:\n  " + "\n  ".join(res.operators))
    if a.json:
        print(json.dumps({"molecule": prob.name, "mapping": a.mapping, "n_qubits": prob.n_qubits,
                          "energy": res.energy, "refined_energy": res.refined_energy, "e_hf": res.e_hf,
                          "e_fci": res.e_fci, "sequence": res.sequence, "n_circuits": res.n_circuits,
                          "wall_time": res.wall_time}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except ValueError as exc:
        sys.exit(f"error: {exc}")
