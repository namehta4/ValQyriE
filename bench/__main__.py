"""Benchmark harness for the SCC performance challenge.

    python -m bench list                                   # the cases
    python -m bench export h2o-24q -o cases/                # case files for non-Python codes
    python -m bench run h2o-16q                             # reference code, correctness gate + timing
    python -m bench run h2o-24q --impl mycode.py:MyEvaluator -o result.json
    mpirun -np 8 python -m bench run anthracene-28q --impl my.mod:Dist --mpi
    python -m bench vqe h2o-24q --impl mycode.py:MyEvaluator   # full UCCSD optimization (stage 6)

An implementation provides ``energy_and_gradient(theta)``; see ``bench.interface``. Every run
checks the energy and gradient at the reference parameter points before any time is reported.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import gate
from .cases import CASES, CaseData, build_case
from .interface import load_impl


class Comm:
    def __init__(self, use_mpi: bool):
        self.comm = None
        if use_mpi:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank() if self.comm else 0
        self.size = self.comm.Get_size() if self.comm else 1

    def barrier(self):
        if self.comm:
            self.comm.Barrier()


def _log(comm):
    def log(*a, **k):
        if comm.rank == 0:
            print(*a, **k, flush=True)
    return log


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, timeout=10).stdout.strip() or None
    except Exception:
        return None


def _case(args) -> CaseData:
    if args.case_file:
        return CaseData.load(args.case_file)
    return build_case(args.case)


def _system():
    try:
        from stress.sysinfo import collect
        si = collect()
        return {"host": si.hostname, "cpu": si.cpu_model, "cores": si.physical_cores, "arch": si.arch,
                "memory_bytes": si.memory_bytes}
    except Exception:
        return {"host": platform.node(), "arch": platform.machine()}


def _make(args, case, comm, log):
    cls = load_impl(args.impl)
    comm.barrier()
    t0 = time.perf_counter()
    impl = cls(case, comm.comm)
    comm.barrier()
    setup = time.perf_counter() - t0
    log(f"implementation  {args.impl or 'reference (ValQyriE, full statevector)'}  setup {setup:.2f} s")
    return impl, setup


def cmd_list(args):
    for name, spec in CASES.items():
        print(f"{name:10s} {spec['role']:26s} {spec['description']}")
    return 0


def cmd_export(args):
    case = build_case(args.case, from_pyscf=args.from_pyscf)
    path = case.save(args.output)
    print(f"wrote {path} and {path.with_suffix('.json')}  ({case.n_qubits} qubits, {case.n_params} parameters)")
    return 0


def cmd_run(args):
    comm = Comm(args.mpi)
    log = _log(comm)
    case = _case(args)
    ref = gate.load_reference(case.name, args.reference)
    if ref is None:
        sys.exit(f"error: no reference answers for {case.name} (bench/references/{case.name}.json); "
                 "judges supply them with --reference for the hidden case")
    log(f"case            {case.name}: {case.description}  ({case.n_qubits} qubits, {case.n_params} parameters)")
    impl, setup = _make(args, case, comm, log)
    points = ref["points"][: args.points] if args.points else ref["points"]
    results, times, passed = [], [], True
    for i, pt in enumerate(points):
        comm.barrier()
        t0 = time.perf_counter()
        e, g = impl.energy_and_gradient(pt["theta"].copy())
        comm.barrier()
        dt = time.perf_counter() - t0
        chk = gate.check_point(e, g, pt, ref.get("e_fci"))
        passed &= chk["passed"]
        times.append(dt)
        results.append({"point": i, "time_s": dt, **chk})
        log(f"point {i}         {dt:10.3f} s   dE = {chk['energy_error']:.1e}   dg = {chk['gradient_error']:.1e}   "
            f"{'PASS' if chk['passed'] else 'FAIL: ' + '; '.join(chk['reasons'])}")
    timed = []
    for _ in range(args.repeats):
        comm.barrier()
        t0 = time.perf_counter()
        impl.energy_and_gradient(points[-1]["theta"].copy())
        comm.barrier()
        timed.append(time.perf_counter() - t0)
    all_t = times + timed
    summary = {"case": case.name, "n_qubits": case.n_qubits, "n_params": case.n_params,
               "impl": args.impl or "reference", "ranks": comm.size, "setup_s": setup,
               "eval_s_min": min(all_t), "eval_s_median": float(np.median(all_t)), "evaluations": len(all_t),
               "gate": "PASS" if passed else "FAIL", "points": results, "system": _system(),
               "git_commit": _git_commit(), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    log(f"\nenergy+gradient {summary['eval_s_min']:.3f} s (best of {len(all_t)}), median {summary['eval_s_median']:.3f} s")
    log(f"correctness     {summary['gate']}" + ("" if passed else "  -- times do not count until the gate passes"))
    if args.output and comm.rank == 0:
        Path(args.output).write_text(json.dumps(summary, indent=1, default=str))
        log(f"report written to {args.output}")
    return 0 if passed else 1


def cmd_vqe(args):
    """Full UCCSD optimization with fixed settings: BFGS from theta = 0, tol 1e-10, maxiter 1000
    (the settings of ValQyriE's VQE). Every rank runs the same deterministic optimizer."""
    from scipy.optimize import minimize
    comm = Comm(args.mpi)
    log = _log(comm)
    case = _case(args)
    ref = gate.load_reference(case.name, args.reference)
    impl, setup = _make(args, case, comm, log)
    nfev = [0]

    def fun(x):
        nfev[0] += 1
        e, g = impl.energy_and_gradient(x)
        return e, np.asarray(g, float)

    comm.barrier()
    t0 = time.perf_counter()
    res = minimize(fun, np.zeros(case.n_params), jac=True, method="BFGS", tol=1e-10,
                   options={"maxiter": args.maxiter})
    comm.barrier()
    wall = time.perf_counter() - t0
    e = float(res.fun)
    out = {"case": case.name, "impl": args.impl or "reference", "ranks": comm.size, "setup_s": setup,
           "time_to_solution_s": wall, "energy": e, "evaluations": nfev[0], "iterations": int(res.nit),
           "gradient_norm": float(np.linalg.norm(res.jac)), "e_fci": case.e_fci, "system": _system(),
           "git_commit": _git_commit()}
    ok = None
    if ref and ref.get("vqe"):
        out["reference_energy"] = ref["vqe"]["energy"]
        ok = abs(e - ref["vqe"]["energy"]) < gate.VQE_ENERGY_TOL
    out["gate"] = {None: "no reference", True: "PASS", False: "FAIL"}[ok]
    log(f"time to solution {wall:.2f} s  ({nfev[0]} evaluations, {res.nit} iterations)")
    log(f"energy           {e:.10f} Ha" + (f"   reference {ref['vqe']['energy']:.10f}" if ok is not None else ""))
    log(f"correctness      {out['gate']}")
    if args.output and comm.rank == 0:
        Path(args.output).write_text(json.dumps(out, indent=1, default=str))
    return 0 if ok in (None, True) else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m bench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the cases")
    ex = sub.add_parser("export", help="write a case as .npz + .json")
    ex.add_argument("case", choices=list(CASES))
    ex.add_argument("-o", "--output", default="cases")
    ex.add_argument("--from-pyscf", action="store_true",
                    help="rebuild with PySCF instead of copying bench/data (orbital signs may differ)")
    for name in ("run", "vqe"):
        r = sub.add_parser(name, help={"run": "correctness gate + timing of energy_and_gradient",
                                       "vqe": "full UCCSD optimization, time to solution"}[name])
        r.add_argument("case", nargs="?", choices=list(CASES))
        r.add_argument("--case-file", help="a case exported with `export` (e.g. the hidden case)")
        r.add_argument("--reference", help="reference answers (default bench/references/<case>.json)")
        r.add_argument("--impl", help="module:Class or path/file.py:Class (default: the reference code)")
        r.add_argument("--mpi", action="store_true", help="run under mpirun; comm is passed to the implementation")
        r.add_argument("-o", "--output", help="write a JSON report")
        if name == "run":
            r.add_argument("--points", type=int, default=0, help="check only the first N reference points")
            r.add_argument("--repeats", type=int, default=0, help="extra timed evaluations after the gate")
        else:
            r.add_argument("--maxiter", type=int, default=1000)
    args = ap.parse_args(argv)
    if args.cmd in ("run", "vqe") and not (args.case or args.case_file):
        ap.error("give a case name or --case-file")
    return {"list": cmd_list, "export": cmd_export, "run": cmd_run, "vqe": cmd_vqe}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
