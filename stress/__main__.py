"""CPU stress test built on the ValQyriE quantum-chemistry kernels.

    python -m stress info                      # hardware, ISA, caches, ceilings (no load)
    python -m stress peak                      # measure what the system can achieve at most
    python -m stress run --duration 300        # verified stress run on every physical core
    python -m stress run --mode mixed --duration 3600 --size dram -o burnin.json
    mpirun -np 4 --map-by node python -m stress run --mpi    # one rank per node (cluster)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time

from . import peaks, runner
from .sysinfo import INTERESTING_FLAGS, collect
from .telemetry import Telemetry
from .workloads import UNITS

ALL_WORKLOADS = ("dgemm", "statevector", "vqe", "transformer")


class Comm:
    """MPI wrapper; degenerates to a single rank without --mpi."""

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

    def gather(self, obj):
        return self.comm.gather(obj, root=0) if self.comm else [obj]


def log_fn(comm):
    def log(*a, **k):
        if comm.rank == 0:
            print(*a, **k, flush=True)
    return log


def select_cpus(si, workers: str):
    if workers in ("physical", "all"):
        return list(si.core_cpus)
    if workers == "logical":
        try:
            import os
            return sorted(os.sched_getaffinity(0))
        except AttributeError:
            return list(range(si.logical_cpus))
    n = int(workers)
    pool = list(si.core_cpus)
    return [pool[i % len(pool)] for i in range(n)]


def workload_params(name, size, si, n_workers, args):
    if name == "dgemm":
        return {"n": args.dgemm_n or {"cache": 256, "llc": 1024, "dram": 2048}.get(size, 1024)}
    if name == "statevector":
        if args.qubits:
            q = args.qubits
        else:
            l2, l3 = si.l2_bytes or 1 << 20, si.l3_bytes or 32 << 20
            mem = si.memory_bytes or 16 << 30
            dram_q = int(math.log2(0.5 * mem / (n_workers * peaks.BYTES_PER_AMP_CIRCUIT)))
            q = {"cache": int(math.log2(l2 / peaks.BYTES_PER_AMP_CIRCUIT)),
                 "llc": int(math.log2(l3 / (n_workers * peaks.BYTES_PER_AMP_CIRCUIT))),
                 "dram": min(26, dram_q)}.get(size, min(24, dram_q))
            q = max(10, min(q, dram_q))
        # keep a step around a second: fewer rotations per step for bigger states
        return {"n_qubits": q, "n_rotations": int(min(16, max(2, 16 * 2 ** (20 - q))))}
    if name == "vqe":
        return {"molecule": args.molecule or {"cache": "H4", "dram": "NH3"}.get(size, "H2O")}
    if name == "transformer":
        return {"n_embd": {"cache": 128, "dram": 512}.get(size, 256)}
    raise ValueError(name)


def available(name):
    if name == "transformer":
        try:
            import torch  # noqa: F401
        except ImportError:
            return False
    return True


def fmt_bytes(b):
    return "?" if not b else (f"{b / 2**30:.1f} GiB" if b >= 2**30 else f"{b / 2**20:.0f} MiB" if b >= 2**20 else f"{b / 2**10:.0f} KiB")


def print_info(si, theo, cap, log):
    log(f"host            {si.hostname}  ({si.os}, {si.arch})")
    log(f"cpu             {si.cpu_model or '?'}  [vendor {si.vendor or '?'}]")
    log(f"topology        {si.sockets or '?'} socket(s), {si.physical_cores} physical cores, "
        f"{si.logical_cpus} logical CPUs, {si.numa_nodes or '?'} NUMA node(s)")
    log(f"caches          L1d {fmt_bytes(si.l1d_bytes)}  L2 {fmt_bytes(si.l2_bytes)}  L3 {fmt_bytes(si.l3_bytes)}")
    log(f"memory          {fmt_bytes(si.memory_bytes)}")
    log(f"ISA flags       {' '.join(sorted(f for f in si.flags if f in INTERESTING_FLAGS)) or '?'}")
    log(f"runtime SIMD    numpy: {', '.join(si.simd.get('numpy', [])) or '?'};  "
        f"torch: {si.simd.get('torch_cpu_capability', 'n/a')}")
    f = f"{theo['freq_ghz']:.2f} GHz" if theo["freq_ghz"] else "? GHz"
    pk = f"{theo['peak_fp64_gflops']:.0f} GFLOP/s" if theo["peak_fp64_gflops"] else "unknown"
    log(f"FP64 peak (est) {pk}  = {theo['cores']} cores x {f} x {theo['flops_per_cycle_fp64'] or '?'} FLOP/cycle"
        f"  [{theo['flops_per_cycle_basis']}; clock: {theo['freq_basis']}]")
    if cap:
        log(f"max problem     statevector {cap['max_qubits_state_only']} qubits (state only), "
            f"{cap['max_qubits_circuit_sim']} (this simulator), VQE {cap['max_qubits_vqe_state']} "
            f"(state + gradient) / {cap['max_qubits_vqe_with_sparse_h']} (with sparse H)  "
            f"[{cap['usable_fraction']:.0%} of RAM]")


def print_peaks(m, theo, log):
    log(f"  DGEMM  BLAS multithreaded : {m['dgemm_blas']['gflops']:9.1f} GFLOP/s (n={m['dgemm_blas']['n']})")
    log(f"  DGEMM  1 core / all cores : {m['dgemm_1core_gflops']:9.1f} / {m['dgemm_allcore_gflops']:.1f} GFLOP/s")
    if theo["peak_fp64_gflops"]:
        pct = 100 * m['measured_peak_fp64_gflops'] / theo['peak_fp64_gflops']
        log(f"  -> {pct:.1f}% of estimated theoretical FP64 peak ({theo['peak_fp64_gflops']:.0f} GFLOP/s)")
        if pct > 100:
            log("     NOTE: above the vector-ISA estimate -- the BLAS uses a matrix unit (e.g. Apple AMX/SME)")
            log("     or the clock/FLOPs-per-cycle inputs are low; pass --freq-ghz / --flops-per-cycle.")
    if m.get("sgemm_torch"):
        log(f"  FP32 GEMM (PyTorch)       : {m['sgemm_torch']['gflops']:9.1f} GFLOP/s (n={m['sgemm_torch']['n']})")
    log(f"  Triad  1 core / all cores : {m['triad_1core_gbs']:9.1f} / {m['triad_allcore_gbs']:.1f} GB/s")
    if theo.get("mem_bw_gbs"):
        log(f"  -> {100 * m['triad_allcore_gbs'] / theo['mem_bw_gbs']:.1f}% of theoretical memory bandwidth")
    log(f"  verification failures     : {m['verification_failures']}")


def ceiling_fraction(name, rate, measured, theo):
    """Achieved throughput relative to the system ceilings."""
    if not measured:
        return {}
    if name == "dgemm":
        out = {"precision": "FP64", "pct_of_measured_peak": 100 * rate / measured["measured_peak_fp64_gflops"]}
        if theo.get("peak_fp64_gflops"):
            out["pct_of_theoretical_peak"] = 100 * rate / theo["peak_fp64_gflops"]
        return out
    if name == "transformer":  # PyTorch default dtype: FP32
        out = {"precision": "FP32"}
        if measured.get("sgemm_torch"):
            out["pct_of_measured_peak"] = 100 * rate / measured["sgemm_torch"]["gflops"]
        if theo.get("peak_fp32_gflops"):
            out["pct_of_theoretical_peak"] = 100 * rate / theo["peak_fp32_gflops"]
        return out
    if name == "statevector":
        # lower bound on traffic: each amplitude update reads and writes 16 bytes
        gbs = 32 * rate
        return {"min_traffic_gbs": gbs, "pct_of_measured_bw": 100 * gbs / measured["triad_allcore_gbs"]}
    return {}


def merge_nodes(results):
    """Combine per-node aggregates (MPI) and check that all nodes agree bit for bit."""
    merged = {}
    for node in results:
        for name, a in node["aggregate"].items():
            m = merged.setdefault(name, {"steps": 0, "work": 0.0, "fails": 0, "rate_total": 0.0,
                                         "inconsistent": [], "fingerprints": {}})
            m["steps"] += a["steps"]
            m["work"] += a["work"]
            m["fails"] += a["fails"]
            m["rate_total"] += a["rate_total"]
            m["inconsistent"] += [f"{node['host']}:cpu{c}" for c in a["inconsistent_cpus"]]
            m["fingerprints"].setdefault(a["majority_fingerprint"], []).append(node["host"])
    for m in merged.values():
        fps = m.pop("fingerprints")
        majority = max(fps.values(), key=len)
        m["inconsistent_nodes"] = sorted(h for hosts in fps.values() if hosts is not majority for h in hosts)
    return merged


def cmd_run(args, comm, log):
    si = collect()
    cpus = select_cpus(si, args.workers)
    theo = peaks.theoretical(si, args.freq_ghz, args.flops_per_cycle, args.mem_bw_gbs)
    cap = peaks.capacity(si)
    print_info(si, theo, cap, log)
    names = [w for w in (args.workloads.split(",") if args.workloads else ALL_WORKLOADS)]
    skipped = [w for w in names if not available(w)]
    names = [w for w in names if w not in skipped]
    if skipped:
        log(f"skipping {', '.join(skipped)} (PyTorch not installed)")
    log(f"\nworkers         {len(cpus)} on CPUs {cpus if len(cpus) <= 16 else str(cpus[:16])[:-1] + ', ...]'}"
        f"{'' if sys.platform.startswith('linux') else '  (no CPU pinning on this OS)'}")
    if comm.size > 1:
        log(f"nodes           {comm.size} MPI ranks")

    measured = None
    if not args.no_peak:
        comm.barrier()
        log("\n== ceilings: measuring what this system can achieve ==")
        measured = peaks.measure(si, cpus, seconds=args.peak_seconds, log=log)
        print_peaks(measured, theo, log)

    specs = {}
    for w in names:
        p = workload_params(w, args.size, si, len(cpus), args)
        if w == "vqe":
            from .workloads import vqe_payload
            payload = vqe_payload(p["molecule"])
            log(f"vqe workload    {payload['molecule']}, {payload['n_qubits']} qubits, "
                f"{payload['circuit'].n_params} UCCSD parameters")
            p = payload
        elif w == "statevector":
            log(f"statevector     {p['n_qubits']} qubits ({fmt_bytes(16 << p['n_qubits'])} state per worker)")
        specs[w] = p
    tel = Telemetry(cpus)
    if tel.power_note:
        log(f"power           {tel.power_note}")

    def status(t, agg, sample):
        parts = [f"{n} {a['rate']:.3g} {UNITS[n]}/s fails={a['fails']}" for n, a in agg.items()]
        extra = []
        if sample.get("freq_avg_ghz"):
            extra.append(f"{sample['freq_avg_ghz']:.2f} GHz")
        if sample.get("temp_max_c"):
            extra.append(f"{sample['temp_max_c']:.0f} C")
        if sample.get("power_w"):
            extra.append(f"{sample['power_w']:.0f} W")
        log(f"  [{t:6.0f}s] " + (" | ".join(parts) or "running (first steps not finished yet)") + ("  (" + ", ".join(extra) + ")" if extra else ""))

    phases = [[w] for w in specs] if args.mode == "phases" else [list(specs)]
    report_phases = []
    for ph in phases:
        comm.barrier()
        log(f"\n== stress: {' + '.join(ph)} for {args.duration:.0f}s on {len(cpus)} workers ==")
        res = runner.run([(w, specs[w]) for w in ph], cpus, args.duration, tel, args.interval,
                         status if comm.rank == 0 else None)
        for name, a in res["aggregate"].items():
            fps = {}
            for wk in res["workers"]:
                fps.setdefault(wk["workloads"][name]["first"], 0)
                fps[wk["workloads"][name]["first"]] += 1
            a["majority_fingerprint"] = max(fps, key=fps.get)
        res["telemetry_summary"] = Telemetry.summarize(res["telemetry"])
        report_phases.append({"workloads": ph, "result": res})

    # ------------------------------------------------------------ summary
    node = {"host": si.hostname, "system": si.as_dict(), "theoretical": theo, "capacity": cap,
            "measured_peaks": measured, "phases": report_phases}
    nodes = comm.gather(node)
    if comm.rank != 0:
        return 0
    log("\n== summary ==")
    verdict = "PASS"
    for i, ph in enumerate(report_phases):
        merged = merge_nodes([{"host": n["host"], "aggregate": n["phases"][i]["result"]["aggregate"]}
                              for n in nodes])
        lost = sum(len(n["phases"][i]["result"]["lost_workers"]) for n in nodes)
        for name, m in merged.items():
            agg = ph["result"]["aggregate"][name]
            cf = ceiling_fraction(name, m["rate_total"] / len(nodes), measured, theo)
            line = (f"  {name:<12} {m['rate_total']:12.4g} {UNITS[name]}/s total"
                    f"  (per core {agg['rate_per_worker_min']:.3g}-{agg['rate_per_worker_max']:.3g})"
                    f"  steps={m['steps']}  failures={m['fails']}")
            if "pct_of_measured_peak" in cf:
                line += f"  | {cf['pct_of_measured_peak']:.1f}% of measured {cf['precision']} GEMM peak"
            if "pct_of_theoretical_peak" in cf:
                line += f", {cf['pct_of_theoretical_peak']:.1f}% of theoretical {cf['precision']}"
            if "pct_of_measured_bw" in cf:
                line += f"  | >= {cf['min_traffic_gbs']:.1f} GB/s ({cf['pct_of_measured_bw']:.0f}% of triad)"
            log(line)
            ph.setdefault("ceilings", {})[name] = cf
            if m["fails"] or lost:
                verdict = "FAIL"
            if m["inconsistent"] or m["inconsistent_nodes"]:
                log(f"    WARNING: results differ from the majority on {m['inconsistent'] + m['inconsistent_nodes']}")
                verdict = "FAIL" if verdict == "FAIL" else "WARN"
            for wk in ph["result"]["workers"]:
                for f in wk["workloads"][name]["fails"][:3]:
                    log(f"    cpu {wk['cpu']}: t={f['t']}s {f['error']}")
        if lost:
            log(f"    {lost} worker(s) died (OOM kill, crash or machine check?)")
        ts = ph["result"]["telemetry_summary"]
        if ts:
            log("    telemetry: " + ", ".join(f"{k} {v['min']:.3g}..{v['max']:.3g}" for k, v in ts.items()))
    log(f"\nverdict: {verdict}")
    if args.output:
        with open(args.output, "w") as fh:
            json.dump({"version": 1, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "verdict": verdict,
                       "args": vars(args), "nodes": nodes}, fh, indent=1, default=str)
        log(f"report written to {args.output}")
    return 0 if verdict != "FAIL" else 1


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m stress", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workers", default="physical",
                        help="physical (one per core, default) | logical (incl. SMT) | N")
    common.add_argument("--freq-ghz", type=float, help="clock for the theoretical peak (default: max boost)")
    common.add_argument("--flops-per-cycle", type=float, help="FP64 FLOPs/cycle/core override")
    common.add_argument("--mem-bw-gbs", type=float, help="theoretical memory bandwidth (channels x MT/s x 8 B)")
    common.add_argument("--mpi", action="store_true", help="multi-node run via mpi4py (one rank per node)")
    sub.add_parser("info", parents=[common], help="show hardware and estimated ceilings")
    pk = sub.add_parser("peak", parents=[common], help="measure DGEMM / bandwidth ceilings")
    pk.add_argument("--peak-seconds", type=float, default=4.0)
    r = sub.add_parser("run", parents=[common], help="verified stress run")
    r.add_argument("--duration", type=float, default=60.0, help="seconds per phase")
    r.add_argument("--mode", choices=["phases", "mixed"], default="phases",
                   help="phases: one workload at a time; mixed: every worker cycles through all")
    r.add_argument("--workloads", help=f"comma list from {','.join(ALL_WORKLOADS)}")
    r.add_argument("--size", choices=["auto", "cache", "llc", "dram"], default="auto",
                   help="working-set size: fits in L2 / in L3 / spills to DRAM")
    r.add_argument("--qubits", type=int, help="statevector qubits (overrides --size)")
    r.add_argument("--dgemm-n", type=int)
    r.add_argument("--molecule", help="VQE molecule (default by --size)")
    r.add_argument("--interval", type=float, default=10.0, help="status/telemetry interval (s)")
    r.add_argument("--no-peak", action="store_true", help="skip the ceiling measurement")
    r.add_argument("--peak-seconds", type=float, default=4.0)
    r.add_argument("-o", "--output", help="write a JSON report")
    args = ap.parse_args(argv)
    if getattr(args, "molecule", None):
        from qcore.molecules import get_molecule
        try:
            get_molecule(args.molecule)  # fail before the (slow) peak measurement
        except ValueError as exc:
            sys.exit(f"error: {exc}")

    comm = Comm(args.mpi)
    log = log_fn(comm)
    if args.cmd == "run":
        return cmd_run(args, comm, log)
    si = collect()
    theo = peaks.theoretical(si, args.freq_ghz, args.flops_per_cycle, args.mem_bw_gbs)
    print_info(si, theo, peaks.capacity(si), log)
    if args.cmd == "peak":
        log("\n== ceilings: measuring what this system can achieve ==")
        print_peaks(peaks.measure(si, select_cpus(si, args.workers), seconds=args.peak_seconds, log=log), theo, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
