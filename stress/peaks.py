"""System ceilings: what the machine can achieve at most.

* theoretical FP64 peak  = cores x max clock x FLOPs/cycle  (estimate from the ISA; an upper
  bound, since all-core clocks under AVX load are lower than the max boost clock)
* measured DGEMM         = multithreaded BLAS on one big matrix, and one single-threaded
                           DGEMM per core (the "dgemm" stress workload)
* measured bandwidth     = triad on all cores / one core
* capacity               = largest statevector / VQE problem that fits in RAM
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import runner
from .sysinfo import SystemInfo, flops_per_cycle_fp64

# bytes per amplitude, measured with tracemalloc (see stress/README section in the main README)
BYTES_PER_AMP_CIRCUIT = 96.0     # random-circuit simulation (stress "statevector" workload)
BYTES_PER_AMP_VQE = 128.0        # energy + adjoint gradient buffers
BYTES_PER_NNZ = 20.0             # CSR complex128 value + int32 column index (+ row pointers)
NNZ_PER_ROW_REF = 116.0          # NH3 / STO-3G (8e,7o), 14 qubits: densest library Hamiltonian


def theoretical(si: SystemInfo, freq_ghz=None, flops_per_cycle=None, mem_bw_gbs=None) -> dict:
    fpc, basis = (flops_per_cycle, "user supplied") if flops_per_cycle else flops_per_cycle_fp64(si)
    f = freq_ghz or si.max_freq_ghz
    out = {"flops_per_cycle_fp64": fpc, "flops_per_cycle_basis": basis, "freq_ghz": f,
           "peak_fp32_gflops": None,
           "freq_basis": "user supplied" if freq_ghz else ("cpuinfo max boost" if f else "unknown: pass --freq-ghz"),
           "cores": si.physical_cores, "peak_fp64_gflops": None, "mem_bw_gbs": mem_bw_gbs}
    if fpc and f:
        out["peak_fp64_gflops"] = si.physical_cores * f * fpc
        out["peak_fp32_gflops"] = 2 * out["peak_fp64_gflops"]  # twice the lanes per vector
    return out


def capacity(si: SystemInfo, usable_fraction: float = 0.8) -> dict:
    mem = si.memory_bytes
    if not mem:
        return {}
    usable = usable_fraction * mem

    def qubits(bytes_per_amp):
        return int(math.floor(math.log2(usable / bytes_per_amp)))

    return {"memory_gb": mem / 1e9, "usable_fraction": usable_fraction,
            "max_qubits_state_only": qubits(16.0),
            "max_qubits_circuit_sim": qubits(BYTES_PER_AMP_CIRCUIT),
            "max_qubits_vqe_state": qubits(BYTES_PER_AMP_VQE),
            "max_qubits_vqe_with_sparse_h": qubits(BYTES_PER_AMP_VQE + BYTES_PER_NNZ * NNZ_PER_ROW_REF),
            "note": "single process using the whole node; the sparse Hamiltonian estimate assumes "
                    "NH3-like density (nnz/row grows with molecule size, so larger molecules hit "
                    "the limit sooner)"}


def blas_dgemm(n: int = 4096, repeats: int = 3) -> dict:
    """Multithreaded BLAS DGEMM in this process (library default thread count)."""
    rng = np.random.default_rng(0)
    a, b = rng.random((n, n)), rng.random((n, n))
    a @ b  # warm-up (thread pool, page faults)
    best = math.inf
    for _ in range(repeats):
        t = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t)
    threads = None
    try:
        from threadpoolctl import threadpool_info
        threads = max((i.get("num_threads") or 0) for i in threadpool_info()) or None
    except Exception:
        pass
    return {"n": n, "gflops": 2 * n ** 3 / best / 1e9, "threads": threads}


def torch_sgemm(n: int = 4096, repeats: int = 3) -> dict | None:
    """Multithreaded FP32 GEMM through PyTorch (the library the transformer workload uses)."""
    try:
        import torch
    except ImportError:
        return None
    g = torch.Generator().manual_seed(0)
    a, b = torch.rand(n, n, generator=g), torch.rand(n, n, generator=g)
    a @ b
    best = math.inf
    for _ in range(repeats):
        t = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t)
    return {"n": n, "gflops": 2 * n ** 3 / best / 1e9, "threads": torch.get_num_threads()}


def _rate(res, name):
    return res["aggregate"][name]["rate_total"]


def triad_size(si: SystemInfo, n_workers: int) -> int:
    """Elements per array so that the arrays are >> caches but bounded by RAM."""
    l3 = si.l3_bytes or 32 << 20
    l2 = si.l2_bytes or 1 << 20
    want = max(8 * l3 / (24 * n_workers), 16 * l2 / 24, 1 << 22)
    if si.memory_bytes:
        want = min(want, 0.2 * si.memory_bytes / (24 * n_workers))
    return int(2 ** math.ceil(math.log2(max(want, 1 << 16))))


def measure(si: SystemInfo, cpus, seconds: float = 4.0, dgemm_n: int = 4096, log=print) -> dict:
    out = {}
    log(f"  BLAS DGEMM (n={dgemm_n}, multithreaded) ...")
    out["dgemm_blas"] = blas_dgemm(dgemm_n)
    sg = torch_sgemm(dgemm_n)
    if sg:
        log(f"  FP32 GEMM via PyTorch (n={dgemm_n}, multithreaded) ...")
        out["sgemm_torch"] = sg
    log(f"  DGEMM, one single-threaded process per core ({len(cpus)}) ...")
    spec = [("dgemm", {"n": 1024})]
    r1 = runner.run(spec, cpus[:1], seconds)
    rn = runner.run(spec, cpus, seconds)
    out["dgemm_1core_gflops"] = _rate(r1, "dgemm")
    out["dgemm_allcore_gflops"] = _rate(rn, "dgemm")
    m = triad_size(si, len(cpus))
    log(f"  memory triad ({m * 24 / 2**20:.0f} MiB per process) ...")
    t1 = runner.run([("triad", {"n_elements": triad_size(si, 1)})], cpus[:1], seconds)
    tn = runner.run([("triad", {"n_elements": m})], cpus, seconds)
    out["triad_1core_gbs"] = _rate(t1, "triad")
    out["triad_allcore_gbs"] = _rate(tn, "triad")
    out["measured_peak_fp64_gflops"] = max(out["dgemm_blas"]["gflops"], out["dgemm_allcore_gflops"])
    fails = sum(r["aggregate"][k]["fails"] for r in (r1, rn, t1, tn) for k in r["aggregate"])
    out["verification_failures"] = fails
    return out
