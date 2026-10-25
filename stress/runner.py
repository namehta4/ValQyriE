"""Run workloads on one single-threaded worker process per core.

Workers use the ``spawn`` start method (identical behaviour on Linux and macOS), are
pinned to one logical CPU each where the OS allows it (Linux ``sched_setaffinity``),
and have every BLAS/OpenMP thread pool limited to one thread so that N workers put
exactly N cores under load.  All workers start on a barrier, so the measured
interval is the same for everyone.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_mod
import statistics
import threading
import time
from contextlib import contextmanager

THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "GOTO_NUM_THREADS")


@contextmanager
def single_threaded_env():
    saved = {k: os.environ.get(k) for k in THREAD_ENV}
    os.environ.update({k: "1" for k in THREAD_ENV})
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _pin(cpu) -> bool:
    try:
        os.sched_setaffinity(0, {cpu})
        return True
    except (AttributeError, OSError):
        return False


def _worker(wid, cpu, specs, duration, barrier, q, hb_interval):
    pinned = _pin(cpu)
    try:
        if any(name == "transformer" for name, _ in specs):
            import torch
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
        from .workloads import build
        loads = [build(name, params) for name, params in specs]
    except Exception as exc:  # report and release the others
        q.put(("error", wid, f"{type(exc).__name__}: {exc}"))
        barrier.abort()
        return
    try:
        barrier.wait()
    except threading.BrokenBarrierError:
        return
    stats = {wl.name: {"steps": 0, "work": 0.0, "time": 0.0, "first": None, "fails": [], "n_fails": 0,
                       "step_times": []} for wl in loads}
    t0 = time.perf_counter()
    end, last_hb, k = t0 + duration, t0, 0
    while True:
        wl = loads[k % len(loads)]
        k += 1
        st = stats[wl.name]
        ts = time.perf_counter()
        try:
            work, fp, err = wl.step()
        except Exception as exc:
            work, fp, err = 0.0, None, f"exception: {type(exc).__name__}: {exc}"
        te = time.perf_counter()
        st["steps"] += 1
        st["work"] += work
        st["time"] += te - ts
        st["step_times"].append(te - ts)
        if st["first"] is None:
            st["first"] = fp
        elif fp != st["first"] and err is None:
            err = f"result changed between identical steps (step {st['steps']})"
        if err:
            st["n_fails"] += 1
            if len(st["fails"]) < 20:
                st["fails"].append({"t": round(te - t0, 3), "error": err})
        if te - last_hb >= hb_interval:
            q.put(("hb", wid, {n: (s["steps"], s["work"], s["n_fails"]) for n, s in stats.items()}))
            last_hb = te
        if te >= end:
            break
    for st in stats.values():
        times = st.pop("step_times")
        st.update(step_min=min(times), step_median=statistics.median(times), step_max=max(times))
    q.put(("done", wid, {"cpu": cpu, "pinned": pinned, "elapsed": time.perf_counter() - t0, "workloads": stats}))


def run(specs, cpus, duration, telemetry=None, interval=5.0, on_status=None, build_timeout=900):
    """Run ``specs`` (list of (name, params)) on one worker per CPU in ``cpus``.

    Every worker cycles through all specs (one spec -> a pure phase).  Returns a dict with
    per-worker results, aggregate rates and telemetry samples.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    barrier = ctx.Barrier(len(cpus) + 1)
    with single_threaded_env():
        procs = [ctx.Process(target=_worker, args=(i, c, specs, duration, barrier, q, min(1.0, interval)),
                             daemon=True) for i, c in enumerate(cpus)]
        for p in procs:
            p.start()
    errors = []
    try:
        barrier.wait(timeout=build_timeout)
    except threading.BrokenBarrierError:
        while True:
            try:
                msg = q.get(timeout=1)
            except queue_mod.Empty:
                break
            if msg[0] == "error":
                errors.append(msg[2])
        for p in procs:
            p.terminate()
        raise RuntimeError("worker start-up failed: " + "; ".join(sorted(set(errors)) or ["timeout"]))
    t_start = time.perf_counter()
    live = {}
    done = {}
    tel = []
    next_status = t_start + interval
    while len(done) < len(procs):
        try:
            msg = q.get(timeout=0.5)
        except queue_mod.Empty:
            msg = None
            if all(not p.is_alive() for p in procs) and q.empty():
                break
        if msg and msg[0] == "hb":
            live[msg[1]] = msg[2]
        elif msg and msg[0] == "done":
            done[msg[1]] = msg[2]
            live[msg[1]] = {n: (s["steps"], s["work"], s["n_fails"]) for n, s in msg[2]["workloads"].items()}
        now = time.perf_counter()
        if now >= next_status:
            sample = telemetry.sample() if telemetry else {}
            tel.append(sample)
            if on_status:
                on_status(now - t_start, _aggregate_live(live, now - t_start), sample)
            next_status += interval
    wall = time.perf_counter() - t_start
    for p in procs:
        p.join(timeout=10)
    lost = [i for i in range(len(procs)) if i not in done]
    return {"workers": [done[i] for i in sorted(done)], "lost_workers": lost, "wall_time": wall,
            "telemetry": tel, "aggregate": aggregate([done[i] for i in sorted(done)])}


def _aggregate_live(live, elapsed):
    out = {}
    for per in live.values():
        for name, (steps, work, fails) in per.items():
            a = out.setdefault(name, {"steps": 0, "work": 0.0, "fails": 0})
            a["steps"] += steps
            a["work"] += work
            a["fails"] += fails
    for a in out.values():
        a["rate"] = a["work"] / elapsed if elapsed > 0 else 0.0
    return out


def aggregate(workers) -> dict:
    """Totals per workload, plus cross-worker consistency: all workers ran identical inputs, so
    their result fingerprints must agree; outliers point at a specific core."""
    out = {}
    for w in workers:
        for name, s in w["workloads"].items():
            a = out.setdefault(name, {"steps": 0, "work": 0.0, "fails": 0, "rates": [], "fingerprints": {}})
            a["steps"] += s["steps"]
            a["work"] += s["work"]
            a["fails"] += s["n_fails"]
            a["rates"].append(s["work"] / w["elapsed"])
            a["fingerprints"].setdefault(s["first"], []).append(w["cpu"])
    for a in out.values():
        rates = a.pop("rates")
        a["rate_total"] = sum(rates)            # unit per second, all workers
        a["rate_per_worker_min"] = min(rates)
        a["rate_per_worker_max"] = max(rates)
        a["rate_per_worker_mean"] = sum(rates) / len(rates)
        fps = a.pop("fingerprints")
        majority = max(fps.values(), key=len)
        a["inconsistent_cpus"] = sorted(c for cpus in fps.values() if cpus is not majority for c in cpus)
    return out
