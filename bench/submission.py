"""The submission tree: what each ministage hands in, and a checker teams run before uploading.

    python -m bench submit-check --layout                 # print the expected tree
    python -m bench submit-check team07/valqyrie          # check a tree (any subset of ministages)

A submission is a folder per ministage, ``<root>/stage<S>/ministage<M>/``. Ministages can be
handed in in any order and at any time; a missing folder is simply not graded yet. Each folder
holds the files listed in ``SPEC``:

* ``run``: a ``python -m bench run ... -o <file>`` report, on the case named (``hidden`` = the
  hidden case file the judges handed out, run with ``--case-file``);
* ``vqe``: a ``python -m bench vqe ... -o <file>`` report;
* ``peak`` / ``stress``: a ``python -m stress peak|run ... -o <file>`` report;
* ``metrics``: a JSON object with the listed keys (numbers unless stated);
* ``text``: a non-empty text file; ``file``: any non-empty file; ``dir``: a non-empty folder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .cases import CASES, build_case


@dataclass(frozen=True)
class Item:
    name: str
    kind: str                    # run | vqe | peak | stress | metrics | text | file | dir
    help: str
    case: str | None = None      # run / vqe: a case name, or "hidden"
    min_repeats: int = 0         # run: --repeats at least this
    point: int | None = None     # run: made with --point this
    min_ranks: int = 1           # run: MPI ranks (one per GPU) at least this
    max_ranks: int | None = None
    keys: tuple = ()             # metrics: required keys
    bool_keys: tuple = field(default=())


NOTES = Item("notes.md", "text", "what you changed and why, with the exact build and run commands")
CODE = Item("code.tar.gz", "file", "your code at this ministage: git archive -o code.tar.gz HEAD")
FIXED = Item("fixed.json", "run", "bench run h2o-24q --impl ... --repeats 4 -o fixed.json",
             case="h2o-24q", min_repeats=4)
HIDDEN = Item("hidden.json", "run", "bench run --case-file <hidden>.npz --impl ... -o hidden.json", case="hidden")
TIMED = (FIXED, HIDDEN, CODE, NOTES)
MULTI_FIXED = Item("fixed.json", "run", "bench run h2o-24q --mpi --impl ... --repeats 4 -o fixed.json on all your GPUs",
                   case="h2o-24q", min_repeats=4, min_ranks=2)


def metrics(help, *keys, bools=()):
    return Item("metrics.json", "metrics", help, keys=keys, bool_keys=bools)


SPEC: dict[str, tuple[Item, ...]] = {
    "1.1": (Item("baseline-20q.json", "run", "bench run h2o-20q --repeats 4 -o baseline-20q.json",
                 case="h2o-20q", min_repeats=4),
            Item("baseline-24q.json", "run", "bench run h2o-24q --point 2 -o baseline-24q.json",
                 case="h2o-24q", point=2),
            NOTES),
    "1.2": (Item("peak.json", "peak", "python -m stress peak -o peak.json"), NOTES),
    "1.3": (Item("profile", "dir", "profiler output (py-spy, perf, VTune, uProf ...)"), NOTES),
    "1.4": (metrics("predicted time per evaluation and, per kernel, bytes and FLOPs per amplitude",
                    "predicted_eval_s", "kernels"), NOTES),
    "2.1": TIMED,
    "2.2": TIMED + (metrics("achieved single-core memory bandwidth", "achieved_gbs"),),
    "2.3": TIMED,
    "2.4": TIMED + (metrics("achieved all-core bandwidth and kernel times", "achieved_gbs",
                            "rotation_pass_s", "hpsi_s"),),
    "3.1": (metrics("sector index build", "index_build_s", "index_memory_bytes", bools=("round_trip_ok",)),
            CODE, NOTES),
    "3.2": TIMED + (metrics("time of one pass of all rotations", "rotation_pass_s"),),
    "3.3": TIMED + (metrics("sector Hamiltonian", "hpsi_s", "h_build_s", "h_memory_bytes"),),
    "3.4": TIMED,
    "4.1": TIMED,
    "4.2": TIMED + (Item("profile", "dir", "GPU profiler output showing host-device transfers"),),
    "4.3": TIMED + (metrics("achieved and measured peak GPU memory bandwidth", "achieved_gbs", "gpu_peak_gbs"),
                    Item("bandwidth", "dir", "BabelStream (or vendor equivalent) output")),
    "4.4": TIMED + (metrics("H psi time on the GPU", "hpsi_s"),),
    "5.1": (Item("fixed-2gpu.json", "run", "bench run h2o-24q --mpi on 2 GPUs (one rank per GPU)",
                 case="h2o-24q", min_ranks=2, max_ranks=2),
            Item("fixed-allgpu.json", "run", "bench run h2o-24q --mpi on all your GPUs", case="h2o-24q", min_ranks=2),
            HIDDEN, CODE, NOTES),
    # 5.2 and 5.3: the same code on all your GPUs, 5.2 with overlap switched off and 5.3 with it on
    "5.2": (MULTI_FIXED, HIDDEN, CODE, NOTES,
            metrics("share of run time spent communicating, overlap off (0-1)", "comm_fraction"),
            Item("profile", "dir", "profiler trace the comm_fraction was measured from")),
    "5.3": (MULTI_FIXED, HIDDEN, CODE, NOTES),
    "5.4": (Item("anthracene.json", "run", "bench run anthracene-28q --mpi --repeats 4 on all your GPUs",
                 case="anthracene-28q", min_repeats=4),
            HIDDEN, CODE, NOTES),
    "6.1": (Item("vqe.json", "vqe", "bench vqe h2o-24q --impl ... -o vqe.json", case="h2o-24q"),
            Item("hidden-vqe.json", "vqe", "bench vqe --case-file <hidden>.npz --impl ... -o hidden-vqe.json",
                 case="hidden"),
            CODE, NOTES),
    "6.2": (metrics("bond-length scan", "scan_time_s", "n_geometries"),
            Item("scan", "dir", "every VQE report of the scan"), CODE, NOTES),
    "6.3": (metrics("energy to solution under the power cap", "joules", "power_cap_w"),
            Item("vqe.json", "vqe", "the bench vqe h2o-24q run the energy was measured on", case="h2o-24q"),
            Item("power", "dir", "power meter trace of that run"), NOTES),
    "6.4": (Item("stress.json", "stress", "python -m stress run --mode mixed --duration 3600 -o stress.json"),
            HIDDEN, Item("report.pdf", "file", "the final report"), CODE, NOTES),
}


def folder(ministage: str) -> str:
    s, m = ministage.split(".")
    return f"stage{s}/ministage{m}"


def layout() -> str:
    lines = ["<team>/valqyrie/"]
    for ms, items in SPEC.items():
        lines.append(f"  {folder(ms)}/")
        lines += [f"      {it.name + ('/' if it.kind == 'dir' else ''):<20} {it.help}" for it in items]
    return "\n".join(lines)


# ------------------------------------------------------------------ checking
def _load_json(path: Path, problems: list):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        problems.append(f"{path.name}: not readable JSON ({exc})")
        return None


def check_item(path: Path, it: Item, fingerprints: dict) -> list[str]:
    """Problems with one file; an empty list means it is acceptable."""
    if it.kind == "dir":
        return [] if path.is_dir() and any(path.iterdir()) else [f"{it.name}/: missing or empty folder"]
    if not path.is_file() or path.stat().st_size == 0:
        return [f"{it.name}: missing or empty"]
    if it.kind in ("text", "file"):
        return []
    problems: list[str] = []
    d = _load_json(path, problems)
    if d is None:
        return problems
    if not isinstance(d, dict):
        return [f"{it.name}: expected a JSON object"]
    if it.kind == "metrics":
        for k in it.keys:
            if k not in d:
                problems.append(f"{it.name}: missing key {k!r}")
            elif k != "kernels" and not isinstance(d[k], (int, float)):
                problems.append(f"{it.name}: {k!r} must be a number")
        if "kernels" in it.keys and not isinstance(d.get("kernels"), dict):
            problems.append(f"{it.name}: 'kernels' must map each kernel to its bytes and FLOPs per amplitude")
        problems += [f"{it.name}: {k!r} must be true or false" for k in it.bool_keys if not isinstance(d.get(k), bool)]
        return problems
    if it.kind == "peak":
        return [] if "triad_1core_gbs" in d.get("measured", {}) else [f"{it.name}: not a stress peak report"]
    if it.kind == "stress":
        return [] if "verdict" in d else [f"{it.name}: not a stress run report"]
    # bench run / vqe reports
    is_run = "points" in d
    if (it.kind == "run") != is_run or "case" not in d:
        return [f"{it.name}: not a bench {it.kind} report"]
    case = d["case"]
    if it.case == "hidden":
        if case in CASES:
            problems.append(f"{it.name}: made on {case}, not on the hidden case")
    elif case != it.case:
        problems.append(f"{it.name}: made on {case}, expected {it.case}")
    elif d.get("case_sha256") != fingerprints.setdefault(case, build_case(case).fingerprint()):
        problems.append(f"{it.name}: the case data differs from the shipped {case}")
    if is_run:
        if d.get("gate") == "FAIL":
            problems.append(f"{it.name}: correctness gate FAIL")
        if d.get("repeats", 0) < it.min_repeats:
            problems.append(f"{it.name}: needs --repeats {it.min_repeats} or more")
        if it.point is not None and [p["point"] for p in d["points"]] != [it.point]:
            problems.append(f"{it.name}: needs --point {it.point}")
        elif it.point is None and len(d["points"]) != 3:
            problems.append(f"{it.name}: must check all three points (no --points / --point)")
        ranks = d.get("ranks", 1)
        if ranks < it.min_ranks or (it.max_ranks is not None and ranks > it.max_ranks):
            want = it.min_ranks if it.max_ranks == it.min_ranks else f">= {it.min_ranks}" if it.max_ranks is None else f"<= {it.max_ranks}"
            problems.append(f"{it.name}: ran on {ranks} rank(s), expected {want}")
    elif d.get("gate") == "FAIL":
        problems.append(f"{it.name}: final energy does not match the reference")
    return problems


def check_tree(root) -> dict:
    """{ministage: [problems]} for every ministage folder present, plus unknown folders under "?"."""
    root = Path(root)
    out, fingerprints = {}, {}
    for ms, items in SPEC.items():
        d = root / folder(ms)
        if d.is_dir():
            out[ms] = [p for it in items for p in check_item(d / it.name, it, fingerprints)]
    known = {folder(ms) for ms in SPEC}
    extra = [str(p.relative_to(root)) for p in sorted(root.glob("stage*/ministage*")) if str(p.relative_to(root)) not in known]
    if extra:
        out["?"] = [f"unknown folder {e}" for e in extra]
    return out
