"""Portable system inspection (Linux x86_64/aarch64, macOS).  Everything is best effort:
missing information is reported as ``None`` rather than guessed."""
from __future__ import annotations

import os
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

SYS_CPU = Path("/sys/devices/system/cpu")


def _read(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _size_to_bytes(s):
    if s is None:
        return None
    m = re.match(r"(\d+)\s*([KMG]?)", str(s))
    if not m:
        return None
    return int(m.group(1)) * {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30}[m.group(2)]


@dataclass
class SystemInfo:
    hostname: str
    os: str
    arch: str
    cpu_model: str | None
    vendor: str | None
    sockets: int | None
    physical_cores: int
    logical_cpus: int
    numa_nodes: int | None
    max_freq_ghz: float | None
    l1d_bytes: int | None
    l2_bytes: int | None
    l3_bytes: int | None
    memory_bytes: int | None
    flags: list = field(default_factory=list)
    core_cpus: list = field(default_factory=list)  # one logical CPU id per physical core
    cpu_family: int | None = None
    cpu_part: str | None = None
    simd: dict = field(default_factory=dict)       # runtime SIMD dispatch of the numerical libraries

    def as_dict(self) -> dict:
        d = asdict(self)
        d["flags"] = [f for f in self.flags if f in INTERESTING_FLAGS]
        return d


INTERESTING_FLAGS = {"sse4_2", "avx", "avx2", "fma", "avx512f", "avx512dq", "avx512bw", "avx512vl",
                     "avx512_vnni", "avx512_bf16", "avx512_fp16", "amx_tile", "amx_bf16", "amx_int8",
                     "asimd", "sve", "sve2", "sme", "fphp", "asimdhp"}


def _allowed_cpus():
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        return list(range(os.cpu_count() or 1))


def _linux_topology(cpus):
    cores, packages = {}, set()
    for c in cpus:
        pkg = _read(SYS_CPU / f"cpu{c}/topology/physical_package_id", "0")
        core = _read(SYS_CPU / f"cpu{c}/topology/core_id", str(c))
        packages.add(pkg)
        cores.setdefault((pkg, core), c)
    return sorted(cores.values()), len(packages)


def _linux_cache(level, kind=("Data", "Unified")):
    best = None
    for d in sorted(SYS_CPU.glob("cpu0/cache/index*")):
        if _read(d / "level") == str(level) and _read(d / "type") in kind:
            best = _size_to_bytes(_read(d / "size"))
    return best


def _linux_meminfo():
    txt = _read("/proc/meminfo", "")
    m = re.search(r"MemTotal:\s+(\d+) kB", txt)
    return int(m.group(1)) * 1024 if m else None


def simd_dispatch() -> dict:
    """Which SIMD extensions NumPy and PyTorch found at runtime (no compile-time pinning)."""
    out = {}
    try:
        import numpy as np
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as feats
        except ImportError:  # NumPy 1.x
            from numpy.core._multiarray_umath import __cpu_features__ as feats
        out["numpy"] = sorted(k for k, v in feats.items() if v and k in {
            "SSE42", "AVX", "AVX2", "FMA3", "AVX512F", "AVX512_SKX", "AVX512_ICL", "AVX512_SPR",
            "ASIMD", "ASIMDHP", "SVE", "NEON"})
        out["numpy_version"] = np.__version__
    except Exception:  # private API may move; information only
        pass
    try:
        import torch
        out["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
        out["torch_version"] = torch.__version__
    except Exception:
        pass
    return out


def collect() -> SystemInfo:
    system = platform.system()
    cpus = _allowed_cpus()
    info = dict(hostname=platform.node(), os=f"{system} {platform.release()}", arch=platform.machine(),
                cpu_model=None, vendor=None, sockets=None, physical_cores=len(cpus), logical_cpus=len(cpus),
                numa_nodes=None, max_freq_ghz=None, l1d_bytes=None, l2_bytes=None, l3_bytes=None,
                memory_bytes=None, flags=[], core_cpus=cpus)
    if system == "Linux":
        cpuinfo = _read("/proc/cpuinfo", "")
        for key in ("model name", "Model", "cpu model"):
            m = re.search(rf"^{key}\s*:\s*(.+)$", cpuinfo, re.M)
            if m:
                info["cpu_model"] = m.group(1).strip()
                break
        m = re.search(r"^vendor_id\s*:\s*(.+)$", cpuinfo, re.M)
        if m:
            info["vendor"] = m.group(1).strip()
        m = re.search(r"^CPU implementer\s*:\s*(\S+)$", cpuinfo, re.M)
        if m and not info["vendor"]:
            info["vendor"] = {"0x41": "ARM", "0x61": "Apple", "0xc0": "Ampere", "0x48": "HiSilicon",
                              "0x46": "Fujitsu", "0x4e": "NVIDIA"}.get(m.group(1), m.group(1))
        m = re.search(r"^cpu family\s*:\s*(\d+)$", cpuinfo, re.M)
        if m:
            info["cpu_family"] = int(m.group(1))
        m = re.search(r"^CPU part\s*:\s*(\S+)$", cpuinfo, re.M)
        if m:
            info["cpu_part"] = m.group(1)
        m = re.search(r"^(flags|Features)\s*:\s*(.+)$", cpuinfo, re.M)
        if m:
            info["flags"] = m.group(2).split()
        core_cpus, sockets = _linux_topology(cpus)
        info.update(core_cpus=core_cpus, physical_cores=len(core_cpus), sockets=sockets)
        info["numa_nodes"] = len(list(Path("/sys/devices/system/node").glob("node[0-9]*"))) or None
        f = _read(SYS_CPU / "cpu0/cpufreq/cpuinfo_max_freq")
        if f:
            info["max_freq_ghz"] = int(f) / 1e6
        info.update(l1d_bytes=_linux_cache(1), l2_bytes=_linux_cache(2), l3_bytes=_linux_cache(3),
                    memory_bytes=_linux_meminfo())
    elif system == "Darwin":
        info["cpu_model"] = _sysctl("machdep.cpu.brand_string")
        info["vendor"] = "Apple" if platform.machine() == "arm64" else _sysctl("machdep.cpu.vendor")
        phys = _sysctl("hw.physicalcpu")
        if phys:
            info["physical_cores"] = int(phys)
            info["core_cpus"] = list(range(int(phys)))  # no pinning on macOS; ids are nominal
        info["sockets"] = int(_sysctl("hw.packages") or 1)
        f = _sysctl("hw.cpufrequency_max")  # Intel Macs only
        if f:
            info["max_freq_ghz"] = int(f) / 1e9
        info["l1d_bytes"] = _size_to_bytes(_sysctl("hw.perflevel0.l1dcachesize") or _sysctl("hw.l1dcachesize"))
        info["l2_bytes"] = _size_to_bytes(_sysctl("hw.perflevel0.l2cachesize") or _sysctl("hw.l2cachesize"))
        info["l3_bytes"] = _size_to_bytes(_sysctl("hw.l3cachesize"))
        info["memory_bytes"] = _size_to_bytes(_sysctl("hw.memsize"))
        feats = " ".join(filter(None, [_sysctl("machdep.cpu.features"), _sysctl("machdep.cpu.leaf7_features")]))
        info["flags"] = feats.lower().replace(".", "_").split() or (["asimd"] if platform.machine() == "arm64" else [])
    si = SystemInfo(**info)
    si.simd = simd_dispatch()
    return si


# ----------------------------------------------------------------- peak model
def flops_per_cycle_fp64(si: SystemInfo) -> tuple[float | None, str]:
    """FP64 FLOPs/cycle/core from the ISA (FMA units x vector lanes x 2).  An *estimate*:
    unit counts differ between SKUs, so it is reported with its basis and can be overridden."""
    flags = set(si.flags)
    arch = si.arch.lower()
    if arch in ("x86_64", "amd64"):
        amd = (si.vendor or "").lower().startswith(("authenticamd", "amd"))
        if "avx512f" in flags:
            if amd and si.cpu_family == 0x19:
                return 16.0, "AMD Zen 4: AVX-512 on 256-bit datapaths (2 FMA x 4 lanes x 2)"
            if amd:
                return 32.0, "AMD Zen 5+: 2 x 512-bit FMA (2 x 8 lanes x 2)"
            return 32.0, "Intel AVX-512, assuming 2 FMA units (Xeon Gold/Platinum); 1-unit SKUs = 16"
        if "avx2" in flags and "fma" in flags:
            return 16.0, "AVX2: 2 x 256-bit FMA (2 x 4 lanes x 2)"
        if "avx" in flags:
            return 8.0, "AVX without FMA"
        return 4.0, "SSE2 baseline"
    if arch in ("aarch64", "arm64"):
        parts = {"0xd40": (16.0, "Neoverse V1: 2 x 256-bit SVE FMA"),
                 "0xd4f": (16.0, "Neoverse V2 (e.g. Grace): 4 x 128-bit FMA"),
                 "0xd84": (16.0, "Neoverse V3: 4 x 128-bit FMA"),
                 "0xd0c": (8.0, "Neoverse N1: 2 x 128-bit FMA"),
                 "0xd49": (8.0, "Neoverse N2: 2 x 128-bit FMA"),
                 "0x46": (32.0, "Fujitsu A64FX: 2 x 512-bit SVE FMA")}
        if si.cpu_part in parts:
            return parts[si.cpu_part]
        if (si.vendor or "") == "Apple":
            return 16.0, "Apple performance core: 4 x 128-bit FMA"
        return None, "unknown ARM core: pass --flops-per-cycle"
    return None, "unknown architecture: pass --flops-per-cycle"
