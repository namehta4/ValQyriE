"""Clock, temperature and package-power sampling from Linux sysfs (best effort).

RAPL energy counters (``/sys/class/powercap/intel-rapl:*``; also used by AMD Zen on
Linux >= 5.8) are usually root-readable only.  On macOS nothing is sampled
(``powermetrics`` needs sudo); fields are then ``None``.
"""
from __future__ import annotations

import time
from pathlib import Path

CPU_HWMON = {"coretemp", "k10temp", "zenpower", "cpu_thermal", "soc_thermal", "tegra_thermal"}


def _read_int(p):
    try:
        return int(Path(p).read_text().strip())
    except (OSError, ValueError):
        return None


class Telemetry:
    def __init__(self, cpus):
        self.freq_files = [f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq" for c in cpus]
        self.temp_files = []
        for hw in Path("/sys/class/hwmon").glob("hwmon*"):
            try:
                name = (hw / "name").read_text().strip()
            except OSError:
                continue
            if name in CPU_HWMON:
                self.temp_files += sorted(hw.glob("temp*_input"))
        if not self.temp_files:
            self.temp_files = sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp"))
        self.rapl = []
        for d in sorted(Path("/sys/class/powercap").glob("intel-rapl:[0-9]*")):
            if ":" in d.name.split("intel-rapl:", 1)[1]:
                continue  # sub-domain (core/uncore/dram); keep package level only
            e, rng = _read_int(d / "energy_uj"), _read_int(d / "max_energy_range_uj")
            if e is not None:
                self.rapl.append((d / "energy_uj", rng or 2 ** 32))
        self.power_note = None if self.rapl else (
            "RAPL present but not readable (run as root for power)" if any(Path("/sys/class/powercap").glob("intel-rapl:*"))
            else "no RAPL power counters")
        self._last = self._energy()
        self.samples = []

    def _energy(self):
        if not self.rapl:
            return None
        return time.monotonic(), [_read_int(f) for f, _ in self.rapl]

    def sample(self) -> dict:
        freqs = [v for v in (_read_int(f) for f in self.freq_files) if v]
        temps = [v for v in (_read_int(f) for f in self.temp_files) if v]
        s = {"t": time.time(),
             "freq_avg_ghz": sum(freqs) / len(freqs) / 1e6 if freqs else None,
             "freq_min_ghz": min(freqs) / 1e6 if freqs else None,
             "temp_max_c": max(temps) / 1000 if temps else None,
             "power_w": None}
        now = self._energy()
        if now and self._last:
            dt = now[0] - self._last[0]
            if dt > 0:
                tot = 0.0
                for (_, rng), a, b in zip(self.rapl, self._last[1], now[1]):
                    if a is None or b is None:
                        tot = None
                        break
                    tot += ((b - a) % rng) / 1e6
                s["power_w"] = tot / dt if tot is not None else None
        self._last = now
        self.samples.append(s)
        return s

    @staticmethod
    def summarize(samples) -> dict:
        out = {}
        for k in ("freq_avg_ghz", "freq_min_ghz", "temp_max_c", "power_w"):
            vals = [s[k] for s in samples if s.get(k) is not None]
            if vals:
                out[k] = {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}
        return out
