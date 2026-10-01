"""Hardware-aware training choices and estimates with no framework dependency."""

from __future__ import annotations

import math
import os
import subprocess
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


EPOCH_CAPS = {"finetune": 1000, "scratch": 2000}
GPU_BATCH_CANDIDATES = (4, 8, 16, 32, 64)
GPU_MEMORY_MARGIN = 0.85


def epoch_cap_for(mode: str) -> int:
    return EPOCH_CAPS[mode]


def effective_cpu_count() -> int:
    affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)
    count = len(affinity)
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.is_file():
        parts = quota.read_text().split()
        if parts[0] != "max":
            count = min(count, max(1, math.ceil(int(parts[0]) / int(parts[1]))))
    return max(1, count)


def physical_cpu_count() -> int:
    try:
        records = Path("/proc/cpuinfo").read_text().split("\n\n")
        cores = set()
        for record in records:
            fields = dict(line.split(":", 1) for line in record.splitlines() if ":" in line)
            if "physical id" in fields and "core id" in fields:
                cores.add((fields["physical id"].strip(), fields["core id"].strip()))
        return min(len(cores), effective_cpu_count()) if cores else effective_cpu_count()
    except OSError:
        return effective_cpu_count()


def workers_for(device: str, physical_cores: int, logical_cores: int, training_samples: int | None = None) -> int:
    cores = max(1, min(physical_cores or logical_cores, logical_cores))
    base = min(8, max(1, cores - 2)) if device == "cuda" else min(2, max(0, cores // 4))
    if training_samples is not None:
        base = min(base, 1 if training_samples < 64 else 2 if training_samples < 256 else 8)
    return base


def available_ram_bytes() -> int:
    """Use the tighter of host availability and cgroup headroom."""
    try:
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        available = int(memory["MemAvailable"].split()[0]) * 1024
        limit_path = Path("/sys/fs/cgroup/memory.max")
        usage_path = Path("/sys/fs/cgroup/memory.current")
        if limit_path.is_file() and usage_path.is_file():
            limit = limit_path.read_text().strip()
            if limit != "max":
                available = min(available, max(0, int(limit) - int(usage_path.read_text().strip())))
        return available
    except (OSError, KeyError, ValueError):
        return 0


def cpu_batch_for(training_samples: int, available_ram: int, longest_utterance_seconds: float) -> int:
    """Conservative memory and sequence-length policy for CPU Auto mode."""
    gib = 1024 ** 3
    if training_samples >= 16 and available_ram >= 24 * gib and longest_utterance_seconds <= 8:
        return 16
    if training_samples >= 8 and available_ram >= 12 * gib and longest_utterance_seconds <= 15:
        return 8
    return min(4, training_samples)


def cpu_threads_for(device: str, physical_cores: int, logical_cores: int, workers: int) -> int:
    cores = max(1, min(physical_cores or logical_cores, logical_cores))
    return max(1, cores - workers) if device == "cpu" else max(1, min(2, cores - workers))


def select_batch_size(trials: Sequence[tuple[int, str, float | None]], total_vram: float | None = None,
                      margin: float = GPU_MEMORY_MARGIN) -> int:
    if not 0 < margin < 1:
        raise ValueError("GPU memory margin must be between 0 and 1")
    safe = [size for size, outcome, peak in trials if outcome == "pass" and (
        total_vram is None or peak is not None and peak <= total_vram * margin)]
    if not safe:
        raise ValueError("No batch size passed the training probe with enough memory headroom")
    return max(safe)


def estimate_runtime(step_seconds: Sequence[float], completed_batches: int, total_batches: int,
                     hourly_rate: float | None, warmup_batches: int = 5, minimum_measured: int = 10,
                     elapsed_seconds: float | None = None) -> dict[str, float] | None:
    measured = [value for value in step_seconds[max(warmup_batches, len(step_seconds) - 50):] if value > 0]
    if len(measured) < minimum_measured or completed_batches < warmup_batches + minimum_measured or total_batches <= 0:
        return None
    average = sum(measured) / len(measured)
    remaining = max(0, total_batches - completed_batches) * average
    result = {"seconds_per_batch": average, "remaining_seconds": remaining,
              "estimated_finish_at": (datetime.now(timezone.utc) + timedelta(seconds=remaining)).timestamp()}
    if hourly_rate is not None:
        result["remaining_cost"] = remaining / 3600 * hourly_rate
        result["total_cost"] = ((elapsed_seconds if elapsed_seconds is not None else completed_batches * average) + remaining) / 3600 * hourly_rate
    return result


def gpu_snapshot() -> dict[str, Any] | None:
    """Prefer NVML; nvidia-smi is a safe fallback when its Python binding is absent."""
    try:
        import pynvml  # type: ignore[import-not-found]
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            return {"name": name.decode() if isinstance(name, bytes) else name, "compute_percent": utilization.gpu,
                    "memory_percent": utilization.memory, "memory_used_bytes": memory.used,
                    "memory_total_bytes": memory.total, "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, 0),
                    "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                    "power_limit_w": pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000,
                    "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, 1),
                    "memory_clock_mhz": pynvml.nvmlDeviceGetClockInfo(handle, 2)}
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass
    fields = "name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.mem"
    try:
        result = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits", "--id=0"],
                                capture_output=True, text=True, timeout=3, check=True)
        parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        def number(value: str) -> float | None:
            try:
                return float(value)
            except ValueError:
                return None
        return {"name": parts[0], "compute_percent": number(parts[1]), "memory_percent": number(parts[2]),
                "memory_used_bytes": number(parts[3]) * 1024**2 if number(parts[3]) is not None else None,
                "memory_total_bytes": number(parts[4]) * 1024**2 if number(parts[4]) is not None else None,
                "temperature_c": number(parts[5]), "power_w": number(parts[6]), "power_limit_w": number(parts[7]),
                "sm_clock_mhz": number(parts[8]), "memory_clock_mhz": number(parts[9])}
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def cpu_snapshot(pid: int, previous: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read Linux counters; percentages are available after a second sample."""
    try:
        lines = Path("/proc/stat").read_text().splitlines()
        counters = {line.split()[0]: [int(item) for item in line.split()[1:]] for line in lines if line.startswith("cpu")}
        allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        process_ticks = int(stat[11]) + int(stat[12])
        memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        total = int(memory["MemTotal"].split()[0]) * 1024
        available = int(memory["MemAvailable"].split()[0]) * 1024
        def percent(name: str) -> float | None:
            if not previous or name not in previous["counters"]:
                return None
            before, after = previous["counters"][name], counters[name]
            span = sum(after) - sum(before)
            idle = after[3] + after[4] - before[3] - before[4]
            return round(100 * (span - idle) / span, 1) if span else None
        elapsed = (datetime.now(timezone.utc).timestamp() - previous["at"]) if previous else 0
        cgroup_stat = Path("/sys/fs/cgroup/cpu.stat")
        usage = next((int(line.split()[1]) for line in cgroup_stat.read_text().splitlines() if line.startswith("usage_usec ")), None) if cgroup_stat.is_file() else None
        cgroup_percent = (round(100 * (usage - previous["usage"]) / 1_000_000 / elapsed / effective_cpu_count(), 1)
                          if previous and usage is not None and previous.get("usage") is not None and elapsed > 0 else None)
        process_percent = (100 * (process_ticks - previous["process_ticks"]) / os.sysconf("SC_CLK_TCK") / elapsed
                           if previous and elapsed > 0 else None)
        state = {"counters": counters, "process_ticks": process_ticks, "usage": usage, "at": datetime.now(timezone.utc).timestamp()}
        return {"total_percent": cgroup_percent if cgroup_percent is not None else percent("cpu"),
                "per_core_percent": {str(i): percent(f"cpu{i}") for i in allowed},
                "process_percent": round(process_percent, 1) if process_percent is not None else None,
                "logical_cores": effective_cpu_count(), "physical_cores": physical_cpu_count(),
                "ram_used_bytes": total - available, "ram_total_bytes": total}, state
    except (OSError, KeyError, ValueError, IndexError):
        return {"logical_cores": effective_cpu_count(), "physical_cores": physical_cpu_count()}, {}
