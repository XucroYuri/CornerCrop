"""Adaptive batch processing primitives for large image sets."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Callable

try:
    import psutil  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without psutil
    psutil = None


@dataclass(frozen=True)
class ResourceSnapshot:
    """A point-in-time snapshot of relevant machine resource usage."""

    cpu_percent: float
    memory_percent: float
    read_mbps: float
    write_mbps: float
    load_ratio: float


@dataclass(frozen=True)
class AdaptiveParallelismConfig:
    """Config for adaptive worker scheduling on heterogeneous Macs."""

    enabled: bool = True
    min_workers: int = 1
    max_workers: int | None = None
    poll_interval: float = 2.0
    cpu_low_water: float = 55.0
    cpu_high_water: float = 85.0
    memory_low_water: float = 70.0
    memory_high_water: float = 85.0
    read_low_mbps: float = 150.0
    read_high_mbps: float = 400.0
    write_low_mbps: float = 60.0
    write_high_mbps: float = 180.0
    progress_interval: int = 25
    heartbeat_interval: float = 30.0


class ResourceMonitor(threading.Thread):
    """Background sampler for CPU, memory, and disk throughput."""

    def __init__(self, poll_interval: float):
        super().__init__(daemon=True)
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._snapshot = ResourceSnapshot(0.0, 0.0, 0.0, 0.0, 0.0)
        self._last_disk = _disk_io_counters()
        self._last_time = time.monotonic()
        _cpu_percent()

    def run(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            now = time.monotonic()
            elapsed = max(now - self._last_time, 1e-6)
            disk = _disk_io_counters()
            if self._last_disk and disk:
                read_mbps = (disk.read_bytes - self._last_disk.read_bytes) / elapsed / (1024 * 1024)
                write_mbps = (disk.write_bytes - self._last_disk.write_bytes) / elapsed / (1024 * 1024)
            else:
                read_mbps = 0.0
                write_mbps = 0.0

            logical_cpus = max(_cpu_count(logical=True) or 1, 1)
            load_ratio = os.getloadavg()[0] / logical_cpus if hasattr(os, "getloadavg") else 0.0
            snapshot = ResourceSnapshot(
                cpu_percent=_cpu_percent(),
                memory_percent=_memory_percent(),
                read_mbps=read_mbps,
                write_mbps=write_mbps,
                load_ratio=load_ratio,
            )
            with self._lock:
                self._snapshot = snapshot
            self._last_disk = disk
            self._last_time = now

    def latest(self) -> ResourceSnapshot:
        with self._lock:
            return self._snapshot

    def stop(self) -> None:
        self._stop_event.set()


def recommend_worker_cap(config: AdaptiveParallelismConfig) -> int:
    """Choose a conservative-but-useful worker cap for the current Mac."""
    if config.max_workers is not None:
        return max(config.min_workers, config.max_workers)

    physical = _cpu_count(logical=False) or 4
    logical = _cpu_count(logical=True) or physical
    mem_gb = _memory_total_bytes() / (1024**3)
    cpu_cap = max(2, min(physical, logical - 1 if logical > 1 else 1))
    mem_cap = max(2, int(mem_gb // 2))
    return max(config.min_workers, min(cpu_cap, mem_cap, 8))


def recommend_target_workers(
    current_target: int,
    config: AdaptiveParallelismConfig,
    snapshot: ResourceSnapshot,
    backlog_exists: bool,
) -> int:
    """Adjust target concurrency based on current hardware pressure."""
    max_workers = recommend_worker_cap(config)
    min_workers = max(1, min(config.min_workers, max_workers))
    if not config.enabled:
        return max_workers

    overloaded = (
        snapshot.cpu_percent >= config.cpu_high_water
        or snapshot.memory_percent >= config.memory_high_water
        or snapshot.read_mbps >= config.read_high_mbps
        or snapshot.write_mbps >= config.write_high_mbps
        or snapshot.load_ratio >= 0.95
    )
    healthy_headroom = (
        backlog_exists
        and snapshot.cpu_percent <= max(config.cpu_low_water, config.cpu_high_water - 12.0)
        and snapshot.memory_percent <= max(config.memory_low_water, config.memory_high_water - 4.0)
        and snapshot.read_mbps <= max(config.read_low_mbps, config.read_high_mbps * 0.75)
        and snapshot.write_mbps <= max(config.write_low_mbps, config.write_high_mbps * 0.75)
        and snapshot.load_ratio <= 0.85
    )

    if overloaded:
        return max(min_workers, current_target - 1)
    if healthy_headroom and current_target < max_workers:
        return min(max_workers, current_target + 1)
    return current_target


def process_batch(
    items: list[Any],
    worker_fn: Callable[[Any], Any],
    config: AdaptiveParallelismConfig,
    progress_callback: Callable[[int, int, int, ResourceSnapshot], None] | None = None,
) -> list[Any]:
    """Run a batch with adaptive concurrency and resource-aware backpressure."""
    if not items:
        return []

    max_workers = recommend_worker_cap(config)
    initial_target = max(config.min_workers, min(max_workers, max(1, max_workers // 2)))
    monitor = ResourceMonitor(config.poll_interval)
    monitor.start()

    results: list[Any | None] = [None] * len(items)
    in_flight: dict[Future, int] = {}
    next_index = 0
    completed = 0
    target_workers = initial_target
    last_progress_callback_at = time.monotonic()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            while next_index < len(items) or in_flight:
                snapshot = monitor.latest()
                target_workers = recommend_target_workers(
                    target_workers,
                    config,
                    snapshot,
                    backlog_exists=next_index < len(items),
                )

                while next_index < len(items) and len(in_flight) < target_workers:
                    future = executor.submit(worker_fn, items[next_index])
                    in_flight[future] = next_index
                    next_index += 1

                if not in_flight:
                    continue

                done, _ = wait(
                    in_flight.keys(),
                    timeout=config.poll_interval,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    now = time.monotonic()
                    if (
                        progress_callback
                        and config.heartbeat_interval > 0
                        and now - last_progress_callback_at >= config.heartbeat_interval
                    ):
                        progress_callback(
                            completed,
                            len(items),
                            target_workers,
                            monitor.latest(),
                        )
                        last_progress_callback_at = now
                    continue

                for future in done:
                    index = in_flight.pop(future)
                    results[index] = future.result()
                    completed += 1
                    if (
                        progress_callback
                        and (
                            completed % max(1, config.progress_interval) == 0
                            or completed == len(items)
                        )
                    ):
                        progress_callback(
                            completed,
                            len(items),
                            target_workers,
                            monitor.latest(),
                        )
                        last_progress_callback_at = time.monotonic()
    finally:
        monitor.stop()
        monitor.join(timeout=1.0)

    return [result for result in results if result is not None]


def _cpu_count(logical: bool) -> int | None:
    if psutil is not None:
        return psutil.cpu_count(logical=logical)
    if logical:
        return os.cpu_count()
    return os.cpu_count()


def _cpu_percent() -> float:
    if psutil is not None:
        return psutil.cpu_percent(interval=None)
    return 0.0


def _memory_percent() -> float:
    if psutil is not None:
        return psutil.virtual_memory().percent
    return 0.0


def _memory_total_bytes() -> int:
    if psutil is not None:
        return psutil.virtual_memory().total
    if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return 8 * 1024**3


def _disk_io_counters():
    if psutil is not None:
        return psutil.disk_io_counters()
    return None
