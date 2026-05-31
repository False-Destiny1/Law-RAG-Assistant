"""Lightweight in-memory metrics collector (Prometheus text format compatible)."""

import threading
import time
from collections import defaultdict


class _Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._start_time = time.time()

    def inc(self, name: str, value: float = 1.0):
        with self._lock:
            self._counters[name] += value

    def observe(self, name: str, value: float):
        with self._lock:
            self._histograms[name].append(value)

    def _quantile(self, data: list[float], q: float) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = int(len(sorted_data) * q)
        return sorted_data[min(idx, len(sorted_data) - 1)]

    def render(self) -> str:
        """Render metrics in Prometheus text format."""
        lines = []
        with self._lock:
            for name, value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")
            for name, values in sorted(self._histograms.items()):
                if not values:
                    continue
                lines.append(f"# TYPE {name} summary")
                lines.append(f'{name}{{quantile="0.5"}} {self._quantile(values, 0.5)}')
                lines.append(f'{name}{{quantile="0.9"}} {self._quantile(values, 0.9)}')
                lines.append(f'{name}{{quantile="0.99"}} {self._quantile(values, 0.99)}')
                lines.append(f"{name}_sum {sum(values)}")
                lines.append(f"{name}_count {len(values)}")
            uptime = time.time() - self._start_time
            lines.append("# TYPE process_uptime_seconds gauge")
            lines.append(f"process_uptime_seconds {uptime:.1f}")
        return "\n".join(lines) + "\n"


metrics = _Metrics()
