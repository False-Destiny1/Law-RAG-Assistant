"""Lightweight A/B testing framework for RAG experiments."""

import hashlib
import logging
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class ABTestManager:
    """Deterministic user assignment to experiment variants with metric tracking."""

    def __init__(self):
        self._experiments: dict[str, dict] = {}
        self._results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._lock = threading.Lock()

    def register_experiment(self, name: str, variants: list[str], traffic_pct: float = 100.0):
        """Register an experiment with its variants.

        Args:
            name: Experiment identifier
            variants: List of variant names (e.g. ["control", "treatment"])
            traffic_pct: Percentage of users to include (0-100)
        """
        with self._lock:
            self._experiments[name] = {
                "variants": variants,
                "traffic_pct": traffic_pct,
                "created_at": time.time(),
            }
        logger.info(f"A/B experiment registered: {name} variants={variants} traffic={traffic_pct}%")

    def assign_variant(self, user_id: int, experiment: str) -> str | None:
        """Deterministically assign a user to a variant.

        Returns None if user is outside the experiment traffic.
        """
        exp = self._experiments.get(experiment)
        if not exp:
            return None

        # Deterministic hash-based assignment
        hash_input = f"{user_id}:{experiment}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 100

        if hash_val >= exp["traffic_pct"]:
            return None  # Outside experiment traffic

        # Assign to variant based on hash
        variant_idx = hash_val % len(exp["variants"])
        return exp["variants"][variant_idx]

    def log_metric(self, experiment: str, variant: str, metric: str, value: float):
        """Record a metric observation for an experiment variant."""
        with self._lock:
            key = f"{experiment}:{variant}"
            self._results[key][metric].append(value)

    def get_results(self, experiment: str) -> dict:
        """Get aggregated results for an experiment.

        Returns: {variant: {metric: {mean, count, p50, p90}}}
        """
        exp = self._experiments.get(experiment)
        if not exp:
            return {}

        results = {}
        for variant in exp["variants"]:
            key = f"{experiment}:{variant}"
            variant_metrics = {}
            with self._lock:
                for metric, values in self._results.get(key, {}).items():
                    if not values:
                        continue
                    sorted_vals = sorted(values)
                    variant_metrics[metric] = {
                        "mean": sum(values) / len(values),
                        "count": len(values),
                        "p50": sorted_vals[len(sorted_vals) // 2],
                        "p90": sorted_vals[int(len(sorted_vals) * 0.9)],
                    }
            results[variant] = variant_metrics
        return results


# Global singleton
ab_manager = ABTestManager()
