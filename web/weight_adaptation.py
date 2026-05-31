"""Adaptive retrieval weight learning based on user feedback."""

import logging
import threading

logger = logging.getLogger(__name__)


class WeightAdapter:
    """Learn optimal retrieval weights from user feedback signals.

    When a user gives thumbs up, the retrieval sources that contributed
    to the answer get their weights increased. Thumbs down decreases them.
    Weights are normalized after each update.
    """

    LEARNING_RATE = 0.05  # 5% adjustment per feedback event
    MIN_WEIGHT = 0.05     # Floor to prevent any source from being silenced

    def __init__(self, initial_weights: dict[str, float] = None):
        self._weights = initial_weights or {"vector": 0.4, "bm25": 0.3, "graph": 0.3}
        self._lock = threading.Lock()
        self._update_count = 0

    @property
    def weights(self) -> dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def update_from_feedback(self, retrieved_sources: list[str], feedback: str):
        """Adjust weights based on which retrieval sources contributed and user feedback.

        Args:
            retrieved_sources: List of source types that contributed (e.g. ["vector", "bm25"])
            feedback: "up" or "down"
        """
        if not retrieved_sources:
            return

        adjustment = self.LEARNING_RATE if feedback == "up" else -self.LEARNING_RATE

        with self._lock:
            for source in self._weights:
                if source in retrieved_sources:
                    self._weights[source] += adjustment
                else:
                    # Sources not used get a small penalty (encourage diversity)
                    self._weights[source] -= adjustment * 0.3

            # Enforce minimum weight
            for source in self._weights:
                self._weights[source] = max(self._weights[source], self.MIN_WEIGHT)

            # Normalize to sum to 1.0
            total = sum(self._weights.values())
            if total > 0:
                self._weights = {k: v / total for k, v in self._weights.items()}

            self._update_count += 1

        logger.info(
            f"权重自适应更新 (第{self._update_count}次, feedback={feedback}): "
            f"{', '.join(f'{k}={v:.3f}' for k, v in self._weights.items())}"
        )

    def get_weights(self) -> dict[str, float]:
        """Return current weights (read-only snapshot)."""
        return self.weights


# Global singleton (initialized with default weights, updated at runtime)
weight_adapter = WeightAdapter()
