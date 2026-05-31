"""Online evaluation: track user satisfaction from feedback signals."""

import logging
import threading
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class OnlineEvaluator:
    """Collect and aggregate user feedback metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ratings: dict[str, dict[str, int]] = defaultdict(lambda: {"up": 0, "down": 0})
        self._confidence_ratings: dict[str, dict[str, int]] = defaultdict(lambda: {"up": 0, "down": 0})
        self._hourly_buckets: dict[str, dict[int, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"up": 0, "down": 0})
        )

    def record_feedback(self, rating: str, confidence_level: str = "high", chat_id: str = None):
        """Record a user feedback event.

        Args:
            rating: 'up' or 'down'
            confidence_level: 'high', 'low', or 'none'
            chat_id: Optional chat identifier for hourly breakdown
        """
        with self._lock:
            self._ratings["overall"][rating] += 1
            self._confidence_ratings[confidence_level][rating] += 1

            if chat_id:
                hour_bucket = int(time.time() // 3600)
                self._hourly_buckets[chat_id][hour_bucket][rating] += 1

    def get_satisfaction_rate(self, confidence_level: str = None) -> dict:
        """Get satisfaction rate, optionally filtered by confidence level."""
        with self._lock:
            if confidence_level:
                counts = self._confidence_ratings.get(confidence_level, {"up": 0, "down": 0})
            else:
                counts = self._ratings.get("overall", {"up": 0, "down": 0})

            total = counts["up"] + counts["down"]
            rate = counts["up"] / total if total > 0 else 0.0

            return {
                "satisfaction_rate": round(rate, 4),
                "upvotes": counts["up"],
                "downvotes": counts["down"],
                "total": total,
            }

    def get_all_metrics(self) -> dict:
        """Get all satisfaction metrics."""
        return {
            "overall": self.get_satisfaction_rate(),
            "high_confidence": self.get_satisfaction_rate("high"),
            "low_confidence": self.get_satisfaction_rate("low"),
            "none_confidence": self.get_satisfaction_rate("none"),
        }


# Global singleton
online_eval = OnlineEvaluator()
