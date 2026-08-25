"""A simple online accept/reject sampler that stops a single feature value
(one language, one domain bucket, one volume type, ...) from dominating a
target-sized sample, without any real balancing/stratification -- exactly
"caps, not sophisticated balancing" as requested.

For each tracked dimension, a per-value cap is computed as
`ceil(target_size * cap_fraction)`. A candidate is accepted only if every
one of its dimension values is currently under its cap; accepting it then
increments all those counters. Dimensions are independent of each other --
e.g. a language cap and a domain cap don't interact, so it's possible (by
design) to end up with, say, 40 German books spread across many different
domain buckets rather than exactly-even coverage of every combination.
"""

import math
from dataclasses import dataclass, field


@dataclass
class DiversitySampler:
    target_size: int
    cap_fractions: dict[str, float]  # dimension name -> max fraction of target_size per value
    _counts: dict[str, dict[str, int]] = field(default_factory=dict)
    _accepted: int = 0

    def __post_init__(self):
        for dimension in self.cap_fractions:
            self._counts.setdefault(dimension, {})

    def _cap_for(self, dimension: str) -> int:
        return max(1, math.ceil(self.target_size * self.cap_fractions[dimension]))

    def is_full(self) -> bool:
        return self._accepted >= self.target_size

    def would_accept(self, features: dict[str, str]) -> bool:
        for dimension, cap_fraction in self.cap_fractions.items():
            value = features.get(dimension)
            if value is None:
                continue
            if self._counts[dimension].get(value, 0) >= self._cap_for(dimension):
                return False
        return True

    def offer(self, features: dict[str, str]) -> bool:
        """Returns True and records the item iff every capped dimension has
        room for it and the sampler isn't already full."""
        if self.is_full() or not self.would_accept(features):
            return False
        for dimension in self.cap_fractions:
            value = features.get(dimension)
            if value is not None:
                self._counts[dimension][value] = self._counts[dimension].get(value, 0) + 1
        self._accepted += 1
        return True

    @property
    def accepted_count(self) -> int:
        return self._accepted

    def counts_summary(self) -> dict[str, dict[str, int]]:
        return {dim: dict(counts) for dim, counts in self._counts.items()}
