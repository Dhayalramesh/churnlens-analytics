"""Small data-structure and algorithm building blocks used by the project (and unit-tested).

- ``TopK``: a bounded min-heap that keeps the k highest-scoring items in O(n log k) time and O(k) memory.
- ``bucketize``: maps a number to a labelled range with binary search (bisect), O(log b).
"""
import heapq
from bisect import bisect_left
from collections.abc import Iterable, Sequence
from itertools import count
from typing import Any


class TopK:
    """Keep the ``k`` items with the highest scores while streaming through the data once."""

    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = k
        self._heap: list[tuple[float, int, Any]] = []   # min-heap: the smallest kept score sits at index 0
        self._tie = count()                              # tie-breaker so items themselves are never compared

    def push(self, score: float, item: Any) -> None:
        entry = (score, next(self._tie), item)
        if len(self._heap) < self.k:
            heapq.heappush(self._heap, entry)
        elif score > self._heap[0][0]:
            heapq.heapreplace(self._heap, entry)

    def extend(self, pairs: Iterable[tuple[float, Any]]) -> "TopK":
        for score, item in pairs:
            self.push(score, item)
        return self

    def items(self) -> list[tuple[float, Any]]:
        """Highest score first."""
        return [(s, it) for s, _, it in sorted(self._heap, key=lambda e: (-e[0], e[1]))]

    def __len__(self) -> int:
        return len(self._heap)


def bucketize(value: float, upper_bounds: Sequence[float], labels: Sequence[str]) -> str:
    """Return the label of the first range whose (inclusive) upper bound is >= value.

    ``upper_bounds`` must be sorted ascending and ``labels`` must have one more entry than ``upper_bounds``
    (the last label catches everything above the final bound).
    """
    if len(labels) != len(upper_bounds) + 1:
        raise ValueError("labels must have exactly one more entry than upper_bounds")
    return labels[bisect_left(upper_bounds, value)]
