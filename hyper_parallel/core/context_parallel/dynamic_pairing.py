# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Immutable cost pairing for local selection of folded sequence chunks."""
from functools import lru_cache
import math


def _folded_owners(total: int, world: int) -> tuple:
    """Natural intervals for the existing two-chunk folded input layout."""
    if world < 1 or total <= 0 or total % (2 * world):
        raise ValueError("Sequence length must be positive and divisible by twice the shard count.")
    width = total // (2 * world)
    return tuple(tuple((chunk * width, (chunk + 1) * width) for chunk in
                       (2 * rank, 2 * world - 1 - 2 * rank)) for rank in range(world))


def validate_pairing_cost(coefficients: tuple, topk: int) -> None:
    """Validate the explicit five-feature model and sparse selection width."""
    if (len(coefficients) != 5 or any(not math.isfinite(x) or x < 0 for x in coefficients)
            or not any(coefficients)):
        raise ValueError("Dynamic pairing requires five finite nonnegative costs, at least one positive.")
    if isinstance(topk, bool) or not isinstance(topk, int) or topk <= 0:
        raise ValueError("Dynamic pairing topk must be a positive integer.")


@lru_cache(maxsize=128)
def paired_owners(ends: tuple, world: int, topk: int, coefficients: tuple) -> tuple:
    """Pair light/heavy equal Q chunks, retaining fixed folding unless peak cost falls.

    Features are calls, dense pairs/1e9, sparse pairs/1e6, physical K/262144,
    and active document prefixes/262144. All ranks use the same host metadata.

    Args:
        ends: Tuple of global cumulative document ends. Repeated padding ends
            and empty documents are allowed.
        world: Number of sequence shards, including sequence-parallel TP.
        topk: Positive sparse selection width.
        coefficients: Five nonnegative costs in the feature order above.

    Returns:
        Tuple of per-rank pairs of natural half-open token intervals. Each rank
        retains two equal-sized chunks. The caller selects these intervals at
        model entry and supplies their chunk IDs through the fold metadata.
    """
    validate_pairing_cost(coefficients, topk)
    if not ends or any(a > b for a, b in zip((0,) + ends, ends)):
        raise ValueError("Pairing requires nondecreasing cumulative document ends.")
    original = _folded_owners(ends[-1], world)
    width = ends[-1] // (2 * world)
    chunks = [(i * width, (i + 1) * width) for i in range(2 * world)]
    documents = tuple((left, right) for left, right in zip((0,) + ends[:-1], ends) if left < right)
    scores = []
    for start, stop in chunks:
        dense = sparse = active = 0
        for left, right in documents:
            low, high = max(start, left), min(stop, right)
            if low >= high:
                continue
            begin, end = low - left, high - left
            dense += (end * (end + 1) - begin * (begin + 1)) / 2
            capped_begin, capped_end = min(begin, topk), min(end, topk)
            sparse += (capped_end * (capped_end + 1) - capped_begin * (capped_begin + 1)) / 2
            sparse += topk * (max(end - topk, 0) - max(begin - topk, 0))
            active += end
        features = (1, dense / 1e9, sparse / 1e6, stop / 262144, active / 262144)
        scores.append(sum(value * weight for value, weight in zip(features, coefficients)))
    order = sorted(range(2 * world), key=lambda index: (scores[index], index))
    pairs = [tuple(sorted((order[i], order[-1 - i]))) for i in range(world)]
    old_peak = max(scores[2 * i] + scores[2 * world - 1 - 2 * i] for i in range(world))
    if max(scores[a] + scores[b] for a, b in pairs) >= old_peak:
        return original
    return _place_pairs(pairs, original, chunks)


def _place_pairs(pairs: list, original: tuple, chunks: list) -> tuple:
    """Match each pair to an original owner, preserving at least one of its chunks.

    Both partitions have two chunks per rank, so the pair/owner bipartite graph
    has degree two (counting parallel edges) and admits a perfect matching.
    Pairs already co-located retain both chunks and are isolated components.
    """
    owners = {interval: rank for rank, row in enumerate(original) for interval in row}
    candidates = [tuple(sorted({owners[chunks[index]] for index in pair})) for pair in pairs]
    assigned = {}

    def place(pair_id: int, visited: set) -> bool:
        """Assign a pair through an augmenting path in the owner graph."""
        for rank in candidates[pair_id]:
            if rank in visited:
                continue
            visited.add(rank)
            previous = assigned.get(rank)
            if previous is None or place(previous, visited):
                assigned[rank] = pair_id
                return True
        return False

    for pair_id in range(len(pairs)):
        if not place(pair_id, set()):
            raise ValueError("Unable to assign paired chunks to their original owners.")
    result = []
    for rank, original_intervals in enumerate(original):
        intervals = tuple(chunks[index] for index in pairs[assigned[rank]])
        result.append(original_intervals if set(intervals) == set(original_intervals) else intervals)
    return tuple(result)
