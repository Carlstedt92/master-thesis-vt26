"""Shard-locality-aware batch sampling for precomputed shard datasets.

Sampling batches uniformly at random across the full flat index space forces
each batch to touch nearly every shard (a batch of 512 drawn from 9.4M items
across 189 shards statistically hits ~176 distinct shards), which thrashes
any per-worker shard cache that's smaller than the whole corpus, regardless
of storage speed. This sampler keeps consecutive batches inside a single
shard: shard order is shuffled each epoch, sample order within a shard is
shuffled, and a worker only needs one or two shards resident at a time.

Under DDP (world_size > 1), shards are additionally partitioned across ranks
each epoch (same shuffled order on every rank, then a `rank::world_size`
stride) so each GPU trains on a disjoint slice of the data instead of every
rank redundantly processing the whole dataset.
"""

import random
from typing import Iterator, List, Optional, Sequence, Tuple

from torch.utils.data import Sampler


class ShardAwareBatchSampler(Sampler):
    """Yields batches of indices drawn one shard at a time."""

    def __init__(
        self,
        cumulative_sizes: Sequence[int],
        shard_ids: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        seed: Optional[int] = None,
        rank: int = 0,
        world_size: int = 1,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank must be in [0, world_size); got rank={rank}, world_size={world_size}")
        self.cumulative_sizes = list(cumulative_sizes)
        self.shard_ids = list(shard_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Vary shuffling across epochs while staying reproducible from config.seed."""
        self._epoch = epoch

    def _shard_range(self, shard_id: int) -> Tuple[int, int]:
        start = self.cumulative_sizes[shard_id - 1] if shard_id > 0 else 0
        end = self.cumulative_sizes[shard_id]
        return start, end

    def _rank_shards_for_epoch(self, rng: random.Random) -> List[int]:
        """Shuffle the full shard order (identically derived on every rank via the
        same seed+epoch), then take this rank's disjoint stride. Every rank computes
        the same shuffled order independently -- no cross-rank communication needed."""
        shard_order = list(self.shard_ids)
        if self.shuffle:
            rng.shuffle(shard_order)
        if self.world_size > 1:
            shard_order = shard_order[self.rank::self.world_size]
        return shard_order

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self._epoch) if self.seed is not None else random.Random()
        rank_shards = self._rank_shards_for_epoch(rng)

        batches: List[List[int]] = []
        for shard_id in rank_shards:
            start, end = self._shard_range(shard_id)
            indices = list(range(start, end))
            if self.shuffle:
                rng.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                batches.append(indices[offset:offset + self.batch_size])

        if self.world_size > 1:
            # DDP requires every rank to call the exact same sequence of
            # collectives (DDP gradient all-reduce, SyncBatchNorm all-gather,
            # DINOLoss center all-reduce, ...) the same number of times per
            # epoch. Per-rank batch counts naturally vary by a handful of
            # batches (shards don't divide evenly across ranks/batch_size),
            # and without this, ranks silently fall out of lockstep -- one
            # rank's Nth collective call pairs with a different rank's
            # (N-1)th, permanently desyncing the process group until NCCL
            # times out. len(self) is a target derived only from rank-
            # invariant globals (total batches, world_size), so every rank
            # can pad/trim to it independently with no extra communication.
            target_len = len(self)
            if len(batches) > target_len:
                batches = batches[:target_len]
            elif len(batches) < target_len and batches:
                pad_needed = target_len - len(batches)
                batches = batches + [batches[i % len(batches)] for i in range(pad_needed)]

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Approximate batch count for this rank (total batches / world_size).

        Not recomputed per-rank-assignment: which specific shards a rank gets
        can shift by epoch, but the *count* of shards per rank is stable
        (a fixed stride of a fixed-length list), so this stays accurate to
        within a batch or two -- fine for schedule-length purposes.
        """
        total_batches = 0
        for shard_id in self.shard_ids:
            start, end = self._shard_range(shard_id)
            shard_len = end - start
            total_batches += (shard_len + self.batch_size - 1) // self.batch_size
        if self.world_size <= 1:
            return total_batches
        return -(-total_batches // self.world_size)  # ceil division
