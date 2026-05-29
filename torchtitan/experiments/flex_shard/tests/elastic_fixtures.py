# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test fixtures for the elastic-shrink rewrite (new FlexShard API, CUDA/NCCL).

Ported from the original (gloo-based) fixtures and adapted to the modular
FlexShard package, which mandates a CUDA mesh — so the torchft PG is NCCL.

- ``SyntheticQuorumResult`` / ``make_synthetic_quorum_result`` — a Python
  stand-in for torchft's PyO3 ``QuorumResult`` (which can't be constructed from
  Python).
- ``FakeManager`` — a ``torchft.Manager`` stand-in whose ``start_quorum``
  reconfigures the underlying ``ProcessGroupWrapper`` directly, bypassing
  Lighthouse. PG-agnostic: works with any wrapper exposing ``.configure``.
- ``make_torchft_nccl_pg`` — build + register + configure a torchft
  ``ProcessGroupNCCL`` so ``_c10d_functional`` collectives resolve it by
  ``group_name``.

NOTE: This module imports torchft lazily so the experiment stays importable
without it. Tests using these fixtures must skip when torchft is absent.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Synthetic QuorumResult
# ---------------------------------------------------------------------------


@dataclass
class SyntheticQuorumResult:
    """Mirrors the subset of ``torchft._torchft.QuorumResult`` we need."""

    quorum_id: int
    replica_rank: int
    replica_world_size: int
    store_address: str
    ranks_in_quorum: list[int]


def make_synthetic_quorum_result(
    new_global_ranks: list[int],
    my_new_rank: int,
    store_addr: str,
    quorum_id: int = 1,
) -> SyntheticQuorumResult:
    """Build a QuorumResult-shaped object for the post-shrink world."""
    return SyntheticQuorumResult(
        quorum_id=quorum_id,
        replica_rank=my_new_rank,
        replica_world_size=len(new_global_ranks),
        store_address=store_addr,
        ranks_in_quorum=list(new_global_ranks),
    )


# ---------------------------------------------------------------------------
# FakeManager
# ---------------------------------------------------------------------------


class FakeManager:
    """Minimal stand-in for ``torchft.Manager``.

    Tests pre-seed the ``QuorumResult`` each rank receives; ``start_quorum``
    reconfigures the underlying PG directly. Spies (``shutdown_calls`` /
    ``start_quorum_calls``) let tests assert ``shrink_flex_shard`` call-site
    behavior.
    """

    def __init__(
        self,
        replica_id: str,
        pg: Any,
        pending_quorum: SyntheticQuorumResult | None = None,
    ) -> None:
        self._replica_id = replica_id
        self._pg = pg
        self._pending_quorum = pending_quorum
        self._last_quorum: SyntheticQuorumResult | None = None
        self.shutdown_calls = 0
        self.start_quorum_calls: list[dict[str, Any]] = []
        self._shutdown_flag = False

    def seed_quorum(self, qr: SyntheticQuorumResult) -> None:
        self._pending_quorum = qr

    def start_quorum(
        self,
        *,
        shrink_only: bool = False,
        allow_heal: bool = True,
        timeout: timedelta = timedelta(seconds=30),
    ) -> SyntheticQuorumResult:
        """Reconfigure the PG to the pre-seeded quorum and return it.

        Real torchft returns None and stores the quorum internally; we return
        it so ``shrink_flex_shard`` can read ``ranks_in_quorum`` without poking
        at manager privates.
        """
        assert self._pending_quorum is not None, (
            "FakeManager.start_quorum called without a seeded QuorumResult"
        )
        qr = self._pending_quorum
        self.start_quorum_calls.append(
            {"shrink_only": shrink_only, "allow_heal": allow_heal, "timeout": timeout}
        )
        store_prefixed_addr = f"{qr.store_address}/shrink/{qr.quorum_id}"
        self._pg.configure(
            store_prefixed_addr,
            self._replica_id,
            qr.replica_rank,
            qr.replica_world_size,
            qr.quorum_id,
            qr.replica_rank,
            qr.replica_world_size,
            qr.ranks_in_quorum,
        )
        self._last_quorum = qr
        self._pending_quorum = None
        return qr

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self._shutdown_flag = True

    def num_participants(self) -> int:
        if self._last_quorum is not None:
            return self._last_quorum.replica_world_size
        return self._pg.size()

    @property
    def pg(self) -> Any:
        return self._pg

    @property
    def replica_id(self) -> str:
        return self._replica_id


# ---------------------------------------------------------------------------
# torchft ProcessGroupNCCL builder
# ---------------------------------------------------------------------------


_PG_NAME_COUNTER = 0
_PG_NAME_LOCK = threading.Lock()


def _unique_pg_name(prefix: str = "elastic_test") -> str:
    global _PG_NAME_COUNTER
    with _PG_NAME_LOCK:
        _PG_NAME_COUNTER += 1
        return f"{prefix}_{os.getpid()}_{_PG_NAME_COUNTER}_{uuid.uuid4().hex[:8]}"


def make_torchft_nccl_pg(
    store_addr: str,
    rank: int,
    world_size: int,
    global_ranks: list[int] | None = None,
    *,
    replica_id: str = "0",
    quorum_id: int = 0,
    name: str | None = None,
    timeout: timedelta = timedelta(seconds=30),
):
    """Build a torchft ``ProcessGroupNCCL``, register it, and configure it.

    ``.register(name)`` installs the PG in the global registry so the
    ``_c10d_functional`` collectives FlexShard uses can resolve it by
    ``group_name``. That side effect is the point — the returned object is the
    wrapper ``shrink_flex_shard`` will detect and reconfigure.
    """
    from torchft.process_group import ProcessGroupNCCL

    pg = ProcessGroupNCCL(timeout=timeout)
    pg.register(name or _unique_pg_name())
    pg.configure(
        store_addr,
        replica_id,
        rank,
        world_size,
        quorum_id,
        rank,
        world_size,
        list(global_ranks) if global_ranks is not None else list(range(world_size)),
    )
    return pg
