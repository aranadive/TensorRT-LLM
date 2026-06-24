# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RemoteG2RankContext:
    """KV-transfer rank identity for remote-G2 connector setup."""

    worker_id: Optional[int]
    dp_rank: int
    kv_rank: int
    kv_world_size: int
    group_ranks: tuple[int, ...]
    enable_attention_dp: bool

    @classmethod
    def from_mapping(
        cls,
        mapping,
        *,
        worker_id: Optional[int] = None,
    ) -> "RemoteG2RankContext":
        if mapping.enable_attention_dp:
            return cls(
                worker_id=worker_id,
                dp_rank=int(mapping.tp_rank),
                kv_rank=0,
                kv_world_size=1,
                group_ranks=(int(mapping.rank),),
                enable_attention_dp=True,
            )

        return cls(
            worker_id=worker_id,
            dp_rank=0,
            kv_rank=int(mapping.tp_rank),
            kv_world_size=int(mapping.tp_size),
            group_ranks=tuple(int(rank) for rank in mapping.tp_group),
            enable_attention_dp=False,
        )

    @property
    def uses_context_qualified_names(self) -> bool:
        return self.enable_attention_dp
