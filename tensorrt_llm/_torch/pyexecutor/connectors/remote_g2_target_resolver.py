# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Sequence

from .remote_g2 import RemoteG2BindingRecord
from .remote_g2_transfer import RemoteG2TransferDescriptor, RemoteG2TransferError

_PRIMARY_CACHE_LEVEL = 0
_SECONDARY_CACHE_LEVEL = 1


def make_target_descriptor_resolver(
    kv_cache_manager: Any,
    *,
    primary_pool_base_ptr: int,
    block_size_bytes: int,
    device_id: int = 0,
    name_prefix: str = "g2-target",
) -> Callable[[RemoteG2BindingRecord], Sequence[RemoteG2TransferDescriptor]]:
    """Build a target_descriptor_resolver for RemoteG2NixlTransferAdapter.

    Reads the authoritative slot of each target_block_id via a pin/unpin
    pair (the block is already refcount-held by the admitted request, so
    the +1/-1 here does not affect residency). Refuses to produce a
    descriptor for any block that the pin reports on the secondary tier,
    since the NIXL adapter expects target descriptors to live in VRAM.
    """
    if block_size_bytes <= 0:
        raise ValueError("block_size_bytes must be positive")
    if primary_pool_base_ptr <= 0:
        raise ValueError("primary_pool_base_ptr must be a non-zero address")

    def resolver(
        record: RemoteG2BindingRecord,
    ) -> Sequence[RemoteG2TransferDescriptor]:
        descriptors: list[RemoteG2TransferDescriptor] = []
        for bound_block in record.bound_blocks:
            block_id = int(bound_block.target_block_id)
            if block_id < 0:
                raise RemoteG2TransferError(
                    f"target block_id {block_id} is invalid"
                )

            locations = kv_cache_manager.pin_blocks_by_id([block_id])
            if not locations:
                raise RemoteG2TransferError(
                    f"pin_blocks_by_id returned no location for block_id={block_id}"
                )
            slot_idx, cache_level = locations[0]
            slot_idx = int(slot_idx)
            cache_level = int(cache_level)
            kv_cache_manager.unpin_blocks_by_id([block_id])

            if cache_level != _PRIMARY_CACHE_LEVEL:
                raise RemoteG2TransferError(
                    f"target block_id={block_id} pinned on cache_level={cache_level}, "
                    f"expected {_PRIMARY_CACHE_LEVEL} (primary VRAM)"
                )

            ptr = primary_pool_base_ptr + slot_idx * block_size_bytes
            descriptors.append(
                RemoteG2TransferDescriptor(
                    ptr=ptr,
                    size=block_size_bytes,
                    device_id=device_id,
                    memory_type="VRAM",
                    name=f"{name_prefix}_{block_id}",
                )
            )
        return descriptors

    return resolver
