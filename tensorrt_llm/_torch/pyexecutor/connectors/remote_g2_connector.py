# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from .kv_cache_connector import (
    KvCacheConnectorScheduler,
    KvCacheConnectorWorker,
    SchedulerOutput,
)
from .remote_g2 import (
    RemoteG2BindingRecord,
    RemoteG2ResolveResult,
    RemoteKvReusePlan,
    TargetRemoteG2BindingStore,
    TargetRemotePlanStore,
    target_remote_g2_plan_store,
)


@dataclass(frozen=True)
class RemoteG2ConnectorMetadata:
    bindings: tuple[RemoteG2BindingRecord, ...] = ()


def _missing_release_lease(lease_id: str, reason: str) -> bool:
    return False


class RemoteG2KvCacheConnectorScheduler(KvCacheConnectorScheduler):
    def __init__(
        self,
        llm_args: Any,
        *,
        plan_store: Optional[TargetRemotePlanStore] = None,
        binding_store: Optional[TargetRemoteG2BindingStore] = None,
        resolve_and_lease: Optional[
            Callable[[RemoteKvReusePlan], RemoteG2ResolveResult]
        ] = None,
        release_lease: Optional[Callable[[str, str], bool]] = None,
    ) -> None:
        super().__init__(llm_args)
        self._plan_store = (
            plan_store if plan_store is not None else target_remote_g2_plan_store()
        )
        self._resolve_and_lease = resolve_and_lease
        self._binding_store = (
            binding_store
            if binding_store is not None
            else TargetRemoteG2BindingStore(release_lease or _missing_release_lease)
        )

    def get_num_new_matched_tokens(
        self, request: Any, num_computed_tokens: int
    ) -> tuple[int, bool]:
        plan = self._plan_store.get(request.request_id)
        if plan is None or self._resolve_and_lease is None:
            return (0, False)

        record = self._binding_store.resolve_for_request(
            request.request_id,
            plan,
            num_computed_tokens,
            self._resolve_and_lease,
        )
        if record is None:
            return (0, False)
        return (record.matched_tokens, True)

    def update_state_after_alloc(self, request: Any, block_ids: list[int]) -> None:
        self._binding_store.bind_target_blocks(request.request_id, block_ids)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> RemoteG2ConnectorMetadata:
        bindings: list[RemoteG2BindingRecord] = []
        seen_request_ids: set[int | str] = set()
        for request_data in (
            scheduler_output.new_requests + scheduler_output.cached_requests
        ):
            request_id = request_data.request_id
            if request_id in seen_request_ids:
                continue
            seen_request_ids.add(request_id)
            record = self._binding_store.get(request_id)
            if record is not None and record.is_transfer_ready:
                bindings.append(record)
        return RemoteG2ConnectorMetadata(tuple(bindings))

    def request_finished(self, request: Any, cache_block_ids: list[int]) -> bool:
        self._binding_store.discard(request.request_id, "request_finished")
        self._plan_store.discard(request.request_id)
        return False


class RemoteG2KvCacheConnectorWorker(KvCacheConnectorWorker):
    def register_kv_caches(self, kv_cache_tensor: Any) -> None:
        self._kv_cache_tensor = kv_cache_tensor

    def start_load_kv(self, stream: Any) -> None:
        metadata = self.get_connector_meta()
        if isinstance(metadata, RemoteG2ConnectorMetadata) and metadata.bindings:
            raise RuntimeError("remote G2 transfer is not available before Phase 5")

    def wait_for_layer_load(self, layer_idx: int, stream: Any) -> None:
        return

    def save_kv_layer(self, layer_idx: int, stream: Any) -> None:
        return

    def wait_for_save(self, stream: Any) -> None:
        return

    def get_finished(
        self, finished_gen_req_ids: list[int], started_loading_req_ids: list[int]
    ) -> tuple[list[int], list[int]]:
        return ([], [])
