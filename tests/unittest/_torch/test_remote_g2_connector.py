# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[3]
_CONNECTOR_PACKAGE = "tensorrt_llm._torch.pyexecutor.connectors"
_REMOTE_G2_PATH = (
    _ROOT / "tensorrt_llm" / "_torch" / "pyexecutor" / "connectors" / "remote_g2.py"
)
_REMOTE_G2_CONNECTOR_PATH = (
    _ROOT
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2_connector.py"
)


def _install_package(name):
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_connector_modules():
    _install_package("tensorrt_llm")
    _install_package("tensorrt_llm._torch")
    _install_package("tensorrt_llm._torch.pyexecutor")
    _install_package(_CONNECTOR_PACKAGE)

    kv_cache_connector = types.ModuleType(f"{_CONNECTOR_PACKAGE}.kv_cache_connector")

    class KvCacheConnectorScheduler:
        def __init__(self, llm_args):
            self._llm_args = llm_args

    class KvCacheConnectorWorker:
        def __init__(self, llm_args):
            self._llm_args = llm_args
            self._metadata = None

        def bind_connector_meta(self, metadata):
            self._metadata = metadata

        def get_connector_meta(self):
            return self._metadata

    kv_cache_connector.KvCacheConnectorScheduler = KvCacheConnectorScheduler
    kv_cache_connector.KvCacheConnectorWorker = KvCacheConnectorWorker
    kv_cache_connector.SchedulerOutput = object
    sys.modules[f"{_CONNECTOR_PACKAGE}.kv_cache_connector"] = kv_cache_connector

    remote_g2 = _load_module(f"{_CONNECTOR_PACKAGE}.remote_g2", _REMOTE_G2_PATH)
    remote_g2_connector = _load_module(
        f"{_CONNECTOR_PACKAGE}.remote_g2_connector", _REMOTE_G2_CONNECTOR_PATH
    )
    return remote_g2, remote_g2_connector


REMOTE_G2, REMOTE_G2_CONNECTOR = _load_connector_modules()

RemoteG2ConnectorMetadata = REMOTE_G2_CONNECTOR.RemoteG2ConnectorMetadata
RemoteG2Descriptor = REMOTE_G2.RemoteG2Descriptor
RemoteG2ResolveResult = REMOTE_G2.RemoteG2ResolveResult
TargetRemotePlanStore = REMOTE_G2.TargetRemotePlanStore


def _plan(**overrides):
    plan = {
        "plan_id": "plan-1",
        "request_id": "dynamo-request-1",
        "target_worker_id": 42,
        "target_dp_rank": 2,
        "source_worker_id": 7,
        "source_dp_rank": 0,
        "source_tier": "host_pinned",
        "block_hashes": [11, 22, 33],
        "planned_prefix_blocks": 3,
        "block_size_tokens": 16,
        "created_at_ms": 100,
        "expires_at_ms": 10_000,
    }
    plan.update(overrides)
    return plan


def _descriptor(block_hash):
    return RemoteG2Descriptor(
        block_hash=block_hash,
        descriptor_generation=1,
        pool_id="host-pool-0",
        byte_offset=block_hash * 4096,
        byte_length=4096,
    )


def _resolve_result(block_hashes=(11, 22, 33), num_tokens=48, lease_id="lease-1"):
    return RemoteG2ResolveResult(
        lease_id=lease_id,
        descriptors=tuple(_descriptor(block_hash) for block_hash in block_hashes),
        num_tokens=num_tokens,
        source_generation=99,
    )


def test_remote_g2_connector_resolves_before_reporting_tokens():
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    resolve_calls = []
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        resolve_and_lease=lambda plan: resolve_calls.append(plan.plan_id)
        or _resolve_result(num_tokens=32),
        release_lease=lambda lease_id, reason: True,
    )

    tokens, load_kv_async = scheduler.get_num_new_matched_tokens(
        SimpleNamespace(request_id=1234), 0
    )

    assert (tokens, load_kv_async) == (32, True)
    assert resolve_calls == ["plan-1"]


def test_remote_g2_connector_returns_zero_without_plan():
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=TargetRemotePlanStore(clock_ms=lambda: 500),
        resolve_and_lease=lambda plan: _resolve_result(),
        release_lease=lambda lease_id, reason: True,
    )

    assert scheduler.get_num_new_matched_tokens(SimpleNamespace(request_id=1234), 0) == (
        0,
        False,
    )

    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(5678, _plan())
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        release_lease=lambda lease_id, reason: True,
    )

    assert scheduler.get_num_new_matched_tokens(SimpleNamespace(request_id=5678), 0) == (
        0,
        False,
    )


def test_remote_g2_connector_binds_after_allocated_block_ids():
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        resolve_and_lease=lambda plan: _resolve_result(),
        release_lease=lambda lease_id, reason: True,
    )
    request = SimpleNamespace(request_id=1234)

    assert scheduler.get_num_new_matched_tokens(request, 16) == (32, True)
    scheduler.update_state_after_alloc(request, [100, 101, 102])
    metadata = scheduler.build_connector_meta(
        SimpleNamespace(
            new_requests=[SimpleNamespace(request_id=1234)], cached_requests=[]
        )
    )

    assert isinstance(metadata, RemoteG2ConnectorMetadata)
    assert len(metadata.bindings) == 1
    assert [block.target_block_id for block in metadata.bindings[0].bound_blocks] == [
        101,
        102,
    ]


def test_remote_g2_connector_releases_once_on_request_finished():
    released = []
    plan_store = TargetRemotePlanStore(clock_ms=lambda: 500)
    plan_store.put(1234, _plan())
    scheduler = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorScheduler(
        None,
        plan_store=plan_store,
        resolve_and_lease=lambda plan: _resolve_result(lease_id="lease-finished"),
        release_lease=lambda lease_id, reason: released.append((lease_id, reason))
        or True,
    )
    request = SimpleNamespace(request_id=1234)

    assert scheduler.get_num_new_matched_tokens(request, 0) == (48, True)
    assert scheduler.request_finished(request, []) is False
    assert scheduler.request_finished(request, []) is False

    assert released == [("lease-finished", "request_finished")]
    assert plan_store.get(1234) is None


def test_remote_g2_worker_refuses_transfer_before_phase5():
    worker = REMOTE_G2_CONNECTOR.RemoteG2KvCacheConnectorWorker(None)

    worker.bind_connector_meta(RemoteG2ConnectorMetadata())
    worker.start_load_kv(None)

    worker.bind_connector_meta(
        RemoteG2ConnectorMetadata(
            bindings=(
                SimpleNamespace(bound_blocks=(SimpleNamespace(target_block_id=100),)),
            )
        )
    )
    with pytest.raises(
        RuntimeError, match="remote G2 transfer is not available before Phase 5"
    ):
        worker.start_load_kv(None)
