# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
import sys
from pathlib import Path

_REMOTE_G2_PATH = (
    Path(__file__).resolve().parents[3]
    / "tensorrt_llm"
    / "_torch"
    / "pyexecutor"
    / "connectors"
    / "remote_g2.py"
)
_SPEC = importlib.util.spec_from_file_location("remote_g2_under_test", _REMOTE_G2_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_REMOTE_G2 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _REMOTE_G2
_SPEC.loader.exec_module(_REMOTE_G2)

REMOTE_G2_REUSE_ENABLED_ENV = _REMOTE_G2.REMOTE_G2_REUSE_ENABLED_ENV
REMOTE_KV_REUSE_PLAN_VERSION = _REMOTE_G2.REMOTE_KV_REUSE_PLAN_VERSION
RemoteKvReusePlan = _REMOTE_G2.RemoteKvReusePlan
SourceG2DescriptorRecord = _REMOTE_G2.SourceG2DescriptorRecord
SourceG2DescriptorRegistry = _REMOTE_G2.SourceG2DescriptorRegistry
TargetRemotePlanStore = _REMOTE_G2.TargetRemotePlanStore


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
        "plan_version": REMOTE_KV_REUSE_PLAN_VERSION,
    }
    plan.update(overrides)
    return plan


def _record(block_hash, generation=1):
    return SourceG2DescriptorRecord(
        block_hash=block_hash,
        source_worker_id=7,
        source_dp_rank=0,
        tier="G2",
        descriptor_generation=generation,
        pool_id="host-pool-0",
        byte_offset=block_hash * 4096,
        byte_length=4096,
    )


def test_target_store_keys_plan_by_trtllm_request_id_and_discards():
    store = TargetRemotePlanStore(clock_ms=lambda: 500)

    stored = store.put(1234, _plan())

    assert stored is not None
    assert store.get(1234).plan_id == "plan-1"
    assert store.get("1234").source_worker_id == 7
    store.discard(1234)
    assert store.get(1234) is None


def test_target_store_rejects_expired_or_non_g2_plan():
    store = TargetRemotePlanStore(clock_ms=lambda: 5_000)

    assert store.put(1, _plan(expires_at_ms=5_000)) is None
    assert store.put(2, _plan(source_tier="device")) is None
    assert store.put(3, {"plan_id": "missing-required-fields"}) is None
    assert len(store) == 0


def test_target_store_respects_remote_g2_kill_switch():
    old_value = os.environ.get(REMOTE_G2_REUSE_ENABLED_ENV)
    os.environ[REMOTE_G2_REUSE_ENABLED_ENV] = "false"
    try:
        store = TargetRemotePlanStore(clock_ms=lambda: 500)
        assert store.put(1234, _plan()) is None
        assert len(store) == 0
    finally:
        if old_value is None:
            os.environ.pop(REMOTE_G2_REUSE_ENABLED_ENV, None)
        else:
            os.environ[REMOTE_G2_REUSE_ENABLED_ENV] = old_value


def test_source_registry_resolve_and_lease_returns_live_contiguous_prefix():
    released = []
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        source_generation=99,
        clock_ms=lambda: 1_000,
        acquire_pin=lambda record, lease_id: f"{lease_id}:{record.block_hash}",
        release_pin=released.append,
        require_trtllm_pin=True,
    )
    first = _record(11, generation=4)
    second = _record(22, generation=5)
    registry.upsert_descriptor(first)
    registry.upsert_descriptor(second)

    result = registry.resolve_and_lease(_plan())

    assert result.reason == "ok"
    assert result.num_tokens == 32
    assert result.source_generation == 99
    assert [d.block_hash for d in result.descriptors] == [11, 22]
    assert [d.descriptor_generation for d in result.descriptors] == [4, 5]
    assert [status.status for status in result.per_block_status] == [
        "live",
        "live",
        "missing",
    ]
    assert first.lease_count == 1
    assert second.lease_count == 1

    assert registry.release_lease(result.lease_id, "success") is True
    assert registry.release_lease(result.lease_id, "duplicate") is False
    assert first.lease_count == 0
    assert second.lease_count == 0
    assert len(released) == 2


def test_source_registry_fails_closed_for_wrong_source_or_tier():
    registry = SourceG2DescriptorRegistry(source_worker_id=7, source_dp_rank=0)
    registry.upsert_descriptor(_record(11))

    assert (
        registry.resolve_and_lease(_plan(source_worker_id=8)).reason
        == "wrong_source_worker"
    )
    assert (
        registry.resolve_and_lease(_plan(source_dp_rank=1)).reason == "wrong_source_rank"
    )
    assert registry.resolve_and_lease(_plan(source_tier="device")).reason == "wrong_source_tier"


def test_source_registry_reports_missing_pin_hook_when_required():
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        clock_ms=lambda: 1_000,
        require_trtllm_pin=True,
    )
    registry.upsert_descriptor(_record(11))

    result = registry.resolve_and_lease(_plan(planned_prefix_blocks=1))

    assert result.lease_id is None
    assert result.reason == "missing_trtllm_pin_hook"


def test_source_registry_reports_invalid_plan_without_leasing():
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        clock_ms=lambda: 1_000,
    )
    registry.upsert_descriptor(_record(11))

    result = registry.resolve_and_lease({"plan_id": "missing-required-fields"})

    assert result.lease_id is None
    assert result.reason == "invalid_plan"


def test_source_registry_reports_first_missing_block_status():
    registry = SourceG2DescriptorRegistry(
        source_worker_id=7,
        source_dp_rank=0,
        clock_ms=lambda: 1_000,
    )

    result = registry.resolve_and_lease(_plan(planned_prefix_blocks=1))

    assert result.reason == "no_live_remote_g2_prefix"
    assert result.per_block_status[0].block_hash == 11
    assert result.per_block_status[0].status == "missing"


def test_remote_plan_parser_truncates_prefix_to_hash_count():
    parsed = RemoteKvReusePlan.from_dict(_plan(planned_prefix_blocks=10))

    assert parsed.planned_prefix_blocks == 3
    assert parsed.planned_hashes == (11, 22, 33)
