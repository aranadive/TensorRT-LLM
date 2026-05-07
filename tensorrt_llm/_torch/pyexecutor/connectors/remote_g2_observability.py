# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Mapping

REMOTE_G2_LIFECYCLE_EVENTS = {
    "planned",
    "resolved",
    "truncated",
    "transferred",
    "fallback",
    "failed",
    "released",
}

_FORBIDDEN_DETAIL_KEY_PARTS = (
    "ptr",
    "address",
    "descriptor",
    "nixl",
    "transfer_tuple",
    "memory_desc",
    "metadata",
)


@dataclass(frozen=True)
class RemoteG2LifecycleEvent:
    event: str
    reason: str = "none"
    tier: str = "g2"
    outcome: str = "unknown"
    request_id: int | str | None = None
    plan_id: str | None = None
    lease_id: str | None = None
    source_worker_id: int | None = None
    source_generation: int | None = None
    block_count: int = 0
    byte_count: int = 0
    token_count: int = 0
    details: Mapping[str, Any] = field(default_factory=dict)


class RemoteG2ObservabilitySink:
    def emit(self, event: RemoteG2LifecycleEvent) -> None:
        raise NotImplementedError


class NullRemoteG2ObservabilitySink(RemoteG2ObservabilitySink):
    def emit(self, event: RemoteG2LifecycleEvent) -> None:
        return


class InMemoryRemoteG2ObservabilitySink(RemoteG2ObservabilitySink):
    def __init__(self) -> None:
        self.events: list[RemoteG2LifecycleEvent] = []
        self.counts: dict[tuple[str, str, str, str], int] = {}
        self.token_histogram: list[int] = []
        self.byte_histogram: list[int] = []

    def emit(self, event: RemoteG2LifecycleEvent) -> None:
        sanitized = replace(
            event, details=sanitize_remote_g2_event_details(event.details)
        )
        self.events.append(sanitized)
        key = (
            sanitized.event,
            sanitized.reason,
            sanitized.tier,
            sanitized.outcome,
        )
        self.counts[key] = self.counts.get(key, 0) + 1
        if sanitized.token_count:
            self.token_histogram.append(sanitized.token_count)
        if sanitized.byte_count:
            self.byte_histogram.append(sanitized.byte_count)


def sanitize_remote_g2_event_details(details: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in details.items():
        key_text = str(key)
        key_lower = key_text.lower()
        if any(part in key_lower for part in _FORBIDDEN_DETAIL_KEY_PARTS):
            continue
        sanitized[key_text] = _sanitize_value(value)
    return sanitized


def log_remote_g2_event(
    event: RemoteG2LifecycleEvent, logger: logging.Logger | None = None
) -> None:
    active_logger = logger if logger is not None else logging.getLogger(__name__)
    payload = {
        "event": event.event,
        "reason": event.reason,
        "tier": event.tier,
        "outcome": event.outcome,
        "request_id": event.request_id,
        "plan_id": event.plan_id,
        "lease_id": event.lease_id,
        "source_worker_id": event.source_worker_id,
        "source_generation": event.source_generation,
        "block_count": event.block_count,
        "byte_count": event.byte_count,
        "token_count": event.token_count,
        "details": sanitize_remote_g2_event_details(event.details),
    }
    active_logger.info(
        "remote_g2_lifecycle event=%s reason=%s outcome=%s",
        event.event,
        event.reason,
        event.outcome,
        extra={"remote_g2_event": payload},
    )


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return sanitize_remote_g2_event_details(value)
    if isinstance(value, tuple):
        return tuple(_sanitize_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value
