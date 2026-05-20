# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the target-side RPC bridge from inside the engine subprocess.

Mirror of remote_g2_source_setup.py for the other direction: the connector
scheduler runs in the engine subprocess, but the dynamo runtime client
that knows how to reach the source worker lives in the dynamo parent.
This module opens a ZMQ REQ socket connecting to the parent's local REP
loop and installs sync callables into the connector's module state so
the scheduler picks them up at call time.

Wire format mirrors the source side: pickle-encoded request/response,
``{"method": ..., "payload": ...}`` ⇄ ``{"ok": bool, "result"|"error": ...}``.

Note on installation order: PyExecutor constructs the connector
scheduler *before* this bootstrap runs (see py_executor_creator.py
where ``scheduler_cls(llm_args)`` is invoked). The scheduler stashes
``self._resolve_and_lease = None`` in that case; ``get_num_new_matched_tokens``
falls back to reading ``remote_g2_connector._installed_resolve_and_lease``
at call time, so installation can happen later.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import pickle
import threading
import time
from typing import Any, Optional

from . import remote_g2_connector
from .remote_g2 import (
    RemoteG2BlockStatus,
    RemoteG2Descriptor,
    RemoteG2ResolveResult,
    RemoteKvReusePlan,
)
from .remote_g2_source_setup import _walk_to_dynamo_worker_pid


class _TargetReqWrapper:
    """Thread-safe ZMQ REQ client to the dynamo parent's local REP bridge.

    REQ/REP is strict send-then-recv on a single socket, so concurrent
    callers serialize on the lock. The connector scheduler already runs
    in TP rank 0 only, so contention is minimal — at most one in-flight
    RPC per scheduler tick.
    """

    def __init__(self, socket_path: str, timeout_ms: int = 30_000) -> None:
        import zmq

        self._socket_path = socket_path
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.REQ)
        self._socket.RCVTIMEO = timeout_ms
        self._socket.SNDTIMEO = timeout_ms
        self._socket.connect(f"ipc://{socket_path}")
        self._lock = threading.Lock()

    def request(self, method: str, payload: dict) -> dict:
        with self._lock:
            self._socket.send(pickle.dumps({"method": method, "payload": payload}))
            raw = self._socket.recv()
        return pickle.loads(raw)


def _plan_to_dict(plan: RemoteKvReusePlan) -> dict:
    """Convert a plan dataclass to a dict for wire transport. The parent
    REP loop forwards the dict to client.direct() as-is."""
    return dataclasses.asdict(plan)


def _result_from_dict(data: dict) -> Optional[RemoteG2ResolveResult]:
    """Reconstruct RemoteG2ResolveResult from the source's dict response.

    Returns None when the response shape is malformed. Caller treats
    that as "plan not resolvable" — same as a transport failure.
    """
    if not isinstance(data, dict):
        return None
    try:
        descriptors = tuple(
            RemoteG2Descriptor(
                block_hash=int(d["block_hash"]),
                descriptor_generation=int(d["descriptor_generation"]),
                pool_id=str(d["pool_id"]),
                byte_offset=int(d["byte_offset"]),
                byte_length=int(d["byte_length"]),
                metadata=dict(d.get("metadata") or {}),
            )
            for d in (data.get("descriptors") or ())
        )
        per_block_status = tuple(
            RemoteG2BlockStatus(
                block_hash=int(s["block_hash"]),
                status=str(s["status"]),
                descriptor_generation=(
                    int(s["descriptor_generation"])
                    if s.get("descriptor_generation") is not None
                    else None
                ),
            )
            for s in (data.get("per_block_status") or ())
        )
        return RemoteG2ResolveResult(
            lease_id=data.get("lease_id"),
            descriptors=descriptors,
            num_tokens=int(data.get("num_tokens", 0)),
            reason=str(data.get("reason", "ok")),
            source_generation=int(data.get("source_generation", 0)),
            per_block_status=per_block_status,
        )
    except (KeyError, TypeError, ValueError):
        logging.exception("remote_g2: malformed resolve response dict: %r", data)
        return None


def _empty_result(reason: str) -> RemoteG2ResolveResult:
    """Build a no-lease result the BindingStore will treat as 'not
    resolvable' so the scheduler falls back to local recompute."""
    return RemoteG2ResolveResult(
        lease_id=None,
        descriptors=(),
        num_tokens=0,
        reason=reason,
        source_generation=0,
        per_block_status=(),
    )


def _make_resolve_callable(wrapper: _TargetReqWrapper):
    def _resolve(plan: RemoteKvReusePlan) -> RemoteG2ResolveResult:
        logging.warning(
            "PROBE rpc_chain target_resolve_callable pid=%d plan_id=%s "
            "source_worker_id=%s",
            os.getpid(),
            plan.plan_id,
            plan.source_worker_id,
        )
        try:
            response = wrapper.request(
                "resolve",
                {
                    "plan": _plan_to_dict(plan),
                    "source_worker_id": int(plan.source_worker_id),
                },
            )
        except Exception:
            logging.exception("remote_g2: target REQ resolve raised")
            return _empty_result("transport_failure")

        if not isinstance(response, dict):
            return _empty_result("malformed_response")
        if not response.get("ok"):
            err = response.get("error", "unknown")
            logging.warning("remote_g2: target REQ resolve returned not-ok: %s", err)
            return _empty_result(f"rpc_error:{err}")
        result = _result_from_dict(response.get("result") or {})
        return result or _empty_result("malformed_result")

    return _resolve


def _make_release_callable(wrapper: _TargetReqWrapper, default_source_worker_id: int):
    """The connector's release_lease signature is ``(lease_id, reason) -> bool``;
    it does not carry source_worker_id. We close over a default ID — the
    parent's REP loop is responsible for routing the release back to the
    correct source via the lease_id-to-source mapping it maintains.

    For the POC, we pass the connector worker's own worker_id as the
    default. The parent's REP loop ignores it for release calls and
    extracts the source from the lease_id prefix instead.
    """

    def _release(lease_id: str, reason: str) -> bool:
        try:
            response = wrapper.request(
                "release",
                {
                    "lease_id": str(lease_id),
                    "reason": str(reason),
                    "source_worker_id": int(default_source_worker_id),
                },
            )
        except Exception:
            logging.exception("remote_g2: target REQ release raised")
            return False
        if not isinstance(response, dict) or not response.get("ok"):
            return False
        return bool(response.get("result"))

    return _release


def _wait_for_socket(path: str, timeout_s: float = 30.0) -> bool:
    """Poll for the parent-side REP socket file to appear. The dynamo
    parent binds it during init_llm_worker, which races slightly with
    engine subprocess setup; a short poll catches the case where this
    bootstrap runs first."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.25)
    return False


def maybe_start_remote_g2_target_client() -> bool:
    """Open the engine→parent ZMQ REQ socket and install module-state
    callables on remote_g2_connector. Returns True on success, False
    when not configured (no dynamo parent reachable) or when the
    parent's REP socket never appears.

    Called from PyExecutor right after the source-side service is
    bootstrapped. Order-independent with respect to scheduler
    construction because the connector reads module state at call time.
    """
    dynamo_pid = _walk_to_dynamo_worker_pid()
    if dynamo_pid is None:
        logging.info(
            "remote_g2: target client skipped "
            "(dynamo parent not reachable from engine subprocess)"
        )
        return False

    socket_path = f"/tmp/dynamo_remote_g2_target_{dynamo_pid}.sock"
    if not _wait_for_socket(socket_path):
        logging.warning(
            "remote_g2: target client skipped (parent REP socket %s never appeared)",
            socket_path,
        )
        return False

    # The default source_worker_id is only used as a payload field on
    # release calls; the parent REP loop derives the real source from
    # the lease_id. Use the dynamo parent's worker_id (via the same
    # env/sidecar that source setup uses) so it's a real number.
    try:
        own_worker_id = int(os.environ.get("DYNAMO_REMOTE_G2_WORKER_ID", "0"))
    except ValueError:
        own_worker_id = 0

    try:
        wrapper = _TargetReqWrapper(socket_path)
    except Exception:
        logging.exception("remote_g2: failed to open target REQ socket at %s", socket_path)
        return False

    resolve_fn = _make_resolve_callable(wrapper)
    release_fn = _make_release_callable(wrapper, own_worker_id)

    remote_g2_connector.install_resolve_and_lease(resolve_fn)
    remote_g2_connector.install_release_lease(release_fn)

    logging.warning(
        "remote_g2: target client installed "
        "(socket=%s own_worker_id=%s)",
        socket_path,
        own_worker_id,
    )
    return True
