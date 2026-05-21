# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap the source-side SourceG2DescriptorRegistry from inside the
engine subprocess.

The KV cache manager (and the C++ APIs we depend on — find_block_by_hash,
pin_blocks_by_id, get_secondary_pool_data) only exists in the engine
subprocess that PyExecutor runs in. So the registry has to be built there.
The connector worker's register_kv_caches hook is the natural anchor: it
runs in that subprocess, right after the KV cache pool is allocated.

Identity (source_worker_id, source_dp_rank) is read from environment
variables that the dynamo worker process sets before spawning the engine.
"""

from __future__ import annotations

import logging
import os
import pickle
import threading
from dataclasses import dataclass
from typing import Any, Optional

from .remote_g2 import SourceG2DescriptorRegistry
from .remote_g2_source_adapter import make_kv_pin_callbacks


@dataclass
class _NixlSourceBundle:
    """Source-side NIXL agent + metadata captured for the metadata RPC.

    `agent` is the live NixlTransferAgent (kept alive for the worker's
    lifetime — its memory registrations and the daemon polling thread
    expire when the agent is GCed). `agent_desc` is the opaque
    bytes-blob other workers' agents need to call `load_remote_agent`.
    `remote_name` is the agent's identifier used as the third argument
    to `TransferRequest(..., remote_name)` from a peer.
    """

    agent: Any
    remote_name: str
    agent_desc: bytes
    pool_base_ptr: int
    pool_size_bytes: int
    source_generation: int = 1


# Process-wide singleton — populated by maybe_start_remote_g2_service after
# the NIXL agent is constructed, read by the ZMQ REP loop when answering
# get_metadata RPCs (Stage T2).
_GLOBAL_NIXL_SOURCE_BUNDLE: Optional[_NixlSourceBundle] = None


def get_nixl_source_bundle() -> Optional[_NixlSourceBundle]:
    return _GLOBAL_NIXL_SOURCE_BUNDLE


def _result_to_dict(result: Any) -> dict:
    """Convert a RemoteG2ResolveResult dataclass into a plain dict for
    wire transport. The dynamo parent and downstream consumers see only
    dicts and do not need to import RemoteG2ResolveResult / its nested
    types.
    """
    descriptors = [
        {
            "block_hash": d.block_hash,
            "descriptor_generation": d.descriptor_generation,
            "pool_id": d.pool_id,
            "byte_offset": d.byte_offset,
            "byte_length": d.byte_length,
            "metadata": dict(d.metadata or {}),
        }
        for d in (result.descriptors or ())
    ]
    per_block_status = [
        {
            "block_hash": s.block_hash,
            "status": s.status,
            "descriptor_generation": s.descriptor_generation,
        }
        for s in (result.per_block_status or ())
    ]
    return {
        "lease_id": result.lease_id,
        "descriptors": descriptors,
        "num_tokens": result.num_tokens,
        "reason": result.reason,
        "source_generation": result.source_generation,
        "per_block_status": per_block_status,
    }


def _start_zmq_rep_service(registry: SourceG2DescriptorRegistry, dynamo_pid: int) -> str:
    """Start a ZMQ REP daemon thread bound to a Unix domain socket
    tagged with the dynamo parent's PID. The dynamo parent process
    constructs the matching REQ client using the same path. Returns
    the socket path for logging.

    Wire format: pickle-encoded {"method": <name>, "payload": <dict>}
    request; pickle-encoded {"ok": bool, "result"|"error": <value>}
    response. Pickle is safe here since both ends are colocated Python
    processes on the same host.
    """
    import zmq  # imported lazily so this module stays importable on hosts without zmq

    socket_path = f"/tmp/dynamo_remote_g2_ipc_{dynamo_pid}.sock"
    try:
        os.unlink(socket_path)
    except FileNotFoundError:
        pass

    ctx = zmq.Context.instance()
    rep = ctx.socket(zmq.REP)
    rep.bind(f"ipc://{socket_path}")

    def _loop() -> None:
        while True:
            try:
                raw = rep.recv()
            except Exception:
                logging.exception("remote_g2: ZMQ REP recv failed; exiting loop")
                return
            try:
                req = pickle.loads(raw)
                method = req.get("method")
                payload = req.get("payload") or {}
                if method == "resolve_and_lease":
                    result = registry.resolve_and_lease(payload.get("plan"))
                    response = {"ok": True, "result": _result_to_dict(result)}
                elif method == "release_lease":
                    completed = registry.release_lease(
                        payload["lease_id"], payload.get("reason", "ack")
                    )
                    response = {"ok": True, "result": completed}
                else:
                    response = {"ok": False, "error": f"unknown method: {method!r}"}
            except Exception as exc:
                logging.exception("remote_g2: ZMQ REP handler raised")
                response = {"ok": False, "error": repr(exc)}
            try:
                rep.send(pickle.dumps(response))
            except Exception:
                logging.exception("remote_g2: ZMQ REP send failed")

    thread = threading.Thread(target=_loop, name="remote_g2_zmq_rep", daemon=True)
    thread.start()
    return socket_path


def _walk_to_dynamo_worker_pid(max_depth: int = 10) -> Optional[int]:
    """Walk up the process tree from this process and return the first
    ancestor whose cmdline mentions 'dynamo.trtllm'. OpenMPI's orted
    strips arbitrary env vars when spawning ranks, so the engine
    subprocess can't read DYNAMO_REMOTE_G2_WORKER_ID directly; this
    helper finds the dynamo parent so we can read a sidecar file
    /tmp/dynamo_remote_g2_worker_<pid>.txt instead.
    """
    try:
        pid = os.getpid()
        for _ in range(max_depth):
            try:
                with open(f"/proc/{pid}/status") as f:
                    status = f.read()
            except (FileNotFoundError, PermissionError):
                return None
            ppid = None
            for line in status.splitlines():
                if line.startswith("PPid:"):
                    ppid = int(line.split()[1])
                    break
            if ppid is None or ppid <= 1:
                return None
            try:
                with open(f"/proc/{ppid}/cmdline") as f:
                    cmdline = f.read().replace("\0", " ")
            except (FileNotFoundError, PermissionError):
                cmdline = ""
            if "dynamo.trtllm" in cmdline:
                return ppid
            pid = ppid
        return None
    except Exception:
        return None


def _resolve_source_identity() -> Optional[tuple[int, int]]:
    """Return (source_worker_id, dynamo_parent_pid) on success, None when
    the dynamo identity cannot be reached from the engine subprocess.

    Reads the worker_id from env var first, falling back to a sidecar
    file written by the dynamo parent (since OpenMPI orted strips env
    vars across the spawn boundary). The dynamo parent's PID is also
    needed so the ZMQ REP service can bind to a Unix domain socket the
    parent can find.
    """
    dynamo_pid = _walk_to_dynamo_worker_pid()
    env_value = os.environ.get("DYNAMO_REMOTE_G2_WORKER_ID")
    if env_value and dynamo_pid is not None:
        try:
            return int(env_value), dynamo_pid
        except ValueError:
            pass
    if dynamo_pid is None:
        return None
    sidecar = f"/tmp/dynamo_remote_g2_worker_{dynamo_pid}.txt"
    try:
        with open(sidecar) as f:
            return int(f.read().strip()), dynamo_pid
    except Exception:
        return None


def _secondary_pool_base_ptr(kv: Any) -> int:
    """Return the secondary KV cache pool's base host address, or 0 when
    not available (no host pool allocated, exposure binding missing, etc.)."""
    try:
        pool = kv.get_secondary_pool_data(0)
        logging.warning(
            "remote_g2 DEBUG: secondary pool type=%s is_none=%s "
            "numel=%s data_ptr=%s",
            type(pool).__name__,
            pool is None,
            getattr(pool, "numel", lambda: "?")() if pool is not None else "?",
            getattr(pool, "data_ptr", lambda: "?")() if pool is not None else "?",
        )
        if pool is None:
            return 0
        return int(pool.data_ptr())
    except Exception as exc:
        logging.warning("remote_g2 DEBUG: get_secondary_pool_data raised: %r", exc)
        return 0


def _derive_block_size_bytes(kv: Any) -> Optional[int]:
    """Compute per-block byte size from the secondary pool's total bytes
    divided by the C++-reported secondary capacity. Uses
    KvCacheIterationStats.secondary_max_num_blocks (constant for the
    worker's lifetime) so we never have to assume the pool tensor's
    dimension order.
    """
    try:
        pool = kv.get_secondary_pool_data(0)
    except Exception:
        return None
    if pool is None or pool.numel() == 0:
        return None

    try:
        iter_stats = kv.get_iteration_stats()
    except Exception:
        return None
    if not iter_stats:
        return None

    stats = next(iter(iter_stats.values()))
    num_secondary = int(getattr(stats, "secondary_max_num_blocks", 0) or 0)
    if num_secondary <= 0:
        return None

    total_bytes = pool.element_size() * pool.numel()
    return total_bytes // num_secondary


def _derive_window_size(kv: Any) -> Optional[int]:
    """Return the attention window size the KV cache manager is configured
    with. Reads from KvCacheIterationStats keys; under the single-window-
    block-manager constraint this connector operates under there is
    exactly one entry."""
    try:
        iter_stats = kv.get_iteration_stats()
    except Exception:
        return None
    if not iter_stats:
        return None
    try:
        return int(next(iter(iter_stats.keys())))
    except (StopIteration, TypeError, ValueError):
        return None


def _pool_size_bytes(kv: Any) -> Optional[int]:
    """Total byte size of the host_pinned secondary pool — element_size *
    numel of the tensor returned by C++ `get_secondary_pool_data(0)`. We
    register this whole range with the NIXL agent so remote workers can
    issue READs against any slot in it."""
    try:
        pool = kv.get_secondary_pool_data(0)
    except Exception:
        return None
    if pool is None or pool.numel() == 0:
        return None
    return int(pool.element_size() * pool.numel())


def _setup_nixl_source_agent(
    pool_base_ptr: int,
    pool_size_bytes: int,
    source_worker_id: int,
) -> Optional[_NixlSourceBundle]:
    """Build a NIXL agent on the source side and register the
    host_pinned secondary pool memory range so it can be read remotely.

    Returns a populated bundle, or ``None`` when the nixl package isn't
    available or registration fails. Caller logs and continues without
    NIXL — remote-G2 resolve calls will still succeed (descriptors
    flow), but no bytes will be moveable until this works.
    """
    try:
        from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
        from tensorrt_llm._torch.disaggregation.nixl.agent import NixlTransferAgent
    except Exception:
        logging.exception(
            "remote_g2: NIXL agent imports failed; transfers will not be available"
        )
        return None

    agent_name = f"remote-g2-source-{source_worker_id}"
    try:
        agent = NixlTransferAgent(agent_name)
    except Exception:
        logging.exception(
            "remote_g2: NixlTransferAgent(%s) construction failed", agent_name
        )
        return None

    # device_id=0 for DRAM is the standard NIXL convention for host
    # memory regardless of which GPU this worker happens to be on.
    # The fourth element is a human-readable name for the registration.
    pool_reg_name = f"remote-g2-host-pool-{source_worker_id}"
    desc_tuple = (pool_base_ptr, pool_size_bytes, 0, pool_reg_name)
    try:
        agent.register_memory(RegMemoryDescs(type="DRAM", descs=[desc_tuple]))
    except Exception:
        logging.exception(
            "remote_g2: register_memory failed for pool at 0x%x size=%d",
            pool_base_ptr,
            pool_size_bytes,
        )
        return None

    try:
        agent_desc = agent.get_local_agent_desc()
    except Exception:
        logging.exception(
            "remote_g2: get_local_agent_desc failed for agent %s", agent_name
        )
        return None

    if not agent_desc:
        logging.warning(
            "remote_g2: agent %s returned empty local agent_desc", agent_name
        )
        return None

    return _NixlSourceBundle(
        agent=agent,
        remote_name=agent_name,
        agent_desc=agent_desc,
        pool_base_ptr=pool_base_ptr,
        pool_size_bytes=pool_size_bytes,
    )


def maybe_start_remote_g2_service(
    kv: Any,
    *,
    lease_ttl_ms: int = 30_000,
    pool_id: str = "g2-host-pinned",
    tier: str = "host_pinned",
) -> Optional[SourceG2DescriptorRegistry]:
    """Start the source-side remote-G2 service against a live kv_cache_manager.

    Called from PyExecutor right after kv_cache_manager is constructed,
    inside the engine subprocess. Builds a SourceG2DescriptorRegistry
    and (in future iterations) spawns a daemon thread that exposes it
    over ZMQ for the dynamo parent process to forward RPC calls into.

    Returns None when the deployment isn't configured for remote-G2
    (env var missing), or when prerequisites aren't met (no secondary
    pool, no host_pinned blocks yet, etc.). Caller treats None as
    "remote-G2 service not started".

    Identity (source_worker_id, source_dp_rank) is read from env vars
    set by the dynamo parent process:
      - DYNAMO_REMOTE_G2_WORKER_ID  (required; matches dynamo
        endpoint.connection_id() for the owning dynamo worker process)
      - DYNAMO_REMOTE_G2_DP_RANK    (defaults to 0)
    """
    identity = _resolve_source_identity()
    if identity is None:
        logging.info(
            "remote_g2: source registry skipped "
            "(DYNAMO_REMOTE_G2_WORKER_ID not reachable via env var or sidecar)"
        )
        return None
    source_worker_id, dynamo_pid = identity

    # PyExecutor.kv_cache_manager is a Python wrapper class
    # (resource_manager.KVCacheManager); the C++ binding with
    # get_secondary_pool_data / find_block_by_hash / pin_blocks_by_id
    # sits at .impl. Unwrap once so the rest of the code (and the
    # SourceG2DescriptorRegistry it builds) talks to the C++ object
    # directly.
    kv = getattr(kv, "impl", kv)
    try:
        source_dp_rank = int(os.environ.get("DYNAMO_REMOTE_G2_DP_RANK", "0"))
    except ValueError:
        source_dp_rank = 0

    pool_base_ptr = _secondary_pool_base_ptr(kv)
    if pool_base_ptr == 0:
        logging.info(
            "remote_g2: source registry skipped (secondary pool unavailable)"
        )
        return None

    block_size_bytes = _derive_block_size_bytes(kv)
    if not block_size_bytes or block_size_bytes <= 0:
        logging.warning(
            "remote_g2: source registry skipped (block_size_bytes unknown)"
        )
        return None

    window_size = _derive_window_size(kv)
    if window_size is None or window_size <= 0:
        logging.warning(
            "remote_g2: source registry skipped (window_size unknown)"
        )
        return None

    acquire_pin, release_pin = make_kv_pin_callbacks(
        kv,
        secondary_pool_base_ptr=pool_base_ptr,
        block_size_bytes=block_size_bytes,
    )

    registry = SourceG2DescriptorRegistry(
        source_worker_id=source_worker_id,
        source_dp_rank=source_dp_rank,
        lease_ttl_ms=lease_ttl_ms,
        acquire_pin=acquire_pin,
        release_pin=release_pin,
        require_trtllm_pin=True,
        kv=kv,
        window_size=window_size,
        pool_id=pool_id,
        pool_base_ptr=pool_base_ptr,
        block_size_bytes=block_size_bytes,
        tier=tier,
    )

    logging.warning(
        "remote_g2: service started "
        "(source_worker_id=%s dp_rank=%s window_size=%s "
        "block_size_bytes=%s pool_base_ptr=0x%x)",
        source_worker_id,
        source_dp_rank,
        window_size,
        block_size_bytes,
        pool_base_ptr,
    )

    try:
        socket_path = _start_zmq_rep_service(registry, dynamo_pid)
        logging.warning(
            "remote_g2: ZMQ REP service bound at %s (source_worker_id=%s)",
            socket_path,
            source_worker_id,
        )
    except Exception:
        logging.exception(
            "remote_g2: failed to start ZMQ REP service; registry built but "
            "not reachable from dynamo parent"
        )

    # Stage T1 — bootstrap the NIXL agent and register the host_pinned
    # secondary pool. The bundle is stashed as a process-wide singleton
    # so the ZMQ REP service can answer the get_metadata RPC (Stage T2)
    # without threading the bundle through the registry constructor.
    pool_size_bytes = _pool_size_bytes(kv)
    if not pool_size_bytes or pool_size_bytes <= 0:
        logging.warning(
            "remote_g2: NIXL agent skipped (pool_size_bytes unknown)"
        )
    else:
        bundle = _setup_nixl_source_agent(
            pool_base_ptr=pool_base_ptr,
            pool_size_bytes=pool_size_bytes,
            source_worker_id=source_worker_id,
        )
        if bundle is not None:
            global _GLOBAL_NIXL_SOURCE_BUNDLE
            _GLOBAL_NIXL_SOURCE_BUNDLE = bundle
            logging.warning(
                "PROBE remote_g2_source_nixl: agent_name=%s pool_base_ptr=0x%x "
                "pool_size=%d agent_desc_bytes=%d source_generation=%d",
                bundle.remote_name,
                bundle.pool_base_ptr,
                bundle.pool_size_bytes,
                len(bundle.agent_desc),
                bundle.source_generation,
            )
        else:
            logging.warning(
                "remote_g2: NIXL source agent bootstrap failed; "
                "resolve will still work, but transfer is disabled"
            )

    return registry
