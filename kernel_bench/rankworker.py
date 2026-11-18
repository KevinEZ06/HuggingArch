"""Child process of a measurement group.

Sets the visible die, initialises the process group, runs the cell, and
puts the rows on the parent's queue. Never opens the results CSV — the
parent is the unique writer (`emit.IncrementalCsvWriter`).
"""
from __future__ import annotations

import os
import traceback
from typing import Any, Dict


def entry(cfg: Dict[str, Any], queue) -> None:
    """`multiprocessing` spawn target. `cfg` is pickle-friendly."""
    rank = int(cfg["rank"])
    world = int(cfg["world"])
    try:
        os.environ.update(cfg["child_env"])
        report = _measure(cfg, rank, world)
        queue.put(report)
    except Exception as exc:
        queue.put({
            "rank": rank,
            "rows": [],
            "error": f"rank {rank}: {type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        })


def _align(dist, backend_dev):
    def align() -> None:
        if dist is not None:
            dist.barrier()
        if backend_dev is not None:
            backend_dev.synchronize()
    return align


def _measure(cfg: Dict[str, Any], rank: int, world: int) -> Dict[str, Any]:
    from dataclasses import replace

    from .device import detect_backend
    from .driver import run_jobs
    from .env import detect_env

    backend_dev = detect_backend()
    if backend_dev is not None:
        backend_dev.set_device(rank)

    import torch.distributed as dist_mod
    dist_mod.init_process_group(
        backend=cfg["backend"], rank=rank, world_size=world)
    dist = dist_mod

    env = detect_env()
    if cfg.get("env_library") is not None:
        env = replace(env, library=cfg["env_library"])
    if cfg.get("gpu_sku") and not env.gpu_sku:
        env = replace(env, gpu_sku=cfg["gpu_sku"])

    job = dict(cfg["job"])
    job["rank"] = rank
    rows: list = []
    result = run_jobs(
        [job], env=env,
        date=cfg.get("date") or "1970-01-01",
        contributor=cfg.get("contributor") or "@kernel-bench",
        run_id=cfg.get("run_id") or "",
        box=cfg.get("box") or "",
        in_group=True, align=_align(dist, backend_dev),
        on_row=rows.append)
    if dist is not None:
        try:
            dist.destroy_process_group()
        except Exception:
            pass
    return {
        "rank": rank,
        "rows": rows,
        "skipped": list(result.skipped),
        "failures": list(result.failures),
        "error": (result.failures[0] if result.failures else None),
    }
