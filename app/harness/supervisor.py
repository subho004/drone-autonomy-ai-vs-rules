"""Run a worker pool (``spawn`` processes) against the SQLite queue."""

from __future__ import annotations

import multiprocessing as mp
import time

from app.core.logging import get_logger
from app.harness.store import Store
from app.harness.worker import worker_main

logger = get_logger(__name__)

POLL_S = 2.0


def run_pool(
    db_path: str, pool_name: str, n_workers: int, experiment_id: int
) -> dict[str, int]:
    """Start ``n_workers`` spawn processes and wait for the queue to drain."""

    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(
            target=worker_main,
            args=(db_path, pool_name, f"{pool_name}-{i}"),
            daemon=False,
        )
        for i in range(n_workers)
    ]
    t0 = time.perf_counter()
    store = Store(db_path)
    for p in procs:
        p.start()
    try:
        while any(p.is_alive() for p in procs):
            time.sleep(POLL_S)
            prog = store.progress(experiment_id)
            logger.info("progress %s (%.0fs)", prog, time.perf_counter() - t0)
    except KeyboardInterrupt:
        logger.warning(
            "interrupted: terminating workers (leases will expire and be reclaimed)"
        )
        for p in procs:
            p.terminate()
    for p in procs:
        p.join()
    final = store.progress(experiment_id)
    store.close()
    logger.info("pool %s done in %.1fs: %s", pool_name, time.perf_counter() - t0, final)
    return final


__all__ = ["run_pool"]
