from __future__ import annotations

import logging
import time

from .config import AppConfig
from .pipeline import bootstrap_pipeline

LOGGER = logging.getLogger(__name__)


def run_worker() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    config = AppConfig.from_env()
    config.ensure_storage_dirs()
    pipeline = bootstrap_pipeline(config)
    task_store = pipeline.task_store

    LOGGER.info("Worker started with config: %s", config.safe_summary())

    while True:
        task_id = task_store.dequeue_task(timeout=config.worker_poll_seconds)
        if not task_id:
            time.sleep(1)
            continue
        task = task_store.get_task(task_id)
        if not task:
            LOGGER.warning("Dequeued task %s but no task payload was found", task_id)
            continue
        try:
            LOGGER.info("Processing task %s (%s)", task_id, task["filename"])
            pipeline.process_task(task_id, task["payload"])
            LOGGER.info("Completed task %s (%s)", task_id, task["filename"])
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Task %s failed: %s", task_id, exc)
            task_store.update_task(task_id, status="failed", stage="failed", progress=100, error=str(exc))
