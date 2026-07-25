"""APScheduler wiring: a BlockingScheduler with two cron jobs (the daily pipeline
run and the daily monitoring pass), configured from config/scheduler.yaml. Every
job is wrapped so that an exception inside it is logged and swallowed rather than
killing the scheduler process -- a single bad run must never take down the whole
long-running scheduler.

APScheduler is only imported here, and this module is only imported lazily by
main.py's `schedule` command, so `main.py run`/`--help`/pytest never require it to
be installed.
"""

from __future__ import annotations

from typing import Callable

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from scheduler.monitor import run_monitoring
from scheduler.tasks import run_pipeline
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger("scheduler.scheduler")


def _guarded(func: Callable[[], None], name: str) -> Callable[[], None]:
    def wrapper() -> None:
        logger.info("Starting scheduled job: %s", name)
        try:
            func()
        except Exception:  # noqa: BLE001 - a job failure must never kill the scheduler
            logger.exception("Scheduled job %s raised an exception; scheduler continues running", name)

    return wrapper


def _trigger_from_job_config(job_config: dict) -> CronTrigger:
    return CronTrigger(
        day_of_week=job_config.get("day_of_week", "mon-fri"),
        hour=job_config.get("hour", 8),
        minute=job_config.get("minute", 30),
        timezone=job_config.get("timezone", "Asia/Kolkata"),
    )


def build_scheduler(config=None) -> BlockingScheduler:
    config = config or get_config()
    scheduler = BlockingScheduler()

    pipeline_job_config = config.get("scheduler.job") or {}
    scheduler.add_job(
        _guarded(run_pipeline, pipeline_job_config.get("id", "energy_daily_pipeline")),
        trigger=_trigger_from_job_config(pipeline_job_config),
        id=pipeline_job_config.get("id", "energy_daily_pipeline"),
        misfire_grace_time=pipeline_job_config.get("misfire_grace_time", 3600),
        coalesce=pipeline_job_config.get("coalesce", True),
        max_instances=pipeline_job_config.get("max_instances", 1),
    )

    monitor_job_config = config.get("scheduler.monitor_job") or {}
    scheduler.add_job(
        _guarded(run_monitoring, monitor_job_config.get("id", "energy_monitor")),
        trigger=_trigger_from_job_config(monitor_job_config),
        id=monitor_job_config.get("id", "energy_monitor"),
        misfire_grace_time=monitor_job_config.get("misfire_grace_time", 3600),
        coalesce=monitor_job_config.get("coalesce", True),
        max_instances=monitor_job_config.get("max_instances", 1),
    )

    return scheduler


def start(config=None) -> None:
    scheduler = build_scheduler(config)
    logger.info("Starting APScheduler (jobs: %s)", [job.id for job in scheduler.get_jobs()])
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped by user")
