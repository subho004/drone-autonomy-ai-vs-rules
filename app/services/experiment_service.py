"""Experiment listing and progress."""

from __future__ import annotations

from datetime import UTC, datetime

from app.core.exceptions import NotFoundError
from app.repositories.experiment_repository import ExperimentRepository
from app.schemas.api import ArmProgress, ExperimentOut, Page, ProgressOut
from app.services.paging import make_page

TERMINAL_STATUSES = ("complete", "failed")
RATE_WINDOW = 500  # most recent completions used for the throughput estimate
STALL_SECONDS = 300.0  # no completion for this long: no ETA


class ExperimentService:
    def __init__(self, repo: ExperimentRepository) -> None:
        self._repo = repo

    async def list_experiments(self, page: int, page_size: int) -> Page[ExperimentOut]:
        rows, total = await self._repo.list_page(page, page_size)
        return make_page([ExperimentOut(**r) for r in rows], total, page, page_size)

    async def get_progress(self, experiment_id: int) -> ProgressOut:
        experiment = await self._repo.get(experiment_id)
        if experiment is None:
            raise NotFoundError(f"Experiment '{experiment_id}' not found")
        counts = await self._repo.status_counts(experiment_id)
        per_arm = await self._repo.per_arm_counts(experiment_id)
        finish_times = await self._repo.recent_finish_times(experiment_id, RATE_WINDOW)

        total = sum(counts.values())
        done = sum(counts.get(s, 0) for s in TERMINAL_STATUSES)
        remaining = total - done
        rate_per_s = _rate_per_second(finish_times)
        eta_s = (
            remaining / rate_per_s
            if rate_per_s and remaining
            else (0.0 if not remaining else None)
        )
        return ProgressOut(
            experiment_id=experiment_id,
            status=str(experiment["status"]),
            total=total,
            counts=counts,
            done=done,
            remaining=remaining,
            fraction=(done / total) if total else 0.0,
            rate_per_min=rate_per_s * 60.0 if rate_per_s else None,
            eta_s=eta_s,
            complete=total > 0 and remaining == 0,
            per_arm=[
                ArmProgress(
                    arm_id=str(r["arm_id"]),
                    total=int(r["total"]),
                    complete=int(r["complete"] or 0),
                )
                for r in per_arm
            ],
        )


def _rate_per_second(finish_times_desc: list[str]) -> float | None:
    """Completions per second from newest-first ISO timestamps; None if idle/unknown."""

    if len(finish_times_desc) < 2:
        return None
    try:
        newest = datetime.fromisoformat(finish_times_desc[0])
        oldest = datetime.fromisoformat(finish_times_desc[-1])
    except ValueError:
        return None
    span = (newest - oldest).total_seconds()
    if span <= 0:
        return None
    now = datetime.now(UTC if newest.tzinfo else None)
    if (now - newest).total_seconds() > STALL_SECONDS:
        return None
    return (len(finish_times_desc) - 1) / span
