"""FastAPI dependencies wiring sessions to repositories and services."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_session
from app.repositories.catalog_repository import ArmRepository, ScenarioRepository
from app.repositories.experiment_repository import ExperimentRepository
from app.repositories.flight_repository import FlightRepository
from app.repositories.trajectory_repository import TrajectoryRepository
from app.services.catalog_service import CatalogService
from app.services.experiment_service import ExperimentService
from app.services.flight_service import FlightService, ReplayService
from app.services.results_service import ResultsService

MAX_PAGE_SIZE = 500

SessionDep = Annotated[AsyncSession, Depends(get_session)]

PageParam = Annotated[int, Query(ge=1, description="1-based page number")]
PageSizeParam = Annotated[
    int, Query(ge=1, le=MAX_PAGE_SIZE, description="Items per page")
]


def get_experiment_service(session: SessionDep) -> ExperimentService:
    return ExperimentService(ExperimentRepository(session))


def get_results_service(session: SessionDep) -> ResultsService:
    return ResultsService(ExperimentRepository(session), FlightRepository(session))


def get_flight_service(session: SessionDep) -> FlightService:
    return FlightService(FlightRepository(session), ExperimentRepository(session))


def get_replay_service(session: SessionDep) -> ReplayService:
    return ReplayService(
        FlightRepository(session), ScenarioRepository(session), TrajectoryRepository()
    )


def get_catalog_service(session: SessionDep) -> CatalogService:
    return CatalogService(ArmRepository(session), ScenarioRepository(session))


ExperimentServiceDep = Annotated[ExperimentService, Depends(get_experiment_service)]
ResultsServiceDep = Annotated[ResultsService, Depends(get_results_service)]
FlightServiceDep = Annotated[FlightService, Depends(get_flight_service)]
ReplayServiceDep = Annotated[ReplayService, Depends(get_replay_service)]
CatalogServiceDep = Annotated[CatalogService, Depends(get_catalog_service)]
