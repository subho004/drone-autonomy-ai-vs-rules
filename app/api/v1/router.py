"""Aggregate API router.

Collects all v1 feature routers into a single router that the app
factory mounts in `main.py`.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.arms import router as arms_router
from app.api.v1.dashboard import router as dashboard_router
from app.api.v1.events import router as events_router
from app.api.v1.experiments import router as experiments_router
from app.api.v1.flights import router as flights_router
from app.api.v1.health import router as health_router
from app.api.v1.replay import router as replay_router
from app.api.v1.results import router as results_router
from app.api.v1.scenarios import router as scenarios_router

API_PREFIX = "/api/v1"

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(dashboard_router)
for _feature_router in (
    experiments_router,
    results_router,
    flights_router,
    replay_router,
    arms_router,
    scenarios_router,
    events_router,
):
    api_router.include_router(_feature_router, prefix=API_PREFIX)
