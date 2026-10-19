"""API tests against a temp SQLite file seeded by the demo generator."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.database import build_engine, get_sessionmaker
from main import create_app
from scripts.seed_demo_db import build_demo_db
from utils.intervals import wilson_interval

N_SCENARIOS = 8
N_TRAJECTORY_SCENARIOS = 2


@pytest.fixture(scope="module")
def seeded(tmp_path_factory: pytest.TempPathFactory) -> Iterator[dict[str, Any]]:
    """Seed one complete experiment (and Parquet files) into a temp dir."""

    root = tmp_path_factory.mktemp("api_seed")
    db_path = root / "test.db"
    summary = build_demo_db(
        db_path,
        root / "data",
        n_scenarios=N_SCENARIOS,
        n_trajectory_scenarios=N_TRAJECTORY_SCENARIOS,
        queued_fraction=0.0,
        seed=7,
    )
    yield {"db_path": db_path, **summary}
    # tmp_path_factory directories are removed by pytest; nothing else to clean.


def _app_for(db_path: Path):  # type: ignore[no-untyped-def]  # FastAPI app type is noise here
    app = create_app()
    engine = build_engine(db_path)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    app.dependency_overrides[get_sessionmaker] = lambda: maker
    return app, engine


@pytest_asyncio.fixture
async def api(seeded: dict[str, Any]) -> AsyncIterator[AsyncClient]:
    app, engine = _app_for(seeded["db_path"])
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    await engine.dispose()


@pytest_asyncio.fixture
async def empty_api(tmp_path: Path) -> AsyncIterator[AsyncClient]:
    app, engine = _app_for(tmp_path / "does_not_exist.db")
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    await engine.dispose()
    assert not (tmp_path / "does_not_exist.db").exists(), (
        "read-only API must not create the DB"
    )


async def _data(client: AsyncClient, url: str) -> Any:
    response = await client.get(url)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "success"
    return body["data"]


@pytest.mark.asyncio
async def test_experiments_pagination(api: AsyncClient, seeded: dict[str, Any]) -> None:
    page = await _data(api, "/api/v1/experiments?page=1&page_size=1")
    assert page["total"] == 1 and page["pages"] == 1 and len(page["items"]) == 1
    assert page["items"][0]["n_flights"] == seeded["flights"]
    empty_page = await _data(api, "/api/v1/experiments?page=2&page_size=1")
    assert empty_page["items"] == []


@pytest.mark.asyncio
async def test_flights_pagination_and_filters(
    api: AsyncClient, seeded: dict[str, Any]
) -> None:
    exp = seeded["experiment_id"]
    page1 = await _data(api, f"/api/v1/experiments/{exp}/flights?page=1&page_size=10")
    page2 = await _data(api, f"/api/v1/experiments/{exp}/flights?page=2&page_size=10")
    assert page1["total"] == seeded["flights"]
    assert len(page1["items"]) == 10
    assert {f["id"] for f in page1["items"]}.isdisjoint(
        {f["id"] for f in page2["items"]}
    )
    collisions = await _data(
        api, f"/api/v1/experiments/{exp}/flights?outcome=collision&page_size=500"
    )
    assert collisions["items"] and all(
        f["outcome"] == "collision" for f in collisions["items"]
    )
    by_arm = await _data(
        api, f"/api/v1/experiments/{exp}/flights?arm_id=jev&page_size=500"
    )
    assert by_arm["items"] and all(f["arm_id"] == "jev" for f in by_arm["items"])
    too_big = await api.get(f"/api/v1/experiments/{exp}/flights?page_size=100000")
    assert too_big.status_code == 422


@pytest.mark.asyncio
async def test_scope_splits_headline_from_impossible_controls(
    api: AsyncClient, seeded: dict[str, Any]
) -> None:
    exp = seeded["experiment_id"]

    async def total(scope: str) -> int:
        d = await _data(
            api, f"/api/v1/experiments/{exp}/results?page_size=500&scope={scope}"
        )
        return int(sum(r["n"] for r in d["items"]))

    head, ctrl, everything = (
        await total("challenge"),
        await total("controls"),
        await total("all"),
    )
    assert head > 0 and ctrl > 0
    assert head + ctrl <= everything
    bad = await api.get(f"/api/v1/experiments/{exp}/results?scope=nonsense")
    assert bad.status_code == 422


@pytest.mark.asyncio
async def test_results_wilson_ci_matches_counts(
    api: AsyncClient, seeded: dict[str, Any]
) -> None:
    exp = seeded["experiment_id"]
    data = await _data(
        api, f"/api/v1/experiments/{exp}/results?page_size=500&scope=all"
    )
    assert data["total"] == len(data["items"]) > 0
    assert sum(r["n"] for r in data["items"]) == seeded["flights"]
    for row in data["items"]:
        low, high = wilson_interval(row["collisions"], row["n"])
        assert row["collision_ci_low"] == pytest.approx(low)
        assert row["collision_ci_high"] == pytest.approx(high)
        assert (
            row["collision_ci_low"] <= row["collision_rate"] <= row["collision_ci_high"]
        )
        assert row["collision_rate"] == pytest.approx(row["collisions"] / row["n"])


@pytest.mark.asyncio
async def test_latency_curves_and_distribution(
    api: AsyncClient, seeded: dict[str, Any]
) -> None:
    exp = seeded["experiment_id"]
    curves = await _data(api, f"/api/v1/experiments/{exp}/latency-curves")
    assert curves["series"] and curves["by_tier"] and curves["by_event"]
    fixed = next(s for s in curves["series"] if s["latency_mode"] == "fixed")
    xs = [p["latency_ms"] for p in fixed["points"]]
    assert xs == sorted(xs) and len(xs) > 1
    sampled = next(s for s in curves["series"] if s["latency_mode"] == "sampled")
    assert all(p["latency_measured"] and p["latency_ms"] > 0 for p in sampled["points"])
    dist = await _data(api, f"/api/v1/experiments/{exp}/latency-distribution")
    assert dist["series"]
    for s in dist["series"]:
        assert s["p50_ms"] <= s["p95_ms"] <= s["p99_ms"] <= s["max_ms"]
        assert s["ecdf"][-1]["p"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_progress_complete(api: AsyncClient, seeded: dict[str, Any]) -> None:
    progress = await _data(
        api, f"/api/v1/experiments/{seeded['experiment_id']}/progress"
    )
    assert progress["complete"] is True
    assert progress["done"] == progress["total"] == seeded["flights"]
    assert progress["fraction"] == 1.0 and progress["eta_s"] == 0.0
    missing = await api.get("/api/v1/experiments/999/progress")
    assert missing.status_code == 404 and missing.json()["status"] == "error"


@pytest.mark.asyncio
async def test_replay_shape(api: AsyncClient, seeded: dict[str, Any]) -> None:
    exp = seeded["experiment_id"]
    flights = await _data(api, f"/api/v1/experiments/{exp}/flights?page_size=500")
    with_traj = next(f for f in flights["items"] if f["has_trajectory"])
    replay = await _data(api, f"/api/v1/flights/{with_traj['id']}/replay")
    traj = replay["trajectory"]
    lengths = {len(v) for v in traj.values()}
    assert len(lengths) == 1 and lengths.pop() > 10
    assert replay["hz"] <= 24.0 + 1e-6
    t = traj["t"]
    assert t == sorted(t)
    assert replay["scenario"]["spec"]["goal"]
    assert (
        replay["decisions"]
        and {"obs_tick", "apply_tick", "source"} <= replay["decisions"][0].keys()
    )
    assert all(d["apply_tick"] >= d["obs_tick"] for d in replay["decisions"])
    assert any(e["kind"] in ("goal", "collision", "end") for e in replay["events"])
    json.dumps(replay, allow_nan=False)  # NaN obstacle columns must be encoded as null

    no_traj = next(f for f in flights["items"] if not f["has_trajectory"])
    assert (await api.get(f"/api/v1/flights/{no_traj['id']}/replay")).status_code == 404
    assert (await api.get("/api/v1/flights/99999999/replay")).status_code == 404


@pytest.mark.asyncio
async def test_decisions_paginated(api: AsyncClient, seeded: dict[str, Any]) -> None:
    flights = await _data(
        api, f"/api/v1/experiments/{seeded['experiment_id']}/flights?page_size=1"
    )
    fid = flights["items"][0]["id"]
    page = await _data(api, f"/api/v1/flights/{fid}/decisions?page_size=5")
    assert len(page["items"]) == 5 and page["total"] >= 5
    assert page["items"][0]["applied_action_name"]


@pytest.mark.asyncio
async def test_catalog_endpoints(api: AsyncClient) -> None:
    arms = await _data(api, "/api/v1/arms")
    assert {a["id"] for a in arms["items"]} >= {"jev", "llm_slow"}
    scenarios = await _data(api, f"/api/v1/scenarios?page_size={N_SCENARIOS}")
    assert scenarios["total"] == N_SCENARIOS
    spec = await _data(api, f"/api/v1/scenarios/{scenarios['items'][0]['id']}/spec")
    assert spec["spec"]["event"]["obstacle"]["radius"] > 0


@pytest.mark.asyncio
async def test_sse_first_event(api: AsyncClient, seeded: dict[str, Any]) -> None:
    # A complete experiment emits one progress event then `done` and closes.
    response = await api.get(f"/api/v1/experiments/{seeded['experiment_id']}/events")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    first_block = response.text.split("\n\n")[0]
    assert first_block.startswith("event: progress")
    payload = json.loads(first_block.split("data: ", 1)[1])
    assert (
        payload["experiment_id"] == seeded["experiment_id"]
        and payload["complete"] is True
    )
    assert "event: done" in response.text
    assert (await api.get("/api/v1/experiments/999/events")).status_code == 404


@pytest.mark.asyncio
async def test_dashboard_served(api: AsyncClient) -> None:
    response = await api.get("/dashboard")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Drone Autonomy: AI vs Rules" in response.text
    static = await api.get("/static/dashboard.html")
    assert static.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/experiments",
        "/api/v1/arms",
        "/api/v1/scenarios",
    ],
)
async def test_empty_db_lists_are_empty(empty_api: AsyncClient, path: str) -> None:
    page = await _data(empty_api, path)
    assert page["items"] == [] and page["total"] == 0 and page["pages"] == 0


@pytest.mark.asyncio
async def test_empty_db_unknown_experiment_is_404_not_500(
    empty_api: AsyncClient,
) -> None:
    for suffix in (
        "progress",
        "results",
        "latency-curves",
        "latency-distribution",
        "flights",
        "events",
    ):
        response = await empty_api.get(f"/api/v1/experiments/1/{suffix}")
        assert response.status_code == 404, suffix
    assert (await empty_api.get("/api/v1/flights/1/replay")).status_code == 404
    assert (await empty_api.get("/api/v1/flights/1/decisions")).status_code == 404


@pytest.mark.asyncio
async def test_session_override_type(seeded: dict[str, Any]) -> None:
    engine = build_engine(seeded["db_path"])
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        assert isinstance(session, AsyncSession)
    await engine.dispose()
