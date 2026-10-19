"""Inspection scene, labels, camera frames, answer parsing and the cost cap (no network)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from app.providers import vision
from app.providers.vision import (
    CostCap,
    CostCapReachedError,
    FatalApiError,
    VisionClient,
    estimate_cost_usd,
    parse_yes_no,
)
from app.sim.inspection import (
    IMAGE_SIZES,
    WALL_Y_M,
    InspectionScene,
    PanelSpec,
    StructureSpec,
    describe_panel,
    draw_structure,
    encode_png,
    paint_panel,
    sample_pose,
)

FAKE_KEY = "sk-test-not-a-real-key"


@pytest.fixture(scope="module")
def structure() -> StructureSpec:
    return draw_structure(7, 2, 3)


@pytest.fixture(scope="module")
def scene(structure: StructureSpec) -> Iterator[InspectionScene]:
    s = InspectionScene(structure)
    yield s
    s.close()


def _red_crack_pixels(tex: np.ndarray) -> int:
    r, g, b = (tex[..., k].astype(int) for k in range(3))
    return int(np.sum((r > 70) & (r < 130) & (g < 40) & (b < 40)))


def _rust_pixels(tex: np.ndarray) -> int:
    r, g, b = (tex[..., k].astype(int) for k in range(3))
    return int(np.sum((r > 140) & (r < 200) & (g > 60) & (g < 110) & (b < 60)))


# ------------------------------------------------------------------ structure and labels
def test_structure_is_deterministic_and_half_defective(
    structure: StructureSpec,
) -> None:
    assert draw_structure(7, 2, 3) == structure
    assert draw_structure(8, 2, 3) != structure
    assert len(structure.panels) == 6 and sum(structure.labels()) == 3


def test_forced_panels_override_the_random_choice() -> None:
    st = draw_structure(3, 2, 4, n_defective=0, forced={1: True, 5: True})
    assert [i for i, v in enumerate(st.labels()) if v] == [1, 5]


def test_labels_agree_with_the_defect_list(structure: StructureSpec) -> None:
    for panel in structure.panels:
        assert panel.has_defect == bool(panel.defects)
        kinds = {d.kind for d in panel.defects}
        expected = (
            "none" if not kinds else "both" if len(kinds) == 2 else next(iter(kinds))
        )
        assert panel.label == expected
        assert (panel.smallest_defect_m > 0) == panel.has_defect


def test_texture_is_deterministic_and_carries_the_labelled_defect(
    structure: StructureSpec,
) -> None:
    seen_crack = seen_rust = False
    for panel in structure.panels:
        tex = paint_panel(panel)
        assert np.array_equal(tex, paint_panel(panel))
        if panel.label in ("crack", "both"):
            assert _red_crack_pixels(tex) > 10
            seen_crack = True
        if panel.label in ("corrosion", "both"):
            assert _rust_pixels(tex) > 50
            seen_rust = True
        if not panel.has_defect:
            assert _red_crack_pixels(tex) == 0 and _rust_pixels(tex) == 0
    assert seen_crack and seen_rust


def test_clean_panels_carry_distractor_marks(structure: StructureSpec) -> None:
    clean = [p for p in structure.panels if not p.has_defect]
    assert clean and all(p.marks for p in clean)


def test_description_is_neutral_and_mentions_every_feature(
    structure: StructureSpec,
) -> None:
    for panel in structure.panels:
        text = describe_panel(panel).lower()
        assert not any(w in text for w in ("defect", "crack", "corrosion"))
        assert text.count(";") == len(panel.defects) + len(panel.marks) - 1 or not (
            panel.defects or panel.marks
        )


# ------------------------------------------------------------------------------- camera
@pytest.mark.parametrize("size", IMAGE_SIZES)
def test_camera_output_shape_and_content(
    scene: InspectionScene, structure: StructureSpec, size: int
) -> None:
    pose = sample_pose(structure, 2)
    img = scene.render_camera(pose, 2, size)
    assert img.shape == (size, size, 3) and img.dtype == np.uint8
    assert float(img.std()) > 5.0  # not blank
    assert np.array_equal(img, scene.render_camera(pose, 2, size))  # deterministic


def test_camera_views_differ_between_panels(
    scene: InspectionScene, structure: StructureSpec
) -> None:
    a = scene.render_camera(sample_pose(structure, 0), 0, 128)
    b = scene.render_camera(sample_pose(structure, 2), 2, 128)
    assert not np.array_equal(a, b)


def test_defect_is_visible_in_the_frame(
    scene: InspectionScene, structure: StructureSpec
) -> None:
    idx = next(p.index for p in structure.panels if p.label in ("corrosion", "both"))
    img = scene.render_camera(sample_pose(structure, idx), idx, 224).astype(int)
    rust = (
        (img[..., 0] > 70)
        & (img[..., 0] - img[..., 1] > 25)
        & (img[..., 1] - img[..., 2] > 12)
    )
    assert int(rust.sum()) > 20


def test_pose_is_seeded_and_in_front_of_the_wall(structure: StructureSpec) -> None:
    panel: PanelSpec = structure.panels[4]
    pose = sample_pose(structure, 4)
    assert pose == sample_pose(structure, 4)
    assert 1.2 < WALL_Y_M - pose[1] < 1.9 and abs(pose[0] - panel.center[0]) <= 0.25


def test_closing_a_scene_removes_its_temp_files(structure: StructureSpec) -> None:
    s = InspectionScene(structure)
    tmp = s._tmp
    assert tmp.exists()
    s.close()
    assert not tmp.exists()


def test_png_roundtrip_is_lossless(
    scene: InspectionScene, structure: StructureSpec
) -> None:
    from io import BytesIO

    from PIL import Image

    img = scene.render_camera(sample_pose(structure, 1), 1, 64)
    back = np.asarray(Image.open(BytesIO(encode_png(img))).convert("RGB"))
    assert np.array_equal(img, back)


# ----------------------------------------------------------------------- answer parsing
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("YES", True),
        ("No", False),
        ("NO.", False),
        ("**YES**", True),
        ('"no"', False),
        ("Yes, there is a crack", True),
        ("The panel shows no defect", False),
        ("There is a crack, so yes.", True),
        ("yes or no?", True),  # a leading answer wins
        ("maybe", None),
        ("", None),
        ("It is unknown, not sure", None),
        ("Unfortunately I cannot say yes. No idea", None),
    ],
)
def test_parse_yes_no(text: str, expected: bool | None) -> None:
    assert parse_yes_no(text) is expected


# ---------------------------------------------------------------------------- cost cap
def test_cost_cap_blocks_before_the_limit_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    cap = CostCap(0.01, path)
    cap.check(0.002)
    cap.add(0.0085)
    with pytest.raises(CostCapReachedError):
        cap.check(0.002)
    again = CostCap(0.01, path)  # a later process sees the spend
    assert again.spent_usd == pytest.approx(0.0085) and again.calls == 1
    assert json.loads(path.read_text())["limit_usd"] == 0.01


def test_cost_estimate_uses_token_prices() -> None:
    assert estimate_cost_usd("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost_usd("gpt-4o-mini", 0, 1_000_000) == pytest.approx(0.60)
    assert estimate_cost_usd("unknown-model", 10, 10) == 0.0


def _openai_ok(text: str = "NO", tin: int = 1000, tout: int = 2) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": tin, "completion_tokens": tout},
    }


def _client(handler: Any, cap: CostCap) -> VisionClient:
    return VisionClient(
        cap,
        transport=httpx.MockTransport(handler),
        openai_key=FAKE_KEY,
        gemini_key="gem-test-not-a-real-key",
        min_pause_s=0.0,
    )


def test_client_stops_when_the_cap_would_be_passed(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_openai_ok(tin=1_000_000))  # $0.15 per call

    cap = CostCap(0.20, tmp_path / "s.json")
    client = _client(handler, cap)
    first = client.ask("gpt-4o-mini", b"\x89PNG")
    assert first.verdict is False and first.cost_usd == pytest.approx(0.15, rel=1e-3)
    client.ask("gpt-4o-mini", b"\x89PNG")  # spent 0.15 + 0.002 estimate still fits
    with pytest.raises(CostCapReachedError):
        client.ask("gpt-4o-mini", b"\x89PNG")
    assert len(calls) == 2 and cap.spent_usd == pytest.approx(0.30, rel=1e-3)
    client.close()


def test_key_travels_only_in_the_header(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "generativelanguage" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": {"parts": [{"text": "YES"}]}}],
                    "usageMetadata": {
                        "promptTokenCount": 300,
                        "candidatesTokenCount": 1,
                    },
                },
            )
        return httpx.Response(200, json=_openai_ok("YES"))

    client = _client(handler, CostCap(1.0, tmp_path / "s.json"))
    g = client.ask("gemini-3.5-flash-lite", b"png-bytes")
    o = client.ask("gpt-4o-mini", b"png-bytes")
    assert g.verdict is True and g.tokens_in == 300 and o.verdict is True
    for req in seen:
        assert FAKE_KEY not in str(req.url) and "gem-test" not in str(req.url)
        assert (
            FAKE_KEY not in req.content.decode()
            and "gem-test" not in req.content.decode()
        )
    assert seen[0].headers["x-goog-api-key"].startswith("gem-test")
    assert seen[1].headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert b'"detail":"low"' in seen[1].content.replace(b" ", b"")
    assert b"inlineData" in seen[0].content
    client.close()


def test_auth_failure_is_fatal_and_scrubbed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"bad key {FAKE_KEY}")

    client = _client(handler, CostCap(1.0, tmp_path / "s.json"))
    with pytest.raises(FatalApiError) as info:
        client.ask("gpt-4o-mini", b"x")
    assert FAKE_KEY not in str(info.value) and "***" in str(info.value)
    client.close()


def test_repeated_429_stops_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vision.time, "sleep", lambda _s: None)
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(429, text="slow down")

    client = _client(handler, CostCap(1.0, tmp_path / "s.json"))
    with pytest.raises(FatalApiError):
        client.ask("gpt-4o-mini", b"x")
    assert count == vision.MAX_CONSECUTIVE_429
    client.close()


def test_unparseable_answer_is_recorded_and_still_costed(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_openai_ok("I can't analyze images"))

    cap = CostCap(1.0, tmp_path / "s.json")
    client = _client(handler, cap)
    r = client.ask("gpt-4o-mini", b"x")
    assert r.verdict is None and r.error == "unparseable" and cap.spent_usd > 0
    client.close()
