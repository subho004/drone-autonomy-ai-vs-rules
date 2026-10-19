from __future__ import annotations

import json

import pytest
from test_providers_common import make_obs, mirror

from app.providers.encoding import (
    EncodingFormat,
    ObservationEncoder,
    battery_state,
    fmt_num,
    goal_state,
    threat_level,
)
from app.providers.prompts import SYSTEM_PROMPT_ALIAS, SYSTEM_PROMPT_NAME


def approx_tokens(text: str) -> float:
    return len(text) / 3.2


def test_hybrid_is_deterministic_and_compact() -> None:
    enc = ObservationEncoder()
    text = enc.encode(make_obs())
    assert text == enc.encode(make_obs())
    assert "rng[F,FL,L,BL,B,BR,R,FR]=3.4,7.8,9.9,10,10,10,3.6,3.4" in text
    assert "near=3.4m@-20 clos=2.1 ttc=1.6" in text
    assert 40 <= approx_tokens(text) <= 170


@pytest.mark.parametrize("fmt", list(EncodingFormat))
def test_all_formats_are_bounded_and_flagged(fmt: EncodingFormat) -> None:
    text = ObservationEncoder(fmt).encode(make_obs())
    assert approx_tokens(text) <= 200
    assert "batt_state" in text and "OK" in text


def test_json_format_parses() -> None:
    data = json.loads(ObservationEncoder("json").encode(make_obs()))
    assert data["threat"] == "CAUTION"
    assert data["sector_ranges_m"][0] == "3.4"


def test_flags_battery_gps_threat_goal() -> None:
    enc = ObservationEncoder()
    text = enc.encode(make_obs(battery_pct=5.0, gps_ok=False, ttc=0.5, goal_dx=0.1))
    assert "batt_state=CRITICAL gps=LOST threat=IMMINENT goal=AT_GOAL" in text
    assert "batt_state=LOW" in enc.encode(make_obs(battery_pct=18.0))


def test_bucket_boundaries() -> None:
    assert battery_state(7.9) == "CRITICAL"
    assert battery_state(8.0) == "LOW"
    assert battery_state(24.9) == "LOW"
    assert battery_state(25.0) == "OK"
    assert threat_level(float("inf")) == "NONE"
    assert threat_level(1.49) == "IMMINENT"
    assert threat_level(2.0) == "CAUTION"
    assert threat_level(3.0) == "NONE"
    assert goal_state(0.29) == "AT_GOAL"
    assert goal_state(0.3) == "EN_ROUTE"


def test_fmt_num_no_negative_zero_or_trailing_zero() -> None:
    assert fmt_num(-0.02) == "0"
    assert fmt_num(3.0) == "3"
    assert fmt_num(3.14159) == "3.1"
    assert fmt_num(-2.6) == "-2.6"


def test_inf_ttc_rendered() -> None:
    assert "ttc=inf" in ObservationEncoder().encode(make_obs(ttc=float("inf")))


def test_mirror_symmetry_of_encoding() -> None:
    """Left/right swap in the observation = swapped sector list, negated bearing."""

    enc = ObservationEncoder()
    obs = make_obs()
    text, flipped = enc.encode(obs), enc.encode(mirror(obs))

    def rng(t: str) -> list[str]:
        line = next(ln for ln in t.splitlines() if ln.startswith("rng["))
        return line.split("=", 1)[1].split(",")

    def bearing(t: str) -> int:
        return int(t.split("@")[1].split()[0])

    swapped = [rng(text)[i] for i in (0, 7, 6, 5, 4, 3, 2, 1)]
    assert rng(flipped) == swapped
    assert bearing(flipped) == -bearing(text)
    assert enc.encode(mirror(mirror(obs))) == text
    # Same non-lateral content in both.
    assert text.splitlines()[1] == flipped.splitlines()[1]


def test_system_prompt_is_static_and_sized() -> None:
    for prompt in (SYSTEM_PROMPT_ALIAS, SYSTEM_PROMPT_NAME):
        assert 350 <= len(prompt) / 4 <= 700
        assert "EMERGENCY_LAND" in prompt
    assert "A = FORWARD" in SYSTEM_PROMPT_ALIAS
    assert "A = FORWARD" not in SYSTEM_PROMPT_NAME
