from __future__ import annotations

import math

import pytest

from app.domain.actions import Action
from app.providers.parsing import (
    ParseStatus,
    parse_action,
    parse_action_ex,
    probabilities_from_logprobs,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A", Action.FORWARD),
        (" k\n", Action.EMERGENCY_LAND),
        ("BRAKE", Action.BRAKE),
        ("avoid left", Action.AVOID_LEFT),
        ("avoid-right", Action.AVOID_RIGHT),
        ("**HOVER**", Action.HOVER),
        ("brake.", Action.BRAKE),
        ("The answer: HOVER", Action.HOVER),
        ('{"action": "RETURN_HOME"}', Action.RETURN_HOME),
        ("I would avoid left now", Action.AVOID_LEFT),
        ("Answer: J", Action.RETURN_HOME),
    ],
)
def test_parse_accepts(text: str, expected: Action) -> None:
    assert parse_action(text) is expected


@pytest.mark.parametrize(
    "text",
    [
        "I think LEFT or RIGHT",
        "AVOID_LEFT is better than LEFT",
        "",
        "   ",
        None,
        "maybe",
        "Z",
    ],
)
def test_parse_rejects_ambiguous_or_unknown(text: str | None) -> None:
    assert parse_action(text) is None


def test_parse_status_reasons() -> None:
    assert parse_action_ex("B")[1] is ParseStatus.EXACT
    assert parse_action_ex("brake now")[1] is ParseStatus.EXTRACTED
    assert parse_action_ex("LEFT or RIGHT")[1] is ParseStatus.AMBIGUOUS
    assert parse_action_ex("")[1] is ParseStatus.EMPTY
    assert parse_action_ex("zzz")[1] is ParseStatus.UNRECOGNISED


def test_logprobs_renormalised_over_aliases() -> None:
    entries = [
        ("F", math.log(0.5)),
        (" F", math.log(0.1)),  # duplicate letter is summed
        ("A", math.log(0.2)),
        ("The", math.log(0.1)),  # non-alias mass is dropped
    ]
    result = probabilities_from_logprobs(entries)
    assert result is not None
    probs, mass = result
    assert mass == pytest.approx(0.8)
    assert probs[Action.BRAKE] == pytest.approx(0.75)
    assert probs[Action.FORWARD] == pytest.approx(0.25)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert set(probs) == set(Action)


def test_logprobs_without_aliases_is_none() -> None:
    assert probabilities_from_logprobs([("hello", -0.1)]) is None
