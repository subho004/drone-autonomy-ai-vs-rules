"""Single-sourced prompt text for every model-backed provider.

The system prompt is static (cache-friendly prefix, ~550 tokens) and is built
once from ``ACTION_GLOSSARY``; the observation always goes last. Two answer
styles exist: letter aliases (chat LLMs, one token) and action names (the
Decisions endpoint, where the choice keys are the names).
"""

from __future__ import annotations

from enum import StrEnum

from app.domain.actions import ACTION_ALIASES, ACTION_GLOSSARY, Action
from app.domain.observation import Observation
from app.providers.encoding import ObservationEncoder

DECISION_QUESTION = (
    "Choose the single best discrete flight action for the next decision cycle "
    "given the state."
)
ALIAS_ANSWER_REMINDER = "Answer with one letter A-K and nothing else."


class AnswerStyle(StrEnum):
    ALIAS = "alias"
    NAME = "name"


_HEADER = (
    "You are the tactical decision module of a quadrotor. A PID controller "
    "handles stabilisation; you only choose ONE discrete action every decision "
    "cycle."
)

_FRAME = (
    "FRAME: x forward toward the goal, y left, z up. f=forward, l=left "
    "(negative = right), u=up. Bearings in degrees: 0=ahead, positive=left, "
    "negative=right. Ranges in metres, speeds in m/s. rng[F,FL,L,BL,B,BR,R,FR] "
    "is the minimum free range per sector, 10 = clear. near/clos/ttc describe "
    "the nearest obstacle (range@bearing, closing speed, time to collision). "
    "age = milliseconds since the data was measured; the drone keeps executing "
    "your previous action meanwhile. The flags line is authoritative: trust "
    "batt_state, gps, threat and goal over your own reading of the numbers."
)

_PRIORITY = (
    "PRIORITY (first match wins):\n"
    "1) batt_state=CRITICAL: EMERGENCY_LAND.\n"
    "2) threat=IMMINENT: BRAKE, or AVOID toward the side with the larger free "
    "range (never toward the side that is blocked).\n"
    "3) batt_state=LOW: RETURN_HOME.\n"
    "4) gps=LOST: HOVER, or RETURN_HOME if the goal is far.\n"
    "5) threat=CAUTION: AVOID toward the freer side, or BRAKE if both sides are "
    "blocked.\n"
    "6) goal=AT_GOAL: HOVER.\n"
    "7) otherwise FORWARD."
)


def _example(obs: Observation, action: Action, style: AnswerStyle) -> str:
    answer = ACTION_ALIASES[action] if style is AnswerStyle.ALIAS else action.name
    return f"{ObservationEncoder().encode(obs)}\n=> {answer}"


def _example_observations() -> tuple[tuple[Observation, Action], ...]:
    blocked_ahead = Observation(
        tick=0,
        t_sim=0.0,
        pos=(0.0, 0.0, 1.5),
        vel=(2.0, 0.0, 0.0),
        goal_dx=12.0,
        home_dist=4.0,
        rays=(),
        sectors=(2.1, 10.0, 10.0, 10.0, 10.0, 10.0, 3.0, 2.5),
        nearest_range=2.1,
        nearest_bearing_deg=0.0,
        closing_speed=2.0,
        ttc=1.0,
        battery_pct=80.0,
        gps_ok=True,
        last_action=int(Action.FORWARD),
        decision_age_ms=120.0,
    )
    low_battery = Observation(
        tick=0,
        t_sim=0.0,
        pos=(0.0, 0.0, 1.5),
        vel=(2.0, 0.0, 0.0),
        goal_dx=30.0,
        home_dist=30.0,
        rays=(),
        sectors=(10.0,) * 8,
        nearest_range=10.0,
        nearest_bearing_deg=0.0,
        closing_speed=0.0,
        ttc=float("inf"),
        battery_pct=20.0,
        gps_ok=True,
        last_action=int(Action.FORWARD),
        decision_age_ms=90.0,
    )
    return (
        (blocked_ahead, Action.AVOID_LEFT),
        (low_battery, Action.RETURN_HOME),
    )


def build_system_prompt(style: AnswerStyle = AnswerStyle.ALIAS) -> str:
    """Static system prompt: glossary, priority rules, two examples."""

    if style is AnswerStyle.ALIAS:
        glossary = "\n".join(
            f"{ACTION_ALIASES[a]} = {a.name}: {ACTION_GLOSSARY[a]}" for a in Action
        )
        output = "Output format: exactly one letter A-K (the letter before the '='), nothing else."
    else:
        glossary = "\n".join(f"{a.name}: {ACTION_GLOSSARY[a]}" for a in Action)
        output = "Output format: one of the 11 action names, uppercase."
    examples = "\n\n".join(
        _example(obs, action, style) for obs, action in _example_observations()
    )
    return (
        f"{_HEADER}\n\n{_FRAME}\n\nACTIONS:\n{glossary}\n\n{_PRIORITY}\n\n"
        f"EXAMPLES\n{examples}\n\n{output}"
    )


SYSTEM_PROMPT_ALIAS = build_system_prompt(AnswerStyle.ALIAS)
SYSTEM_PROMPT_NAME = build_system_prompt(AnswerStyle.NAME)


def action_criteria() -> dict[str, str]:
    """Choice criteria for the Decisions endpoint: action name -> description."""

    return {a.name: ACTION_GLOSSARY[a] for a in Action}


__all__ = [
    "ALIAS_ANSWER_REMINDER",
    "DECISION_QUESTION",
    "SYSTEM_PROMPT_ALIAS",
    "SYSTEM_PROMPT_NAME",
    "AnswerStyle",
    "action_criteria",
    "build_system_prompt",
]
