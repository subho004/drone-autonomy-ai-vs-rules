"""The shared discrete action set.

Every decision arm (Jev, chat LLMs, FSM, reflex, oracle, ...) emits one of
these actions. Integer codes follow enum order and are persisted in SQLite.
"""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    FORWARD = 0
    LEFT = 1
    RIGHT = 2
    ASCEND = 3
    DESCEND = 4
    BRAKE = 5
    HOVER = 6
    AVOID_LEFT = 7
    AVOID_RIGHT = 8
    RETURN_HOME = 9
    EMERGENCY_LAND = 10


ACTION_NAMES: tuple[str, ...] = tuple(a.name for a in Action)

# Single-character wire aliases for single-token LLM outputs.
ACTION_ALIASES: dict[Action, str] = {a: chr(ord("A") + int(a)) for a in Action}
ALIAS_TO_ACTION: dict[str, Action] = {v: k for k, v in ACTION_ALIASES.items()}

ACTION_GLOSSARY: dict[Action, str] = {
    Action.FORWARD: "keep flying toward the goal at cruise speed (+x)",
    Action.LEFT: "shift 1 m to the left (+y), keep forward motion",
    Action.RIGHT: "shift 1 m to the right (-y), keep forward motion",
    Action.ASCEND: "climb 0.5 m",
    Action.DESCEND: "descend 0.5 m",
    Action.BRAKE: "decelerate hard to a stop and hold position",
    Action.HOVER: "stop gently and hold position",
    Action.AVOID_LEFT: "evasive sidestep 1.2 m left while slowing",
    Action.AVOID_RIGHT: "evasive sidestep 1.2 m right while slowing",
    Action.RETURN_HOME: "fly back to the home point and hold",
    Action.EMERGENCY_LAND: "descend and land immediately (irreversible)",
}


def parse_action_name(text: str) -> Action | None:
    """Return the Action whose name equals ``text`` (case-insensitive) or None."""

    key = text.strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return Action[key]
    except KeyError:
        return None


__all__ = [
    "ACTION_ALIASES",
    "ACTION_GLOSSARY",
    "ACTION_NAMES",
    "ALIAS_TO_ACTION",
    "Action",
    "parse_action_name",
]
