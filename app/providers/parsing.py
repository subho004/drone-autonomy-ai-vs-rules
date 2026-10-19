"""Model output -> Action, and logprobs -> probabilities.

Policy (docs/research/07 section 6, MASTER-PLAN section 9): exact alias or
name, else a unique extraction, else fail. Ambiguity is a failure; a direction
is never guessed and the first match never wins.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from enum import StrEnum

from app.domain.actions import (
    ACTION_NAMES,
    ALIAS_TO_ACTION,
    Action,
    parse_action_name,
)

_NAME_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(ACTION_NAMES, key=len, reverse=True)) + r")\b"
)
_COMPOUND_PATTERN = re.compile(
    r"\b(AVOID|RETURN|EMERGENCY)[ -](LEFT|RIGHT|HOME|LAND)\b"
)
_LABELLED_ALIAS = re.compile(r"^(?:ANSWER|ACTION|CHOICE)\s*[:=]\s*([A-K])\b")
_WRAPPER_CHARS = " \t\r\n\"'`*_.,;:!()[]{}<>"


class ParseStatus(StrEnum):
    EXACT = "exact"
    EXTRACTED = "extracted"
    EMPTY = "empty"
    AMBIGUOUS = "ambiguous"
    UNRECOGNISED = "unrecognised"


def parse_action_ex(text: str | None) -> tuple[Action | None, ParseStatus]:
    """Parse ``text`` into an Action plus how the decision was reached."""

    if text is None or not text.strip():
        return None, ParseStatus.EMPTY
    stripped = text.strip(_WRAPPER_CHARS)
    if len(stripped) == 1 and stripped.upper() in ALIAS_TO_ACTION:
        return ALIAS_TO_ACTION[stripped.upper()], ParseStatus.EXACT
    named = parse_action_name(stripped)
    if named is not None:
        return named, ParseStatus.EXACT

    upper = _COMPOUND_PATTERN.sub(r"\1_\2", text.upper())
    found = {m.group(1) for m in _NAME_PATTERN.finditer(upper)}
    if len(found) == 1:
        return Action[next(iter(found))], ParseStatus.EXTRACTED
    if len(found) > 1:
        return None, ParseStatus.AMBIGUOUS
    labelled = _LABELLED_ALIAS.match(upper.strip(_WRAPPER_CHARS))
    if labelled:
        return ALIAS_TO_ACTION[labelled.group(1)], ParseStatus.EXTRACTED
    return None, ParseStatus.UNRECOGNISED


def parse_action(text: str | None) -> Action | None:
    return parse_action_ex(text)[0]


def probabilities_from_logprobs(
    top_logprobs: Iterable[tuple[str, float]],
) -> tuple[dict[Action, float], float] | None:
    """Softmax over the letter-alias entries of one token position.

    Returns ``(probs, alias_mass)``: probabilities renormalised over the 11
    aliases (all 11 keys present) and the raw un-renormalised alias mass, a
    free quality signal (low mass = the model wanted something else). Entries
    for the same letter (``"F"``, ``" F"``) are summed. None when no alias
    token is present.
    """

    mass: dict[Action, float] = {}
    for token, logprob in top_logprobs:
        action = ALIAS_TO_ACTION.get(token.strip().upper())
        if action is None or not math.isfinite(logprob):
            continue
        mass[action] = mass.get(action, 0.0) + math.exp(logprob)
    total = sum(mass.values())
    if total <= 0.0:
        return None
    probs = {a: mass.get(a, 0.0) / total for a in Action}
    return probs, total


__all__ = [
    "ParseStatus",
    "parse_action",
    "parse_action_ex",
    "probabilities_from_logprobs",
]
