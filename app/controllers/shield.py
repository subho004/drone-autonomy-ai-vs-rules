"""Reflex collision shield (S1): brakes when the sensed range is inside the
stopping envelope. It sees only the noisy ray ring, never ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.controllers.executor import A_BRAKE
from app.domain.observation import MAX_RANGE_M

SHIELD_REACTION_S = 0.08  # perception + one control tick of lag
SHIELD_MARGIN_M = 0.45  # radius + sensor noise + slack
SHIELD_MIN_CLOSING = 0.3
SHIELD_RELEASE_S = 1.2  # hold braking at least this long after the last trigger


@dataclass(slots=True)
class ReflexShield:
    """Stateless-ish brake trigger with a short release hysteresis."""

    enabled: bool = True
    a_brake: float = A_BRAKE * 0.85
    last_trigger_s: float = -1e9
    overrides: int = 0

    def reset(self) -> None:
        self.last_trigger_s = -1e9
        self.overrides = 0

    def should_brake(
        self, t_sim: float, forward_range: float, closing: float, vx: float
    ) -> bool:
        if not self.enabled:
            return False
        speed = max(vx, closing, 0.0)
        triggered = False
        if forward_range < MAX_RANGE_M * 0.98 and closing > SHIELD_MIN_CLOSING:
            d_stop = speed * SHIELD_REACTION_S + speed * speed / (2.0 * self.a_brake)
            triggered = forward_range < d_stop + SHIELD_MARGIN_M
        if triggered:
            if t_sim - self.last_trigger_s > SHIELD_RELEASE_S:
                self.overrides += 1
            self.last_trigger_s = t_sim
        return triggered or (t_sim - self.last_trigger_s) < SHIELD_RELEASE_S


__all__ = ["ReflexShield"]
