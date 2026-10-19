"""Observation -> prompt text.

Pure, deterministic functions. The default ``HYBRID`` format is the "hybrid
terse" encoding F from docs/research/07 (body frame f/l/u, rounded numbers,
8 sector minima, harness-precomputed ``near / clos / ttc``) plus an explicit
semantic flags line. The live probe (docs/research/16c) showed small decision
models ignore a raw ``bat=5`` but obey ``batt_state=CRITICAL``.

Frame (matches ``app.domain.observation``): x forward, y left, z up; bearings
in degrees, 0 = ahead, positive = left. Sector order is the fixed
``SECTOR_LABELS`` order (F, FL, L, BL, B, BR, R, FR).
"""

from __future__ import annotations

import json
import math
from enum import StrEnum
from typing import Any

from app.domain.actions import Action
from app.domain.observation import Observation

SECTOR_LABELS: tuple[str, ...] = ("F", "FL", "L", "BL", "B", "BR", "R", "FR")

BATT_CRITICAL_PCT = 8.0
BATT_LOW_PCT = 25.0
TTC_IMMINENT_S = 1.5
TTC_CAUTION_S = 3.0
AT_GOAL_DX_M = 0.3
TTC_CLEAR_S = 99.0
AGE_BUCKET_MS = 10.0


class EncodingFormat(StrEnum):
    HYBRID = "hybrid"
    KEYVALUE = "keyvalue"
    JSON = "json"


# ----------------------------------------------------------------- bucketing
def battery_state(battery_pct: float) -> str:
    if battery_pct < BATT_CRITICAL_PCT:
        return "CRITICAL"
    if battery_pct < BATT_LOW_PCT:
        return "LOW"
    return "OK"


def gps_state(gps_ok: bool) -> str:
    return "OK" if gps_ok else "LOST"


def threat_level(ttc: float) -> str:
    if not math.isfinite(ttc):
        return "NONE"
    if ttc < TTC_IMMINENT_S:
        return "IMMINENT"
    if ttc < TTC_CAUTION_S:
        return "CAUTION"
    return "NONE"


def goal_state(goal_dx: float) -> str:
    return "AT_GOAL" if goal_dx < AT_GOAL_DX_M else "EN_ROUTE"


def fmt_num(value: float, digits: int = 1) -> str:
    """Rounded number with no trailing ``.0`` and no negative zero."""

    rounded = round(value, digits)
    if rounded == 0:
        return "0"
    if rounded == int(rounded):
        return str(int(rounded))
    return f"{rounded:.{digits}f}"


def fmt_ttc(ttc: float) -> str:
    if not math.isfinite(ttc) or ttc >= TTC_CLEAR_S:
        return "inf"
    return fmt_num(ttc)


def last_action_name(code: int) -> str:
    try:
        return Action(code).name
    except ValueError:
        return "NONE"


def _rounded_age(age_ms: float) -> int:
    return int(round(age_ms / AGE_BUCKET_MS) * AGE_BUCKET_MS)


# ----------------------------------------------------------------- encoders
def _fields(obs: Observation) -> dict[str, Any]:
    """Ordered, rounded field dict shared by the key=value and JSON formats."""

    return {
        "age_ms": _rounded_age(obs.decision_age_ms),
        "last_action": last_action_name(obs.last_action),
        "vel_fwd_mps": fmt_num(obs.vel[0]),
        "vel_left_mps": fmt_num(obs.vel[1]),
        "vel_up_mps": fmt_num(obs.vel[2]),
        "alt_m": fmt_num(obs.pos[2]),
        "battery_pct": fmt_num(obs.battery_pct, 0),
        "batt_state": battery_state(obs.battery_pct),
        "gps": gps_state(obs.gps_ok),
        "goal_ahead_m": fmt_num(obs.goal_dx),
        "home_dist_m": fmt_num(obs.home_dist),
        "sector_ranges_m": [fmt_num(s) for s in obs.sectors],
        "nearest_range_m": fmt_num(obs.nearest_range),
        "nearest_bearing_deg": round(obs.nearest_bearing_deg),
        "closing_mps": fmt_num(obs.closing_speed),
        "ttc_s": fmt_ttc(obs.ttc),
        "threat": threat_level(obs.ttc),
        "goal": goal_state(obs.goal_dx),
    }


PHASE_NAMES = {
    0: "FAR",
    1: "NEAR",
    2: "HERE",
    3: "PASSED",
    4: "PAD_FAR",
    5: "PAD_HERE",
    6: "PAD_NEAR",
    7: "PAD_ALIGN",
}
ALIGN_LAT_M = 0.5
ALIGN_ALT_M = 0.25

MISSION_RULES = (
    "MISSION RULES (these override priority rules 6 and 7; obstacle and battery "
    "rules still come first): "
    "if moving=YES a sidestep is still in progress: answer FORWARD (or HOVER if phase=NEAR/HERE), "
    "do not repeat it. phase=FAR: if lat!=OK answer LEFT or RIGHT as shown, else if alt!=OK answer "
    "ASCEND or DESCEND as shown, else FORWARD. phase=NEAR: if speed>0.4 answer HOVER (you coast "
    "about 1 m), if speed<=0.4 answer FORWARD. phase=HERE: if lat or alt is not OK correct it, "
    "else HOVER (the station scans while you hold still). phase=PASSED: FORWARD. "
    "phase=PAD_FAR: align lat then FORWARD. phase=PAD_NEAR: like NEAR (HOVER if speed>0.4, "
    "else FORWARD). phase=PAD_ALIGN: correct lat. phase=PAD_HERE: EMERGENCY_LAND."
)


GATE_RULES = (
    "GATE RULES (these override priority rules 6 and 7; obstacle and battery rules still come "
    "first): fly THROUGH each gate without stopping. If moving=YES answer FORWARD. Else if "
    "lat!=OK answer LEFT or RIGHT as shown; else if alt!=OK answer ASCEND or DESCEND as shown; "
    "else FORWARD. Never answer HOVER or BRAKE just because a gate is near."
)
FOLLOW_RULES = (
    "FOLLOW RULES (these override priority rules 6 and 7; obstacle and battery rules still come "
    "first): keep the target ahead of you inside the band, do not pass it and do not fall "
    "behind. gap=TOO_FAR: FORWARD. gap=TOO_CLOSE: HOVER. gap=OK: FORWARD if trend=OPENING, "
    "HOVER if trend=CLOSING, else FORWARD. If lat is LEFT or RIGHT and moving=NO answer that "
    "sidestep once, then re-check."
)
ZONE_RULES = (
    "RESTRICTED ZONES: never enter one. If state=BLOCKED and dx<3: answer HOVER if moving=YES, "
    "else the step shown (LEFT or RIGHT). If state=INSIDE leave sideways. Otherwise follow the "
    "mission rules."
)
LANDING_RULES = (
    "LANDING RULES (these override priority rules 1, 3, 6 and 7): until batt_state=CRITICAL answer "
    "FORWARD. When batt_state=CRITICAL you have about 8 seconds: choose a site with hazard=NO "
    "(a BIG site is easier to hit if you decide slowly; never choose hazard=YES). If its lat is "
    "LEFT or RIGHT, moving=NO and dx>1 answer that sidestep. When dx<=0.7 and lat=OK answer "
    "EMERGENCY_LAND, otherwise FORWARD."
)
ORB_RULES = (
    "ORB RULES (these override the mission rules while an orb is shown): an orb is homing on you "
    "and turns only slowly, so dodge LATE. While range>1.2 answer FORWARD (do not sidestep, do "
    "not answer LEFT or RIGHT). When range<=1.2 and closing is not OPENING: answer AVOID_RIGHT "
    "if the orb is LEFT, AVOID_LEFT if it is RIGHT, either if AHEAD. Once closing=OPENING go on "
    "with the mission rules."
)
BIG_SITE_RADIUS_M = 1.2


def align_lat(dy: float) -> str:
    """Which way to sidestep so the station is centred (positive dy = station on the left)."""

    if dy > ALIGN_LAT_M:
        return "LEFT"
    if dy < -ALIGN_LAT_M:
        return "RIGHT"
    return "OK"


def align_alt(dz: float) -> str:
    if dz > ALIGN_ALT_M:
        return "UP"
    if dz < -ALIGN_ALT_M:
        return "DOWN"
    return "OK"


def _follow_lines(obs: Observation) -> str:
    ex = obs.extras
    if "fol_dx" not in ex:
        return ""
    mid = 0.5 * (ex["fol_bmin"] + ex["fol_bmax"])
    gap = (
        "TOO_FAR"
        if ex["fol_dx"] > mid + 0.8
        else "TOO_CLOSE"
        if ex["fol_dx"] < mid - 0.8
        else "OK"
    )
    trend = (
        "OPENING"
        if ex["fol_rvx"] > 0.2
        else "CLOSING"
        if ex["fol_rvx"] < -0.2
        else "STEADY"
    )
    return (
        f"\nfollow: target dx={fmt_num(ex['fol_dx'])} gap={gap} lat={align_lat(ex['fol_dy'])} "
        f"trend={trend} moving={'YES' if ex.get('moving') else 'NO'}\n{FOLLOW_RULES}"
    )


def _zone_lines(obs: Observation) -> str:
    ex = obs.extras
    if "zn_dx" not in ex:
        return ""
    state = "INSIDE" if ex["zn_inside"] else "BLOCKED" if ex["zn_blocked"] else "CLEAR"
    step = "LEFT" if ex["zn_dir"] > 0 else "RIGHT" if ex["zn_dir"] < 0 else "NONE"
    return f"\nzones: nearest dx={fmt_num(ex['zn_dx'])} state={state} step={step}\n{ZONE_RULES}"


def _landing_lines(obs: Observation) -> str:
    ex = obs.extras
    if "ls_n" not in ex:
        return ""
    rows = []
    for k in (1, 2, 3):
        if f"ls{k}_dx" not in ex:
            break
        size = "BIG" if ex[f"ls{k}_r"] >= BIG_SITE_RADIUS_M else "SMALL"
        hazard = "YES" if ex[f"ls{k}_hz"] else "NO"
        rows.append(
            f"{k}) dx={fmt_num(ex[f'ls{k}_dx'])} lat={align_lat(ex[f'ls{k}_dy'])} "
            f"size={size} hazard={hazard}"
        )
    moving = "YES" if ex.get("moving") else "NO"
    return (
        f"\nsites (nearest first): {'; '.join(rows)} moving={moving}\n{LANDING_RULES}"
    )


def _task_lines(obs: Observation) -> str:
    return (
        _mission_lines(obs)
        + _zone_lines(obs)
        + _follow_lines(obs)
        + _landing_lines(obs)
        + _orb_lines(obs)
    )


def _orb_lines(obs: Observation) -> str:
    ex = obs.extras
    if "orb_range" not in ex:
        return ""
    bearing = ex["orb_bearing_deg"]
    side = "AHEAD" if abs(bearing) < 15 else ("LEFT" if bearing > 0 else "RIGHT")
    if abs(bearing) > 100:
        side = "BEHIND"
    closing = ex["orb_closing"]
    trend = "OPENING" if closing < 0.1 else ("FAST" if closing > 1.5 else "SLOW")
    return f"\norb: range={fmt_num(ex['orb_range'])} dir={side} closing={trend}\n{ORB_RULES}"


def _mission_lines(obs: Observation) -> str:
    """Mission block for the prompt; empty for ordinary (non-mission) observations."""

    ex = obs.extras
    if "stn_n" not in ex:
        return ""
    gate = bool(ex.get("gate"))
    phase = PHASE_NAMES.get(int(ex.get("st_phase", 0)), "FAR")
    pad = phase.startswith("PAD")
    head = (
        f"mission: {'landing pad' if pad else 'station ' + fmt_num(ex['stn_i'], 0) + '/' + fmt_num(ex['stn_n'], 0)} "
        f"phase={phase} moving={'YES' if ex.get('moving') else 'NO'}"
    )
    body = (
        f"target: dx={fmt_num(ex['st_dx'])} lat={align_lat(ex['st_dy'])} "
        f"alt={'OK' if pad else align_alt(ex['st_dz'])} speed={fmt_num(obs.speed)}"
    )
    rules = GATE_RULES if gate else MISSION_RULES
    return f"\n{head}\n{body}\n{rules}"


def _encode_hybrid(obs: Observation) -> str:
    f = _fields(obs)
    labels = ",".join(SECTOR_LABELS)
    status = (
        f"age={f['age_ms']} last={f['last_action']} "
        f"v={f['vel_fwd_mps']}f,{f['vel_left_mps']}l,{f['vel_up_mps']}u "
        f"alt={f['alt_m']} bat={f['battery_pct']}"
    )
    flags = (
        f"flags: batt_state={f['batt_state']} gps={f['gps']} "
        f"threat={f['threat']} goal={f['goal']}"
    )
    target = f"tgt={f['goal_ahead_m']}f home={f['home_dist_m']}"
    sectors = f"rng[{labels}]={','.join(f['sector_ranges_m'])}"
    nearest = (
        f"near={f['nearest_range_m']}m@{f['nearest_bearing_deg']} "
        f"clos={f['closing_mps']} ttc={f['ttc_s']}"
    )
    return f"{status}\n{flags}\n{target}\n{sectors}\n{nearest}{_task_lines(obs)}"


def _encode_keyvalue(obs: Observation) -> str:
    lines = []
    for key, value in _fields(obs).items():
        text = ",".join(value) if isinstance(value, list) else str(value)
        lines.append(f"{key}={text}")
    return "\n".join(lines)


def _encode_json(obs: Observation) -> str:
    return json.dumps(_fields(obs), separators=(",", ":"))


_ENCODERS = {
    EncodingFormat.HYBRID: _encode_hybrid,
    EncodingFormat.KEYVALUE: _encode_keyvalue,
    EncodingFormat.JSON: _encode_json,
}


class ObservationEncoder:
    """Deterministic ``Observation`` -> text. Same input always gives same text."""

    def __init__(self, fmt: EncodingFormat | str = EncodingFormat.HYBRID) -> None:
        self._fmt = EncodingFormat(fmt)

    @property
    def fmt(self) -> EncodingFormat:
        return self._fmt

    def encode(self, obs: Observation) -> str:
        return _ENCODERS[self._fmt](obs)


__all__ = [
    "AT_GOAL_DX_M",
    "BATT_CRITICAL_PCT",
    "BATT_LOW_PCT",
    "MISSION_RULES",
    "SECTOR_LABELS",
    "TTC_CAUTION_S",
    "TTC_IMMINENT_S",
    "EncodingFormat",
    "ObservationEncoder",
    "align_alt",
    "align_lat",
    "battery_state",
    "fmt_num",
    "fmt_ttc",
    "goal_state",
    "gps_state",
    "last_action_name",
    "threat_level",
]
