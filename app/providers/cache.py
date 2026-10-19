"""Response cache over the ``model_cache`` SQLite table.

A hit replays the stored decision with its ORIGINAL latency (so latency-aware
experiments stay faithful) and zero cost, flagged ``from_cache=True``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from typing import Any

from app.core.logging import get_logger
from app.domain.actions import Action
from app.domain.decision import Decision
from app.harness.store import utc_now

logger = get_logger(__name__)


def _encode(decision: Decision) -> str:
    return json.dumps(
        {
            "action": None if decision.action is None else int(decision.action),
            "probabilities": None
            if decision.probabilities is None
            else {a.name: p for a, p in decision.probabilities.items()},
            "confidence": decision.confidence,
            "raw": decision.raw,
            "model_id": decision.model_id,
            "tokens_in": decision.tokens_in,
            "tokens_out": decision.tokens_out,
            "cost_usd": decision.cost_usd,
            "extras": decision.extras,
        },
        sort_keys=True,
    )


def _decode(payload: str, latency_ms: float) -> Decision:
    data: dict[str, Any] = json.loads(payload)
    probs = data["probabilities"]
    extras = {k: float(v) for k, v in data["extras"].items()}
    extras["orig_cost_usd"] = float(data["cost_usd"])
    return Decision(
        action=None if data["action"] is None else Action(data["action"]),
        latency_ms=latency_ms,
        probabilities=None
        if probs is None
        else {Action[k]: float(v) for k, v in probs.items()},
        confidence=data["confidence"],
        valid=data["action"] is not None,
        raw=data["raw"],
        model_id=data["model_id"],
        tokens_in=data["tokens_in"],
        tokens_out=data["tokens_out"],
        cost_usd=0.0,
        from_cache=True,
        extras=extras,
    )


class ResponseCache:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(provider: str, model: str, params: Mapping[str, Any]) -> str:
        """sha256 over provider, model and the full request body (params + prompt)."""

        blob = json.dumps(
            {"provider": provider, "model": model, "params": params},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Decision | None:
        row = self._conn.execute(
            "SELECT response_json, latency_ms FROM model_cache WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        try:
            decision = _decode(row["response_json"], float(row["latency_ms"]))
        except (KeyError, ValueError, TypeError):
            logger.warning("Corrupt cache row ignored for key %s", key[:12])
            self.misses += 1
            return None
        self.hits += 1
        return decision

    def put(self, key: str, provider: str, decision: Decision) -> None:
        """Store a valid decision; invalid ones are never cached."""

        if decision.action is None or not decision.valid:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO model_cache(key, provider, response_json, latency_ms, "
            "created_at) VALUES (?,?,?,?,?)",
            (key, provider, _encode(decision), decision.latency_ms, utc_now()),
        )


__all__ = ["ResponseCache"]
