"""Experiment configuration (YAML -> pydantic). Hashed into the run manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class ArmCfg(BaseModel):
    """One decision arm. ``provider='classical'`` selects a built-in policy by ``kind``."""

    id: str
    provider: str = "classical"
    kind: str = ""
    model: str = ""
    pool: Literal["cpu", "api"] = "cpu"
    config: dict[str, Any] = Field(default_factory=dict)
    sweep: bool = True  # fly the injected-latency sweep cells
    measured: bool = True  # fly once with the provider's own measured latency
    subsample: int | None = None  # use only the first N scenarios (API arms)
    sampled: bool = (
        False  # also fly the experiment's sampled-latency cells (config: sigma)
    )


class SuiteCfg(BaseModel):
    name: str = "pilot"
    kind: Literal[
        "challenge",
        "mission",
        "follow",
        "delivery",
        "pursuit",
        "orbs",
        "landing",
        "gates",
        "canyon",
    ] = "challenge"  # mission and the task families are certified by a reference pilot
    n: int = 200
    base_seed: int = 1000
    mix: dict[str, float] | None = None
    n_impossible: int = (
        0  # extra uncertifiable control scenarios (excluded from headline)
    )
    budget_min_s: float | None = None  # narrow the reaction-budget draw (seconds)
    budget_max_s: float | None = None


class ExperimentConfig(BaseModel):
    name: str
    suite: SuiteCfg = Field(default_factory=SuiteCfg)
    arms: list[ArmCfg]
    latency_sweep_ms: list[float] = Field(
        default_factory=lambda: [0.0, 100.0, 300.0, 1000.0]
    )
    latency_sampled_ms: list[float] = Field(default_factory=list)  # lognormal medians
    shield: list[int] = Field(default_factory=lambda: [0])
    cadence: list[str] = Field(default_factory=lambda: ["sequential"])
    reps: int = 1
    notes: str = ""

    def sha(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_config(path: str | Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(Path(path).read_text()))


__all__ = ["ArmCfg", "ExperimentConfig", "SuiteCfg", "load_config"]
