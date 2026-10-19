# Contributing

Thanks for helping. This is a research benchmark, so correctness of the *measurement* matters more than features.

## Setup

```bash
bash scripts/setup_sim_env.sh      # Python 3.14.2 venv, dependencies, pybullet + gym-pybullet-drones
.venv/bin/python -m pytest -q      # no network or API keys needed
```

## Before you open a pull request

```bash
.venv/bin/ruff check . && .venv/bin/ruff format app scripts tests utils \
  && .venv/bin/mypy --explicit-package-bases . && .venv/bin/python -m pytest -q
```

All four must pass. Project conventions are in [`.github/copilot-instructions.md`](.github/copilot-instructions.md): full type hints, class-based services with injected dependencies, no `print` (use the logger), constants at the top of scripts, and layer boundaries (`domain` -> `sim`/`controllers`/`providers` -> `harness` -> `scripts`/`api`).

## Guidelines

- **Every behaviour change needs a test**, and a bug fix needs a regression test that fails without the fix.
- **Do not change the time model casually.** A decision requested at tick `t` is applied at `t + ceil(L/dt)`; lockstep timing hides latency and is only an ablation.
- **Report results with their scenario distribution.** Collision rate depends on the reaction-budget distribution; never quote one aggregate rate without it.
- **New decision arms** implement `DecisionProvider` (`app/domain/decision.py`), return an invalid `Decision` instead of raising on network or parse errors, and get the same observation, glossary and fallback policy as the others. Tune rule-based baselines on the held-out `tune` suite with an equal budget.
- **Never log, print or commit API keys.** They come from `.env` via `app.core.config.settings`.
- **New scenarios** must pass oracle certification (`app/scenarios/generator.py`).
- Keep PRs small and describe what you measured, not just what you changed.

## Reporting issues

Please include the experiment config, the commit, your Python and macOS/Linux versions, and, for latency results, your region and network (latency is network-specific).
