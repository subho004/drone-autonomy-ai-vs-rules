# Assets

Screenshots, videos and data used by the README and docs. Everything here is generated from the real app and data; do not edit by hand.

| Folder | What | Regenerate with |
|---|---|---|
| `demo/` | `dashboard-tour.gif` / `.mp4`: a tour of the dashboard ending in a side-by-side replay | `scripts/capture_docs_media.py` |
| `dashboard/` | Dashboard, replay and static-report screenshots (`dashboard-*.png`, `replay-*.png`, `static-report.png`) | `scripts/capture_docs_media.py` |
| `mission/winner/` | 3D render of the mission-audition winner: `flight.mp4`, `highlight.gif`, key-moment `.jpg` frames | `scripts/render_mission.py` |
| `mission/jev/` | The same for Jev 1.13 (`flight.mp4` and frames) | `scripts/render_mission.py` |
| `sims/<slug>/` | One folder per simulation (`follow`, `delivery`, `pursuit`, `landing`, `human`, `ghost`, `gates`, `canyon`, `two-drone`, `inspection`): side-by-side `compare.mp4` / `compare.gif`, screenshots (`compare-sideN-early|mid|end.jpg`), sweep charts, `results.json` | `scripts/run_sims.py`, `analyze_sims.py`, `render_sims.py` (and the A5 to A7 and A10 scripts) |
| `sims/orbs/<variant>/` | Orb Dodge game, one subfolder per variant (`orb-dodge`, `duel`, `chaser-drones`, `rule-vs-models`, `roles`, `swarm`, `tournament`), plus `results.json`: `orb-dodge`, `orb-duel`, `chaser-drones`, `rule-vs-models-*`, `roles`, `swarm*` (`.mp4`, `.gif`, screenshots) and `results.json` | `scripts/run_orb_game.py`, `scripts/render_orbs.py` |
| `mission/selection.json` | The audition table and which flights were rendered | `scripts/render_mission.py` |

Both scripts need ffmpeg; the capture script also needs `requirements/docs.txt` and `playwright install chromium`. See the [README](../README.md#quick-start).
