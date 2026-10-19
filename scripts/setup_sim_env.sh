#!/usr/bin/env bash
# One-time simulation environment setup (macOS arm64, Python 3.14.2, no Docker).
# pybullet has no macOS wheel: it compiles from source and needs the fdopen define
# (Apple SDK zlib header clash). gym-pybullet-drones is git-only; pinned to a SHA.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GPD_SHA="7ebad1ecabd28a7000add2d05f888aa2e837c2cc"
GPD_URL="git+https://github.com/learnsyslab/gym-pybullet-drones@${GPD_SHA}"

if [ ! -d .venv ]; then
  uv venv --python 3.14.2
fi
uv pip install -r requirements.txt

CFLAGS="-Dfdopen=fdopen" uv pip install "pybullet==3.2.7"
uv pip install --no-deps "gym-pybullet-drones @ ${GPD_URL}"

.venv/bin/python - <<'PY'
import pybullet as p
cid = p.connect(p.DIRECT)
p.disconnect(cid)
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary  # noqa: F401
print("sim env OK")
PY
