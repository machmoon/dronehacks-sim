#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p artifacts/camera-slam
export GZ_GUI_PORT=${GZ_GUI_PORT:-8094}
docker --context "${SIM_DOCKER_CONTEXT:-colima-dronehacks-sim}" compose \
    -f compose.px4.yaml -f compose.camera-slam.yaml up --no-build -d --force-recreate
echo 'Camera navigation simulator started; disarmed. See docs/camera-slam.md.'
