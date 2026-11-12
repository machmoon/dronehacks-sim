#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker --context colima-dronehacks-sim compose -f compose.yaml stop sim
SIM_TEST_MARKERS=0 SIM_GUI=1 docker --context colima-dronehacks-sim compose \
    -f compose.px4.yaml up --no-build -d --force-recreate
mkdir -p artifacts/preview
docker --context colima-dronehacks-sim exec -d dronehacks-px4-sim-1 \
    bash -c 'exec python3 -m vision.preview > /workspace/artifacts/preview/preview.log 2>&1'
printf 'Marker-free course and raw camera: http://127.0.0.1:8081/\n'
printf 'PX4 remains disarmed. Marker-free localization is not connected yet.\n'
