#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ${SIM_TEST_MARKERS:-0} != 1 ]]; then
    echo 'The existing flight controller requires artificial test markers and cannot navigate the real course.' >&2
    echo 'Use ./runtime/preview-marker-free.sh for the current course. Marker-free flight is not connected yet.' >&2
    echo 'For the historical diagnostic only: SIM_TEST_MARKERS=1 ./runtime/replay-px4.sh course' >&2
    exit 2
fi
task=${1:-hover}
if [[ "$task" != hover && "$task" != course ]]; then
    echo "Usage: $0 [hover|course] [mission options...]" >&2
    exit 2
fi
if (( $# )); then shift; fi
docker --context colima-dronehacks-sim compose -f compose.yaml stop sim
docker --context colima-dronehacks-sim compose -f compose.px4.yaml \
    up --no-build -d --force-recreate
mkdir -p artifacts/px4
docker --context colima-dronehacks-sim exec -d dronehacks-px4-sim-1 \
    bash -c 'exec python3 -m mission.px4_run "$@" > /workspace/artifacts/px4/mission.log 2>&1' \
    -- --task "$task" "$@"
printf 'PX4 %s test started. Viewer: http://127.0.0.1:8081/\n' "$task"
printf 'Progress: tail -f artifacts/px4/mission.log\n'
