#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# < 1 )); then
    echo 'Usage: replay-evaluation.sh RUN_NAME [mission options...]' >&2
    exit 2
fi
run_name=$1
shift
if [[ ! "$run_name" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Run name must contain only letters, numbers, underscore or hyphen.' >&2
    exit 2
fi
if [[ -e "artifacts/evaluation/$run_name" ]]; then
    echo 'Run directory already exists; choose a new name.' >&2
    exit 2
fi
docker --context colima-dronehacks-sim compose -f compose.yaml stop sim
SIM_GUI=${SIM_GUI:-0} docker --context colima-dronehacks-sim compose -f compose.px4.yaml up --no-build -d --force-recreate
mkdir -p artifacts/px4
docker --context colima-dronehacks-sim exec -d dronehacks-px4-sim-1 \
    bash -c 'exec python3 -m mission.evaluate "$@" > /workspace/artifacts/px4/evaluation.log 2>&1' \
    -- --output "/workspace/artifacts/evaluation/$run_name" -- "$@"
printf 'Recorded evaluation started: %s\n' "$run_name"
