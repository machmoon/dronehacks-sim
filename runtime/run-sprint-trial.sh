#!/usr/bin/env bash
# One isolated, recorded full-course trial. Never connects to hardware.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if (( $# != 2 )); then
    echo 'Usage: bash runtime/run-sprint-trial.sh RUN_NAME HOVER_SUMMARY' >&2
    exit 2
fi
name=$1
hover=$2
if [[ ! "$name" =~ ^[A-Za-z0-9_-]+$ || ! -f "$hover" ]]; then
    echo 'Use a simple unique run name and an existing verified hover summary.' >&2
    exit 2
fi
output="artifacts/markerfree/$name"
if [[ -e "$output" ]]; then
    echo 'Run output already exists; recordings are never overwritten.' >&2
    exit 2
fi
python3 -c 'import shutil,sys; sys.exit(0 if shutil.disk_usage(".").free >= 80*1024**2 else "At least 80 MiB free is required before a recorded trial.")'
context=${SIM_DOCKER_CONTEXT:-colima-dronehacks-sim}
project=${SIM_COMPOSE_PROJECT:-dronehacks-sprint}
container="${project}-sim-1"
export SIM_PX4_IMAGE=${SIM_PX4_IMAGE:-dronehacks-markerfree:hrt-v1.16.2}
export SIM_GUI=0
export GZ_GUI_PORT=${GZ_GUI_PORT:-8084}
docker --context "$context" image inspect "$SIM_PX4_IMAGE" >/dev/null
docker --context "$context" compose -p "$project" -f compose.px4.yaml \
    -f compose.markerfree.yaml up --no-build -d --force-recreate
set +e
docker --context "$context" exec -w /workspace "$container" \
    python3 -m mission.markerfree_evaluate --output "$output" --wall-timeout 900 -- \
    --task course --course mission/course-window-sprint.json --heading-lookahead 1.8 \
    --hover-summary "$hover" --launch-position -2.5146 0.9144 0.24 \
    --launch-yaw-deg 90 --use-px4-tilt --survey-normals fals7
mission_exit=$?
docker --context "$context" exec -w /workspace "$container" \
    python3 -m tools.score_markerfree "$output" --output "$output/score.json"
score_exit=$?
set -e
python3 - "$output" <<'PY'
import json, sys
from pathlib import Path
from tools.report_sprint import audit_attempt
report = audit_attempt(sys.argv[1])
Path(sys.argv[1], 'sprint-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k: report[k] for k in ('run_id', 'passed', 'elapsed_sim_s', 'failures')}))
sys.exit(0 if report['passed'] else 1)
PY
(( mission_exit == 0 && score_exit == 0 ))
