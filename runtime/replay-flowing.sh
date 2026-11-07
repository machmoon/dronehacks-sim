#!/usr/bin/env bash
# Isolated replay of the historical marker-aided x500 speed experiment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
run_name=${1:?Usage: bash runtime/replay-flowing.sh RUN_NAME}
if [[ ! "$run_name" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'Run name must contain only letters, numbers, underscore or hyphen.' >&2
    exit 2
fi
root=$(pwd)
output="$root/artifacts/flowing-isolated/evaluation"
container=dronehacks-flowing-eval
docker_cmd=(docker --context colima-dronehacks-sim)
if [[ -e "$output/$run_name" || -e "$output/$run_name-course.json" ]]; then
    echo 'Run already exists; choose a new name.' >&2
    exit 2
fi
if "${docker_cmd[@]}" exec "$container" pgrep -f '^python3 -m mission.evaluate' >/dev/null 2>&1; then
    echo 'An isolated evaluation is still running; wait for its evaluation.json.' >&2
    exit 2
fi
mkdir -p "$output"
cp mission/course-flowing.json "$output/$run_name-course.json"
"${docker_cmd[@]}" rm -f "$container" >/dev/null 2>&1 || true
"${docker_cmd[@]}" run -d --init --name "$container" --shm-size=256m --cpu-shares=4096 \
    --log-opt max-size=10m --log-opt max-file=2 \
    -e SIM_GUI=0 -e SIM_TEST_MARKERS=1 -e GZ_PARTITION=dronehacks-flowing-eval \
    -e SIM_EV_VELOCITY=1 \
    -e LP_NUM_THREADS=2 -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    -e OPENCV_FOR_THREADS_NUM=1 \
    -v "$root:/workspace:ro" \
    -v "$root/artifacts/flowing-isolated:/workspace/artifacts" \
    dronehacks-px4:local bash /workspace/runtime/px4-start.sh
"${docker_cmd[@]}" exec -d "$container" bash -c \
    'exec python3 -m mission.evaluate --output "/workspace/artifacts/evaluation/$1" -- --course "/workspace/artifacts/evaluation/$1-course.json" --speed-scale 1 --startup-timeout 600 > "/workspace/artifacts/evaluation/$1.log" 2>&1' \
    -- "$run_name"
echo "Isolated test started: $output/$run_name"
