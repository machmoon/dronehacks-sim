#!/usr/bin/env bash
set -euo pipefail

cd /workspace
python3 -m vision.scene --world "${SIM_WORLD:-worlds/course.sdf}" --output /tmp/dronehacks-px4.sdf

pids=()
cleanup() {
    trap - EXIT
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 0' INT TERM

SIM_MISSION=0 SIM_WORLD=/tmp/dronehacks-px4.sdf SIM_HEADLESS_RENDERING=1 SIM_SEPARATE_GUI=1 \
    SIM_DESKTOP_SIZE="${SIM_DESKTOP_SIZE:-800x600}" \
    /usr/local/bin/dronehacks-start &
pids+=("$!")

bash /workspace/runtime/px4-sitl.sh &
pids+=("$!")

wait -n "${pids[@]}"
