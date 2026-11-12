#!/usr/bin/env bash
set -euo pipefail

cd /workspace
scene_module=vision.scene
if [[ ${SIM_VISION_BACKEND:-legacy} == rgbd ]]; then
    scene_module=vision.rgbd_scene
fi
python3 -m "$scene_module" --world "${SIM_WORLD:-worlds/course.sdf}" --output /tmp/dronehacks-px4.sdf

unset SIM_MARKERFREE_GZ_SEED
noise_case=${SIM_RGBD_DEPTH_NOISE:-off}
if [[ $noise_case != off ]]; then
    if [[ $scene_module != vision.rgbd_scene || ${DRONEHACKS_SITL:-} != 1 ||
          ${GZ_PARTITION:-} != dronehacks_markerfree ]]; then
        echo 'RGBD depth noise requires the isolated marker-free PX4 SITL.' >&2
        exit 1
    fi
    SIM_MARKERFREE_GZ_SEED=$(python3 -m tools.check_gazebo_depth_noise configure "$noise_case" \
        --world /tmp/dronehacks-px4.sdf --manifest /tmp/dronehacks-rgbd-depth-noise.json)
    export SIM_MARKERFREE_GZ_SEED
    cat /tmp/dronehacks-rgbd-depth-noise.json
fi

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
    /workspace/runtime/start.sh &
pids+=("$!")

/workspace/runtime/px4-sitl.sh &
pids+=("$!")

wait -n "${pids[@]}"
