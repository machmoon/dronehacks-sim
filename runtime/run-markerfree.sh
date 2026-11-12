#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p artifacts/markerfree
export GZ_GUI_PORT=${GZ_GUI_PORT:-8082}
image=${SIM_PX4_IMAGE:-dronehacks-markerfree:base}
if ! docker --context "${SIM_DOCKER_CONTEXT:-colima-dronehacks-sim}" image inspect \
    "$image" >/dev/null 2>&1; then
    if [[ $image != dronehacks-markerfree:base ]]; then
        printf 'Requested PX4 image is unavailable: %s\nBuild or load that exact image first; see docs/px4-hrt-backport.md.\n' "$image" >&2
        exit 1
    fi
    docker --context "${SIM_DOCKER_CONTEXT:-colima-dronehacks-sim}" image tag \
        dronehacks-px4:local "$image"
fi
docker --context "${SIM_DOCKER_CONTEXT:-colima-dronehacks-sim}" compose \
    -f compose.px4.yaml -f compose.markerfree.yaml up --no-build -d --force-recreate
echo 'Marker-free forward RGBD Gazebo/PX4 started. This command does not arm or fly.'
