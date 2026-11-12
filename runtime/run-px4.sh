#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
profile=dronehacks-sim
colima start --profile "$profile" --activate=false --arch aarch64 --vm-type vz \
    --cpu 4 --memory 6 --disk 16 --runtime docker
docker --context "colima-$profile" build -t dronehacks-sim:local .
docker --context "colima-$profile" build -f Dockerfile.px4 -t dronehacks-px4:local .
docker --context "colima-$profile" compose -f compose.yaml stop sim
docker --context "colima-$profile" compose -f compose.px4.yaml up --no-build -d
printf 'Gazebo GUI is available at http://127.0.0.1:%s/vnc.html?autoconnect=1&resize=scale\n' "${GZ_GUI_PORT:-8080}"
