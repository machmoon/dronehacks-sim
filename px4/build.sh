#!/usr/bin/env bash
set -euo pipefail

scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source_dir=/tmp/PX4-Autopilot

if [[ ! -d "$source_dir/.git" ]]; then
    git -c advice.detachedHead=false clone --depth 1 --branch v1.16.2 \
        https://github.com/PX4/PX4-Autopilot.git "$source_dir"
fi
cd "$source_dir"
test "$(git rev-parse HEAD)" = 54f0455ffcd755534539a7cf33a09a20bf71d29d
git submodule update --init --recursive --depth 1 --jobs 4 \
    src/modules/mavlink/mavlink src/drivers/gps/devices \
    src/lib/events/libevents src/lib/heatshrink/heatshrink \
    src/modules/uxrce_dds_client/Micro-XRCE-DDS-Client Tools/simulation/gz

# The optical-flow plugin is unused. The Gazebo bridge and RGB camera
# use Gazebo's Harmonic transport and standard sensor systems.
sed -i 's/^CONFIG_MODULES_SIMULATION_GZ_PLUGINS=y$/CONFIG_MODULES_SIMULATION_GZ_PLUGINS=n/' \
    boards/px4/sitl/default.px4board
# Restrict the simulator firmware's UDP listener to its own container.
sed -i 's/_myaddr.sin_addr.s_addr = htonl(INADDR_ANY);/_myaddr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);/' \
    src/modules/mavlink/mavlink_main.cpp
make -j"${PX4_BUILD_JOBS:-3}" px4_sitl

"$scripts/install.sh" "$source_dir"
