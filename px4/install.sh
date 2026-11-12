#!/usr/bin/env bash
set -euo pipefail

scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$1"

build=build/px4_sitl_default
install -d /opt/px4/models
cp -a "$build/bin" "$build/etc" /opt/px4/
cp -a Tools/simulation/gz/models/{x500,x500_base} /opt/px4/models/
install -m 644 LICENSE /opt/px4/LICENSE
install -m 644 "$scripts/px4-rc.mavlink" /opt/px4/etc/init.d-posix/px4-rc.mavlink
sed -i '/^uxrce_dds_client start /d' /opt/px4/etc/init.d-posix/rcS
strip /opt/px4/bin/px4
