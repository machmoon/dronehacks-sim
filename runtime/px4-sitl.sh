#!/usr/bin/env bash
set -euo pipefail

workdir=$(mktemp -d /tmp/px4-sitl.XXXXXX)
mkdir -p /workspace/artifacts/px4/log
ln -s /workspace/artifacts/px4/log "$workdir/log"

export PX4_SYS_AUTOSTART=4001
export PX4_SIM_MODEL=gz_x500
export PX4_GZ_MODEL_NAME=course_quad
export PX4_GZ_WORLD=dronehacks
export PX4_GZ_STANDALONE=1

# Vision already reports the body pose in NED. Fuse position and yaw,
# with vision as the only position and yaw aiding source.
export PX4_PARAM_EKF2_GPS_CTRL=0
export PX4_PARAM_EKF2_BARO_CTRL=0
export PX4_PARAM_EKF2_RNG_CTRL=0
export PX4_PARAM_EKF2_OF_CTRL=0
export PX4_PARAM_EKF2_EV_CTRL=11
export PX4_PARAM_EKF2_HGT_REF=3
export PX4_PARAM_EKF2_MAG_TYPE=5
export PX4_PARAM_EKF2_EV_DELAY=0
if [[ ${SIM_VISION_BACKEND:-legacy} == rgbd ]]; then
    # Camera capture timestamps are accurate, but software-rendered RGBD and
    # registration can arrive >300 ms late. Buffer enough IMU history to fuse
    # those captures instead of dropping EV and resetting local position.
    export PX4_PARAM_EKF2_DELAY_MAX=500
    export PX4_PARAM_EKF2_EVP_NOISE=0.03
fi
export PX4_PARAM_COM_RC_IN_MODE=4

# The pinned x500 and camera weigh 2.0693 kg. Four rotors at
# 8.54858e-6 N/(rad/s)^2 each need 770 rad/s to balance gravity.
# The x500 airframe maps normalized motor output linearly from 150 to 1000 rad/s.
export PX4_PARAM_MPC_THR_HOVER=0.73

exec /opt/px4/bin/px4 -d -w "$workdir" /opt/px4/etc
