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
if [[ ${SIM_EV_VELOCITY:-0} == 1 ]]; then
    export PX4_PARAM_EKF2_EV_CTRL=15
    export PX4_PARAM_EKF2_EVV_NOISE=0.05
fi
export PX4_PARAM_EKF2_HGT_REF=3
export PX4_PARAM_EKF2_MAG_TYPE=5
export PX4_PARAM_EKF2_EV_DELAY=0
export PX4_PARAM_COM_RC_IN_MODE=4

# Use one physical profile for the Gazebo rotors and the PX4 mixer.
airframe=${SIM_AIRFRAME:-skybeta}
if [[ ${SIM_TEST_MARKERS:-0} == 1 && -z ${SIM_AIRFRAME:-} ]]; then
    airframe=x500
fi
case "$airframe" in
    skybeta)
        # Fail the startup if the physical profile is invalid; never silently
        # fall back to the stock x500 mixer or its hover throttle.
        python3 -m vehicle.skybeta --shell > "$workdir/airframe.env"
        source "$workdir/airframe.env"
        ;;
    # Historical 2.0693 kg x500 fixture, 150–1000 rad/s actuator range.
    x500) export PX4_PARAM_MPC_THR_HOVER=0.73 ;;
    *) echo "Unknown SIM_AIRFRAME: $airframe" >&2; exit 1 ;;
esac

exec /opt/px4/bin/px4 -d -w "$workdir" /opt/px4/etc
