"""Explicit ENU/FLU camera pose to PX4 local-down/FRD frame conversion.

The surveyed marker map defines the horizontal reference, not a compass.
These transforms match the existing drone's px4_odom_bridge convention.
"""
import math


def enu_to_ned(vector):
    x, y, z = vector
    return (y, x, -z)


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw*bw-ax*bx-ay*by-az*bz,
            aw*bx+ax*bw+ay*bz-az*by,
            aw*by-ax*bz+ay*bw+az*bx,
            aw*bz+ax*by-ay*bx+az*bw)


def camera_pose_to_px4(position, orientation_wxyz):
    q = quat_mul((0.0, math.sqrt(.5), math.sqrt(.5), 0.0), orientation_wxyz)
    w, x, y, z = quat_mul(q, (0.0, 1.0, 0.0, 0.0))
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    return enu_to_ned(position), (roll, pitch, yaw)


def pose_covariance(position_std=.03, angle_std=.04):
    """MAVLink upper-triangle 6x6 pose covariance; conservative fixed floors."""
    values = [0.0] * 21
    for index in (0, 6, 11):
        values[index] = position_std ** 2
    for index in (15, 18, 20):
        values[index] = angle_std ** 2
    return values
