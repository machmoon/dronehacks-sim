"""Prepare a pinned SANDO evaluation config and report dependencies. Does not fly."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SANDO_COMMIT = '1a738df919f9abe60cee3c1afbecd72579da8f6f'
INTERFACES_COMMIT = 'f5697027bf87c2b940f91959d11a49cd6cfe6b58'


def revision(path):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if revision(source) != SANDO_COMMIT:
        raise ValueError('SANDO revision changed; review upstream interfaces before preparing')
    if revision(source/'deps/dynus_interfaces') != INTERFACES_COMMIT:
        raise ValueError('dynus_interfaces revision changed')
    output.mkdir(parents=True, exist_ok=True)
    config = source/'config/sando.yaml'
    shutil.copyfile(config, output/'upstream.yaml')
    # Separate overlay. The official config remains byte-identical.
    # Limits are experiment assumptions; the real footprint includes propellers
    # which are explicitly absent from the downloaded Sky Beta CAD.
    overrides = {
        'sim_env': 'fake_sim', 'use_hardware': False, 'vehicle_type': 'uav',
        'visual_level': 0, 'use_benchmark': False, 'ignore_other_trajs': True,
        'use_global_pc': True, 'environment_assumption': 'static',
        'force_goal_z': False, 'skip_initial_yawing': True,
        'v_max': .45, 'a_max': .8, 'j_max': 2.0, 'w_max': .4,
        'dynamic_constraint_type': 'L2', 'drone_bbox': [.72, .72, .72],
        'x_min': -3.048, 'x_max': 3.048, 'y_min': -3.048, 'y_max': 3.048,
        'z_min': .36, 'z_max': 1.35, 'sando_map_res': .05,
        'inflation_hgp': .40, 'obst_position_error': .04,
        'goal_radius': .08, 'goal_seen_radius': .25, 'horizon': 3.0,
        'sfc_size': [1.5, 1.5, 1.0], 'initial_wdx': 6.096,
        'initial_wdy': 6.096, 'initial_wdz': 3.048,
        'max_gurobi_comp_time_sec': .1, 'use_free_start': False,
        'use_free_goal': False, 'use_soft_cost_obstacles': False,
        'hover_avoidance_enabled': False, 'dc': .02,
    }
    lines = ['/**:', '  ros__parameters:']
    lines.extend('    '+key+': '+json.dumps(value) for key, value in overrides.items())
    (output/'course-overlay.yaml').write_text('\n'.join(lines)+'\n')
    license_file = Path(os.environ.get('GRB_LICENSE_FILE', str(Path.home()/'gurobi.lic')))
    report = {
        'upstream_url': 'https://github.com/mit-acl/sando', 'source': str(source),
        'upstream_commit': SANDO_COMMIT, 'interfaces_commit': INTERFACES_COMMIT,
        'license': 'BSD-3-Clause (source); separate Gurobi runtime license required',
        'upstream_config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
        'host': {'system': platform.system(), 'architecture': platform.machine()},
        'ros2_on_path': shutil.which('ros2') is not None,
        'python_rclpy_available': importlib.util.find_spec('rclpy') is not None,
        'gurobi_license_file_present': license_file.is_file(),
        'gurobi_license_validated': False, 'planner_built': False, 'planner_executed': False,
        'hardware_authorized': False,
        'notes': [
            'Provided upstream installer downloads x86 Linux Gurobi; ARM requires armlinux64.',
            'Educational purpose does not provide a Gurobi runtime license automatically.',
            'The overlay is not tuned or verified by SANDO execution.',
            'Use NX01 namespace suffix: upstream parses its final two characters as an integer.',
            'Use fake_sim plus a real obstacle cloud; rviz_only starts with an empty map.',
            'Goal.power defaults false in the publisher; never use it to arm or disarm.',
            'Do not use the upstream cmd_vel converter: it implements planar ground-robot tracking.',
            'No hardware deployment or real-time claim follows from these preparation checks.',
        ],
        'evaluation_topics': {'state': '/sando_eval/NX01/state',
                              'terminal_goal': '/sando_eval/NX01/term_goal',
                              'trajectory_sample': '/sando_eval/NX01/goal',
                              'map': '/sando_eval/published_course_cloud'},
    }
    (output/'readiness.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT.parent/'research/sando')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/sando-preparation')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
