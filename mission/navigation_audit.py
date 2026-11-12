"""Fail closed if this isolated SITL can inject a simulator odometry feed."""
import hashlib
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

TRUTH_TOPIC = '/model/course_quad/odometry_with_covariance'


def _read_model_sources(world, model_roots):
    pending, seen, sources = [Path(world)], set(), []
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        if len(seen) > 40:
            raise RuntimeError('unexpectedly large model include graph')
        raw = path.read_bytes()
        tree = ET.fromstring(raw)
        sources.append({'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()})
        for model in tree.iter('model'):
            if model.get('name', '').startswith('vision_landmarks'):
                raise RuntimeError('artificial landmark fixture is present')
        for plugin in tree.iter('plugin'):
            description = (plugin.get('name', '') + plugin.get('filename', '')).lower()
            if 'odometry' in description or 'posepublisher' in description:
                raise RuntimeError('simulator pose/odometry publisher plugin is present')
        for include in tree.iter('include'):
            uri = include.findtext('uri', '')
            if not uri.startswith('model://'):
                raise RuntimeError('unreviewed model include: ' + uri)
            relative = uri[len('model://'):]
            matches = [Path(root) / relative / 'model.sdf' for root in model_roots]
            resolved = next((p for p in matches if p.is_file()), None)
            if resolved is None:
                raise RuntimeError('unresolved model include: ' + uri)
            pending.append(resolved)
    return sources


def assert_no_truth_publisher():
    result = subprocess.run(['gz', 'topic', '-i', '-t', TRUTH_TOPIC],
                            capture_output=True, text=True, timeout=6, check=True)
    output = result.stdout + result.stderr
    if 'No publishers on topic [' + TRUTH_TOPIC + ']' not in output:
        raise RuntimeError('simulator odometry topic has a publisher or could not be audited')
    return output.strip()


def audit_navigation_sources(world='/tmp/dronehacks-px4.sdf'):
    if os.environ.get('DRONEHACKS_SITL') != '1' or os.environ.get('GZ_PARTITION') != 'dronehacks_markerfree':
        raise RuntimeError('navigation audit requires the isolated marker-free SITL partition')
    roots = os.environ.get('GZ_SIM_RESOURCE_PATH', '').split(':')
    sources = _read_model_sources(world, roots)
    topic_state = assert_no_truth_publisher()
    return {'world_and_model_sources': sources, 'truth_topic': TRUTH_TOPIC,
            'truth_topic_publishers': 0, 'topic_audit': topic_state,
            'artificial_landmark_models': 0,
            'position_aiding': 'MAVLink camera/depth estimate only; EKF parameters verified separately',
            'static_survey_used': True, 'live_simulator_position_used': False}


if __name__ == '__main__':
    import json
    print(json.dumps(audit_navigation_sources(), indent=2))
