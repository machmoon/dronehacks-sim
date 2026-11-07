#!/usr/bin/env python3
"""Read-only ordered PX4 ULog RGB-D capture cadence analysis.

Run with the local minimal pyulog implementation on PYTHONPATH, for example:
  PYTHONPATH=/tmp/dronehacks_score_ulog:$PWD /usr/bin/python3 \
    docs/experiments/analyze_native_capture_gaps.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    'policy-05': '14_20_41.ulg',
    'policy-06': '14_51_45.ulg',
    'hover-05': '14_49_06.ulg',
    'hover-06': '15_18_51.ulg',
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered)-1)*p
    low = int(position)
    high = min(low+1, len(ordered)-1)
    return ordered[low] + (ordered[high]-ordered[low])*(position-low)


def dataset(ulog, name):
    return next(d for d in ulog.data_list if d.name == name)


def armed_bounds(status):
    stamps = status.data['timestamp']
    states = status.data['arming_state']
    start = end = None
    was_armed = False
    for stamp, state in zip(stamps, states):
        armed = int(state) == 2  # vehicle_status arming_state: ARMED=2
        if armed and not was_armed:
            start = int(stamp)
        if was_armed and not armed:
            end = int(stamp)
            break
        was_armed = armed
    if start is None or end is None:
        raise RuntimeError('could not identify closed armed interval from vehicle_status')
    return start, end


def summarize(name, filename, ULog):
    log_path = ROOT/'artifacts/px4/log/2026-11-07'/filename
    ulog = ULog(str(log_path))
    if ulog._file_corrupt:
        raise RuntimeError(f'ULog parser marked file corrupt: {log_path}')
    vvo = dataset(ulog, 'vehicle_visual_odometry').data
    status = dataset(ulog, 'vehicle_status')
    armed_start, armed_end = armed_bounds(status)

    # Preserve dataset record order. Select armed samples by publication time,
    # then compute deltas on capture timestamps without sorting/deduplication.
    pub = vvo['timestamp']
    cap = vvo['timestamp_sample']
    indexes = [i for i, t in enumerate(pub) if armed_start <= int(t) <= armed_end]
    armed_caps = [int(cap[i]) for i in indexes]
    gaps = [b-a for a, b in zip(armed_caps, armed_caps[1:])]
    positive = [gap for gap in gaps if gap > 0]
    result = {
        'ulog': str(log_path.relative_to(ROOT)),
        'ulog_sha256': sha(log_path),
        'parser_marked_corrupt': bool(ulog._file_corrupt),
        'ulog_dropouts_reported': len(ulog._dropouts),
        'armed_interval_usec': [armed_start, armed_end],
        'armed_vvo_record_count': len(indexes),
        'ordered_adjacent_pair_count': len(gaps),
        'duplicate_capture_timestamp_pairs': sum(gap == 0 for gap in gaps),
        'nonmonotonic_capture_timestamp_pairs': sum(gap < 0 for gap in gaps),
        'positive_capture_gap_pair_count': len(positive),
        'positive_capture_gap_ms': {
            'median': statistics.median(positive)/1000 if positive else None,
            'p95': percentile(positive, .95)/1000,
            'p99': percentile(positive, .99)/1000,
            'max': max(positive)/1000 if positive else None,
            'gt_120ms_count': sum(gap > 120_000 for gap in positive),
        },
        'armed_receive_minus_capture_ms': {
            'count': len(indexes),
            'median': statistics.median(int(pub[i])-int(cap[i]) for i in indexes)/1000,
            'p95': percentile([int(pub[i])-int(cap[i]) for i in indexes], .95)/1000,
            'max': max(int(pub[i])-int(cap[i]) for i in indexes)/1000,
        },
        'full_log_vvo_record_count': len(pub),
        'full_log_receive_minus_capture_ms': {
            'count': len(pub),
            'median': statistics.median(int(a)-int(b) for a,b in zip(pub,cap))/1000,
            'p95': percentile([int(a)-int(b) for a,b in zip(pub,cap)], .95)/1000,
            'max': max(int(a)-int(b) for a,b in zip(pub,cap))/1000,
        },
    }

    telem_path = ROOT/'artifacts/markerfree'/name/'telemetry.jsonl'
    metadata = next(json.loads(line) for line in telem_path.open()
                    if json.loads(line).get('type') == 'metadata')
    result['telemetry_jsonl_sha256'] = sha(telem_path)
    result['flight_source_sha256'] = {
        key: metadata['runtime_source_sha256'].get(key)
        for key in ('vision/rgbd_odometry.py', 'vision/worker.py', 'mission/markerfree_run.py')
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('native-capture-gap-evidence.json'))
    args = parser.parse_args()
    try:
        from pyulog.core import ULog
    except ImportError as error:
        raise SystemExit('pyulog parser required on PYTHONPATH; see script docstring') from error
    parser_modules = [Path(sys.modules['pyulog.core'].__file__), Path(sys.modules['pyulog.exceptions'].__file__)]
    runs = {name: summarize(name, filename, ULog) for name, filename in RUNS.items()}
    result = {
        'title': 'Native RGB-D visual odometry capture cadence evidence',
        'collected_date': '2026-11-07',
        'purpose': 'Assess whether a 120 ms maximum inter-capture gap would reject cadence observed in closed native flights. This is source-timestamp evidence, not a claim about every camera callback or processing latency.',
        'proposed_limits_s': {'max_inter_capture_gap': 0.12, 'max_capture_age_at_processing': 0.30},
        'method': {
            'source': 'PX4 ULog vehicle_visual_odometry.timestamp_sample (capture time); records selected into armed interval by timestamp (publication time) using vehicle_status transitions.',
            'gap_calculation': 'Keep ULog dataset record order; compute adjacent timestamp_sample differences without sorting or deduplication. Duplicate and nonmonotonic pairs are reported separately. Gap quantiles/counts include only positive adjacent differences.',
            'receive_capture_scope': 'Report timestamp - timestamp_sample separately for armed records and full-log records; these are publication-minus-capture delays, not inter-capture gaps and not RGB-D estimator processing age.',
            'ulog_parser': 'Local minimal pyulog parser; module hashes below. Parsed vehicle_visual_odometry and vehicle_status datasets.',
            'parser_limitations': [
                'ULog topics can be logged at lower rates or with dropped records; maxima are observed archived VVO sample gaps, not proof every RGB-D callback was present.',
                'Mission frames.jsonl is throttled to about 5 Hz and is not the native continuity stream.',
                'Only closed logs were parsed; archived topic records may still be incomplete relative to the camera stream.',
                'Publication-minus-capture does not directly measure estimator processing age.'
            ]
        },
        'runs': runs,
        'aggregate_armed_intervals': {
            'run_count': len(runs),
            'ordered_adjacent_pair_count': sum(row['ordered_adjacent_pair_count'] for row in runs.values()),
            'duplicate_capture_timestamp_pairs': sum(row['duplicate_capture_timestamp_pairs'] for row in runs.values()),
            'nonmonotonic_capture_timestamp_pairs': sum(row['nonmonotonic_capture_timestamp_pairs'] for row in runs.values()),
            'positive_capture_gap_pair_count': sum(row['positive_capture_gap_pair_count'] for row in runs.values()),
            'gt_120ms_count': sum(row['positive_capture_gap_ms']['gt_120ms_count'] for row in runs.values()),
            'max_observed_positive_gap_ms': max(row['positive_capture_gap_ms']['max'] for row in runs.values()),
            'scope': 'Sum of the four separately identified armed intervals; not a pooled independent-trials reliability estimate.'
        },
        'archive_5hz_note': 'Recorded frames.jsonl replay is about 200 ms between frames (occasionally up to about 228 ms); it would be rejected by the proposed 120 ms gate and is not evidence of native camera cadence.',
        'non_claims': [
            'Finite sample of four runs; does not establish a hard camera scheduling guarantee.',
            'No simulator or flight was run for this analysis.',
            'The 120 ms value is an engineering continuity threshold, not a formal safety guarantee.'
        ],
        'parser_source_sha256': {str(path): sha(path) for path in parser_modules},
        'analysis_script_sha256': sha(Path(__file__)),
    }
    output = args.output if args.output.is_absolute() else ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(output)
    print(json.dumps({'aggregate_armed_intervals': result['aggregate_armed_intervals'], 'runs': {name: {k: row[k] for k in ('armed_vvo_record_count', 'ordered_adjacent_pair_count', 'duplicate_capture_timestamp_pairs', 'nonmonotonic_capture_timestamp_pairs', 'positive_capture_gap_pair_count', 'positive_capture_gap_ms', 'armed_receive_minus_capture_ms', 'full_log_vvo_record_count', 'full_log_receive_minus_capture_ms')} for name,row in result['runs'].items()}}, indent=2))

if __name__ == '__main__':
    main()
