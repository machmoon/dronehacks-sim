# Timing budget evidence

The [reproducible audit](experiments/latency-audit-01/latency-summary.json) summarizes policy-05/07/08/09 telemetry. Rebuild it from the uncompressed run records with:

```sh
python3 docs/experiments/latency-audit-01/analyze_latency.py
```

The audit reports three separate quantities:

- **Policy capture age** is the simulation-time difference between the action’s originating image capture stamp and action acceptance. It is a simulator-clock freshness measure.
- **Policy wall latency** is mission-host monotonic wall time from writing a policy request to accepting its response. It includes local inference/exchange and mission polling delay. It excludes image exposure/readout, camera transport and any wait before a request becomes eligible.
- **Capture gap** is the difference between capture stamps on consecutive accepted actions. It is not a direct request-send cadence measurement. The logs do not contain a timestamp for every request send.

These values do not establish a Jetson, OAK, network, or hardware real-time budget. Simulation-time age does not convert to elapsed wall time when the simulator runs at a different rate.

## Run comparison

The compact distributions and per-phase data are in the JSON. These runs share the raw policy-depth mode, checkpoint, 1.0 speed scale, 1.8 m heading lookahead, boundary braking, and inertial tilt prior. Their sensor/normal configurations differ, so pool-level comparisons are descriptive rather than controlled performance comparisons.

| Run | Sensor / normal configuration | Accepted actions | Wall latency p50 / p95 / max (ms) | Capture age p50 / p95 / max (sim ms) |
|---|---|---:|---:|---:|
| policy-05 | Control depth, zero configured Gaussian noise; default `central2` normals | 359 | 14.1 / 20.8 / 81.2 | 48 / 80 / 100 |
| policy-07 | 10 mm Gaussian depth noise; `fals7` survey normals | 366 | 12.4 / 18.6 / 26.8 | 28 / 44 / 68 |
| policy-08 | Control depth, zero configured Gaussian noise; `fals7` survey normals | 364 | 12.1 / 18.9 / 40.7 | 28 / 60 / 92 |
| policy-09 | 10 mm Gaussian depth noise; `fals7` survey normals | 363 | 12.7 / 19.1 / 52.3 | 32 / 48 / 76 |

Per-run distributions in the JSON are the source of truth; this short table is a quick comparison. The JSON retains each run’s maximum-gap adjacent action records and nearby compact samples. The 300 ms policy-05 maximum (58.280→58.604 simulated seconds) and 352 ms policy-09 maximum (67.616→68.000 seconds) both cross waypoint handoffs. The next accepted responses had capture ages of 52 and 48 simulated milliseconds and wall latencies of 13.3 and 13.6 ms, respectively. The samples show empty action fields during the new waypoint transition. Policy-05 records a zero command at 58.604; policy-09 shows a small residual command at 67.900. That residual is consistent with the command being slewed toward zero, but telemetry has no explicit pending-request field, so it does not prove that a request was pending. The action acceptance intervals are 324 ms and 384 ms respectively, while the capture-stamp gaps are 300 ms and 352 ms.

The policy-07 and policy-08 maximum gaps are each 200 ms within waypoint 14, with no waypoint transition in the pair. Those maxima should not be conflated with the policy-05/09 handoff cases or characterized from capture gaps alone as inference dropouts. The sample is too small to establish a timing guarantee. `policy_setpoints_sent` is an aggregate count of policy-active setpoints while an action exists. Its counting interval differs from `route_elapsed_sim_s`, which starts at arming before takeoff and hover; do not divide the two to claim a setpoint frequency. Actions are held across repeated setpoints.

## Archive versus native timestamps

The `camera/frames.jsonl` recorder is throttled to about 5 Hz: its observed capture gaps are typically 200 ms. It must not be used to estimate the native navigation camera cadence. Separate PX4 `vehicle_visual_odometry.timestamp_sample` evidence in [native-capture-gap-evidence.json](native-capture-gap-evidence.json) reports roughly 48–52 ms median adjacent positive gaps in four closed logs, approximately 20 Hz. Those are native timestamp intervals, not proof of all camera callbacks, and not end-to-end inference age or processing latency.

## Hardware timing measurements to collect

For a real platform, log each stage with a clearly declared clock and correlate records by frame/request ID:

1. Camera acquisition/capture timestamp and its clock domain.
2. Host receive time on a monotonic wall clock.
3. Inference start and completion times on that same host clock.
4. Outgoing command send time and the controller’s receive/accept time, plus measured controller lag where available.
5. Frame/request IDs, dropped or repeated timestamps, and deadline misses so gaps and end-to-end age can be reconstructed.

Only those measurements can characterize deployment timing. The simulation results provide no hard real-time guarantee.
