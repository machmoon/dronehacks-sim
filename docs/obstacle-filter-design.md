# Nominal CAD obstacle filter feasibility

The existing planar brake budgets stopping distance to the course boundary. A separate filter could reject learned velocity intents that approach surveyed interior walls. The offline probe reuses the delay-aware transfer plant and x500 multipart envelope. It leaves the trained teacher as a source of candidate ideas because the teacher uses a simulated bias, wind and randomized scene, and its sampled clearance enters a soft cost.

## Candidate design

1. Seed each preview with PX4-estimated ENU position and velocity, capture-aligned attitude, the active route stage and yaw command, and the timestamped FIFO of setpoints actually issued after slew. Use a common PX4 simulation timestamp. Carry the FIFO across stage changes and invalidate it on an estimator reset, clock discontinuity, or missing bracket.
2. Test the actor intent and a slew-to-zero alternative first. For the actor, hold intent for at most one 160 ms action period, then preview stopping. Preserve the `.8 m/s²` command slew, account for attitude motion and evaluate delay, response and gain hypotheses. Extend the preview until the issued command and FIFO are zero and the remaining motion fits a terminal clearance bound. Check the command after composition with the planar guard.
3. Broad-phase against a swept center AABB expanded by the multipart radius, then check the nearby nominal CAD boxes. Use the sampled face clearance and a bound on motion between samples to cull candidates. Run the existing continuous component check on ambiguous finalists before acceptance. A small bounded set of slower or waypoint-directed intents can be tried if the actor is rejected. Prefer the nearest certified action to the actor intent. Keep the previously certified stop as a deadline fallback when it is still valid. Escalate to the existing abort path when the current state and command history no longer support a certified stop.

Run the work in a deadline-limited worker. The live setpoint, heartbeat and health loops need to keep running while it computes. The actor period is 160 ms in simulation and the present runner checks estimate age against 60 ms in simulation. A completed preview still needs matching stage, issued-command history and a fresh estimate before use. Test deadline behavior under the target machine's actual simulation rate and load before any live use.

## Offline evidence

The nominal fixture starts at ENU `[-1.37, 1.38, .95]` with velocity `[0, -.45, 0]` m/s, yaw zero and six recent issued commands `[0, -.35, 0]` m/s. The actor requests `[0, -.495, 0]` m/s and the planar brake emits `[0, -.382, 0]` on the next 40 ms step without an intervention. Actor intent followed by braking clears the inflated CAD envelope in 3 of 8 tested delay, response and gain combinations. Immediate braking clears 8 of 8 with the same 40 mm total component margin and a 0.76 mm minimum terminal gap after accounting for residual stopping motion.

The exact step-27720 replay reproduced the randomized MAZE failure at 15.36 s. Using only the reconstructed estimator state, outgoing commands and nominal map, the probe rejects an intent 2.72 s before that failure. The early alternative increases the sampled clearance in its randomized evaluation from 5.4 mm to 12.3 mm. Some later nominal-cleared alternatives still produce unresolved overlaps in the randomized scene. The replay establishes a detectable risk signal and a map/estimate limitation. It does not establish closed-loop recovery or address the separate final-waypoint verification miss.

On one Mac CPU thread, the stock teacher measured 11.3 ms p95 over 100 calls. The nine-candidate prototype measured 59.1 ms p95 in the local MAZE replay. Its geometry alone reached 256.7 ms p95 for all 42 proxy boxes. A dense full-map check with two candidates and eight hypotheses measured 79.7 ms p95. Broad-phase selection, bounded candidate count and an asynchronous deadline are therefore part of the design.

The 25 mm reserve added to the existing 15 mm component margin is an experimental allowance. A deployment needs a measured localization and map error budget, including attitude uncertainty, state age, command-hold uncertainty and EKF resets. The eight hypotheses are sampled endpoints, so passing them does not certify every possible plant. Static CAD cannot detect an unseen obstacle, changed aperture or shifted barrier. Limit the filter to the surveyed map, or provide a separately validated sensor obstacle input before relying on it for changed geometry.

## Reproduce

Run from the marker-free worktree with the existing PyTorch environment. The artifact contains the exact checkpoint bytes and hashes the frozen sources and CAD inputs in its report.

```sh
probe=artifacts/markerfree/obstacle-filter-probe-01
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ../research/sotapilot-env/bin/python "$probe/probe.py" \
  --checkpoint "$probe/checkpoint-27720.pt" --output /tmp/obstacle-filter-probe-replay.json
```
