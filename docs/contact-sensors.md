# Contact-sensor controls

## Result

The isolated Gazebo controls observed the expected collider–pedestal contact pair on all nine source collision sensors. The displaced negative control observed all 27 expected body collision–reference pedestal records and zero body collision–displaced pedestal records.

This verifies the contact-sensor topic and collider-name wiring under these short test conditions. It is not a flight-safety, swept-rotor, or hardware validation.

| Source collision | Contact topic | Expected pair records | Result |
| --- | --- | ---: | --- |
| `base_link_collision_0` | `/model/course_quad/contact/base_link` | 27 | Pass |
| `base_link_collision_1` | `/model/course_quad/contact/base_link` | 27 | Pass |
| `base_link_collision_2` | `/model/course_quad/contact/base_link` | 27 | Pass |
| `base_link_collision_3` | `/model/course_quad/contact/base_link` | 27 | Pass |
| `base_link_collision_4` | `/model/course_quad/contact/base_link` | 27 | Pass |
| `rotor_0_collision` | `/model/course_quad/contact/rotor_0` | 22 | Pass |
| `rotor_1_collision` | `/model/course_quad/contact/rotor_1` | 22 | Pass |
| `rotor_2_collision` | `/model/course_quad/contact/rotor_2` | 22 | Pass |
| `rotor_3_collision` | `/model/course_quad/contact/rotor_3` | 22 | Pass |
| Body 0, displaced negative | `/model/course_quad/contact/base_link` | 0 displaced; 27 reference | Pass |

The case records include every observed timestamp and exact collision-name pair, the contact JSONL and fixture SDF SHA-256, source model hashes, per-case final simulation time, worker status, and unique partition. See the [machine-readable evidence](contact-sensor-controls-results.json), [fixture manifest](../artifacts/markerfree/contact-controls-fixtures-01/manifest.json), and [run summary](../artifacts/markerfree/contact-controls-01/summary.json). The artifacts under `artifacts/markerfree/` are workspace-local outputs; those relative links resolve in this checkout and may be absent from a GitHub source checkout.

## Test setup

Each of the ten cases ran in its own paused Gazebo world and isolated Transport partition. The positive cases use one named source collision and a tiny static pedestal below a sampled exposed point. The model begins with a 15 mm clearance. A 4 ms setup iteration followed by 40 bounded iterations advances simulation time by 164 ms total. The first observed positive pairs arrive at 60 ms, consistent with the intended short gravity drop.

The nine targets are the five named `base_link` collision boxes and four named rotor collision boxes from the unchanged `flight_x500` / `flight_x500_base` source. The leg cases use a fixture orientation selected to expose each target face; the source SDF was not edited. Rotor collision geometry remained stationary during these tests.

The negative world contains two separate named pedestals: a reference pedestal at the positive body 0 location and the test pedestal displaced by 0.5 m in X and Y. The negative passes only when the expected reference pair appears on the body topic and the displaced pair does not. Silence alone is inconclusive.

No PX4 process or hardware was used, and no motor commands were sent. These controls exercise discrete contact reporting for a short gravity drop. They do not test continuous swept-volume collision detection, rotating propellers, commanded flight, airframe safety, or physical hardware.

## Runtime and provenance

The run used Gazebo Sim 8.15.0 with DART, a 4 ms step, and container image `dronehacks-markerfree:hrt-v1.16.2` (`sha256:7bd4a86400ddfef3bd846670059bfaa1057bc8eaa46dab45d581a33148376f3f`). The exact hashes for `models/flight_x500/model.sdf`, `models/flight_x500_base/model.sdf`, and `worlds/course.sdf` are in the evidence JSON and fixture manifest; they are identical across all ten fixtures. The recorded Git HEAD was `cf807d0a0895ca5d6dcbf5dd3a53a25e4c845460`; the checkout also had unrelated uncommitted work, so the source hashes are the authoritative model provenance.

To regenerate fixtures and repeat the controls from the repository root in the simulator container, use new output directories:

```sh
docker --context colima-dronehacks-sim exec -w /workspace dronehacks-markerfree-sim-1 \
  python3 tools/check_contact_sensors.py generate \
  --output artifacts/markerfree/contact-controls-fixtures-rerun-02

docker --context colima-dronehacks-sim exec -w /workspace dronehacks-markerfree-sim-1 \
  python3 tools/run_contact_sensor_controls.py \
  --fixtures artifacts/markerfree/contact-controls-fixtures-rerun-02 \
  --output artifacts/markerfree/contact-controls-rerun-02
```

The runner subscribes to `/clock` and the expected contact topic before stepping. It uses bounded `gz service` subprocesses for pause and step requests, requires a successful process exit and the exact `data: true` response, and does not retry a timed-out multi-step request. A run exits nonzero if any case fails or is inconclusive.

## Transport request diagnosis

An earlier smoke attempt failed at its one-step setup request, before the drop. That runner version used Python `Node.request`; it combined the transport success flag and response Boolean into one result, so the failure itself did not distinguish timeout from a negative response. It produced no expected contact records and is retained at `artifacts/markerfree/contact-controls-smoke-01`.

The controlled comparisons explain why the final runner uses the CLI for control requests:

- Pause and single-step requests through Python and `gz service` succeeded in the preliminary probes, including with clock/contact subscriptions.
- In probe 03, Python `Node.request` for 40 steps timed out after about 2.003 seconds. Despite the timeout, the world reached 164 ms and recorded 27 contact messages.
- The same 40-step request through `gz service` returned `data: true` in about 0.331 seconds, also reached 164 ms, and recorded the same 27 contact messages.

A timed-out request can therefore have advanced the world. The runner does not retry multi-step requests after timeout; it treats that case as failure and preserves the partial clock/contact evidence. Probe outputs are retained under `artifacts/markerfree/contact-transport-probe-01`, `-02`, and `-03-{python,cli}`.
