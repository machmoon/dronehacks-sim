# Patrick team branch

The `patrick` branch combines the latest simulator experiments with the
`codex/markerfree-flight` implementation. The hardware-facing ROS package is in
[Drone008_Dronehacks/patrick](https://github.com/AgastyaK87/Drone008_Dronehacks/tree/patrick).

## Navigation and reproduction

Follow the [Mac quickstart](mac-quickstart.md) for the built-image requirements,
fresh hover, full window route, recording, and independent scoring commands.
The waypoint baseline needs no learned-model service. The optional learned
controller uses the included `checkpoints/window-depth-v1.pt` and a separate
inference process. Neither simulator runner sends commands to a physical drone.

The marker-free flight fixture uses forward RGB-D and downward range, registration
to the published course survey, and PX4 EKF2. It has no added visual tags and does
not consume simulator truth for navigation. Its x500 vehicle and ideal rendered
sensors are simulation surrogates. The separate Sky Beta CAD/physics preview
remains available, but has not demonstrated the same autonomous route.

The two airframes are selected explicitly: `compose.markerfree.yaml` pins x500,
and its scene generator requests x500 even when the general scene defaults to
Sky Beta. The PX4 startup rejects a mismatched Sky Beta mixer for the RGB-D
fixture. The merge also preserves the distinct course visual/collision names
and updates the net-removal experiment to match them.

## Validation

The combined simulator suite ran 314 tests in the installed simulation image:
309 passed and five PyTorch-dependent tests skipped. The two policy test modules
also passed all eight tests in the existing host PyTorch environment. Shell syntax
checks passed for the simulator startup scripts.

Historical full-route results, including 77–78-second learned-policy runs and
82–83-second waypoint baselines, are retained with their source hashes in the
[flight inventory](flight-trials.md). They do not automatically validate this
merge. Fresh branch integration results are recorded below.

The later [frozen sprint batch](sprint-validation.md) completed five consecutive
marker-free routes in 57–58 simulated seconds. Physical flight remains unverified.
The rendered-net removal experiment still exposes a localization limitation; see
[the transfer notes](net-transfer-next-steps.md).


## Fresh integration flight

Revision `1e426b1` completed the recorded `patrick-course-01` waypoint baseline
in **85.000 simulated seconds**, including landing and disarm. All 15 waypoints
and all eight independent score checks passed. The full modeled envelope crossed
the window with 103.669 mm minimum aperture clearance after 15 mm inflation.
The preceding fresh hover passed all seven applicable checks and settled for
three seconds. No marker fixture or simulator truth was used for navigation.

This is one fresh integration repeat, not a reliability estimate. Contact-topic
coverage remains partial. The successful result uses the x500 surrogate and
rendered depth, and does not demonstrate physical Sky Beta flight or a finish
under one minute.

[Watch the recorded RGB/depth run](media/patrick-full-course.mp4) or inspect the
[machine-readable result](patrick-results.json). Raw logs remain under
`artifacts/markerfree/patrick-course-01` locally; the public result includes their
hashes. The source revision above predates only this result and video publication.
