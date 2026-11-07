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
merge. Fresh branch integration results will be recorded separately below.

Physical flight and a sub-60-second marker-free finish remain unverified. The
rendered-net removal experiment still exposes a localization limitation; see
[the transfer notes](net-transfer-next-steps.md).
