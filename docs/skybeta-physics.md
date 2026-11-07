# Sky Beta physics profile

The default marker-free PX4 scene uses the official **Sky Beta** aircraft geometry
from `sky-beta-cad.zip`, rather than a stock x500. The supplied course archive is
byte-identical to `assets/course-v3.zip`; no course geometry was replaced.

## What is sourced

The [official aircraft page](https://dronehacks.groups.stanford.edu/connect/challenge/drone)
identifies the Pixhawk 6C, Jetson Orin Nano, OAK-D S2, DYS X2807 1500KV motors,
OVONIC 4S battery and VL53L1X. The supplied STEP provides assembled geometry;
the separate STL provides the camera mount. Source SHA-256 values are stored in
`models/skybeta/geometry.json`.

Visual meshes are decimated for rendering; contact bounds and solid volumes retain
the CAD measurements. The importer converts millimetres to metres and rotates CAD front (-Y) to body
forward (+X), retaining +Z up. Body +Y points left. The origin is the Pixhawk
**enclosure centre**, an approximation to the primary IMU position. CAD part names
use a different left/right convention, so motor numbering is selected by transformed
coordinates, not the labels. PX4 uses FRD; its mixer offsets are converted from
FLU and measured about the estimated aircraft centre of mass.

## What is estimated

`vehicle/skybeta.json` is the editable physical profile. Component weights,
material densities, propeller size/mass, thrust and reaction-torque coefficients,
motor lag, aerodynamic drag and optical calibration are **not measured hardware
values**. Motor KV alone cannot determine propeller thrust. The initial propeller
radius assumes seven inches diameter; the source CAD has no propellers.

The model sums component masses, estimates each component as a uniform box,
and uses the parallel-axis theorem to compute the full inertia tensor and centre
of mass. Wiring, fasteners and the rangefinder have an explicit extra mass budget.
Propeller masses are separate links; the camera mass is included exactly once.

Gazebo supplies rigid-body motion, gravity, per-rotor quadratic thrust,
reaction torque, asymmetric spin-up/spin-down lag, rotor drag and collisions.
Component boxes approximate body contacts; standoffs use separate boxes so their
compound does not become a solid enclosure. Swept propeller cylinders provide
conservative rotor-strike contacts. These proxies are not precise surface contact
or propeller damage models. Battery sag, ground effect and detailed airframe
body drag are not calibrated or added here.

The two ideal forward cameras approximate OAK-D stereo with a 75 mm baseline.
Their lens positions are inferred from the enclosure; their intrinsics are nominal.
The left stream retains `/dronehacks/down_camera` for viewer compatibility despite
now facing forward. The right stream is `/dronehacks/oak/right`. This does not
implement DepthAI or cuVSLAM, and marker-free autonomous navigation remains pending.
The existing RGBD odometry diagnostic explicitly retains its downward x500 fixture.

## Reproduce and improve

```sh
# CAD import needs CadQuery 2.8.0 + fast-simplification 0.2.0; runtime needs NumPy.
python tools/import_skybeta.py --source ../official-drone-files
python -m vehicle.skybeta --build
python -m unittest discover -s tests -p 'test_skybeta.py'

# Actual Gazebo force-balance probes, isolated from the running PX4 world.
docker --context colima-dronehacks-sim exec dronehacks-px4-sim-1 \
  python3 -m tools.check_skybeta_physics

# Restart the default disarmed Sky Beta camera/course preview.
./runtime/preview-marker-free.sh
```

The model and PX4 environment are generated from the same profile at startup,
including motor positions, maximum/idle speeds, reaction torque ratio and computed
hover throttle. `python -m vehicle.skybeta` reports the derived quantities.
`--build` refreshes the standalone `models/skybeta/model.sdf`; regenerate it after
editing the profile. Impossible hover configurations fail rather than starting
with an unrelated x500 throttle setting.

`SIM_AIRFRAME=x500` selects the historical aircraft. Explicit `SIM_TEST_MARKERS=1`
selects x500 automatically, preserving the calibrated downward-camera diagnostic.
Combining synthetic markers with Sky Beta is rejected because the old estimator's
extrinsics would be wrong.

For calibration, first replace the component weight estimates with measured
flight-ready masses, confirm propeller diameter and motor rotation order, and
measure the optical-to-IMU transform. Then fit thrust versus RPM, motor rise/fall
response and drag from bench data or flight logs. Repeat the regression probes
and controller flight tests after each change. The saved report is
`artifacts/physics/skybeta-check.json`; it tests simulator consistency, not a
match to the physical aircraft.

The active learning experiment is a separate analytic velocity simulator. Its
current checkpoint and training process are unchanged; it does not automatically
acquire this Gazebo airframe's dynamics. Historical x500/X3 scores remain historical
and do not establish Sky Beta course clearance or flight performance.

## Verified in this pass

The standalone model and expanded course passed Gazebo SDF validation. Duplicate
course collision/visual names were resolved by renaming only visuals; geometry
and collision identifiers are unchanged. Thirty-one relevant tests passed in the
existing simulator image. Five Gazebo probes passed at the course's 4 ms physics
step: freefall, hover force balance, climb thrust, roll torque and yaw torque.
The angular probes disable aerodynamic drag to isolate the inertia/mixer check.
See [the saved physics report](skybeta-validation.json).

The live PX4 process was checked disarmed, with `CA_ROTOR0_PX=0.0959` and
`MPC_THR_HOVER=0.5564`, matching the generated profile. The forward camera stream
is live. No autonomous Sky Beta flight or physical-aircraft calibration was
performed by these checks.
