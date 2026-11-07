# Apple Silicon marker-free quickstart

This quickstart targets the repository branch `patrick` on an Apple Silicon Mac. It starts the Gazebo Harmonic/PX4 SITL project and describes the current learned-policy reproduction path. It does not establish transfer to a physical camera or aircraft.

A fresh clone does not include run artifacts, Docker images, or the optional patched PX4 image. The [tracked-export runtime check](experiments/clean-export-01/README.md) passed a hover and full policy route using existing dependencies and the installed simulator image. The commands below describe the expected setup; a complete clean-clone build has not been run and verified today. The Ubuntu image tag and OSRF apt packages are not version-locked, even though the PX4 source revision is pinned. Read the [runtime source-map scope](evidence-provenance.md) before treating preserved run hashes as a clean-clone check.

## Clone and build the simulator images

Install Docker CLI and Colima. From a new terminal:

```sh
git clone --branch patrick https://github.com/machmoon/dronehacks-sim.git
cd dronehacks-sim

colima start --profile dronehacks-sim --activate=false \
  --arch aarch64 --vm-type vz --cpu 4 --memory 6 --disk 16 --runtime docker

docker --context colima-dronehacks-sim build -t dronehacks-sim:local .
docker --context colima-dronehacks-sim build \
  -f Dockerfile.px4 -t dronehacks-px4:local .
```

The PX4 Dockerfile uses the base image produced by the first build and compiles PX4 v1.16.2 from the repository's pinned source commit. The standard runtime image is `dronehacks-px4:local`. The marker-free runtime's default image tag is `dronehacks-markerfree:base`; create that tag without starting a container:

```sh
docker --context colima-dronehacks-sim image tag \
  dronehacks-px4:local dronehacks-markerfree:base
```

## Build and select the tested HRT image

The latest policy reproductions use the separately patched PX4 HRT image. It is not included in a fresh clone. Follow [the HRT build and verification instructions](px4-hrt-backport.md) after creating `dronehacks-markerfree:base`. The build uses an isolated container and a PX4 source compile. The recorded smoke, publisher soak, and hover are finite checks; they do not prove that the intermittent scheduler issue is eliminated.

The unpatched `dronehacks-markerfree:base` image remains the launcher default and can behave differently around PX4 battery/system-power publication. Select the patched image explicitly for the latest-policy reproductions:

```sh
SIM_PX4_IMAGE=dronehacks-markerfree:hrt-v1.16.2 \
  SIM_RGBD_DEPTH_NOISE=gaussian-10mm ./runtime/run-markerfree.sh
```

Use `SIM_RGBD_DEPTH_NOISE=control` for the nominal-depth control, or `off` for the unmodified depth renderer. Keep the same selection for the hover and route. This launcher starts the dedicated `dronehacks-markerfree` Compose project and does not arm the aircraft.

## Host viewer and policy service

The viewer uses the Python standard library. The learned-policy service uses the local CPU with the checked-in packaged v1 checkpoint. Its observed host runtime was Apple Silicon arm64, Python 3.11.7, PyTorch 2.15.0, and NumPy 2.4.6. The pins in [`requirements-policy.txt`](../requirements-policy.txt) cover these two direct packages; they are version pins, not a transitive lock. PyPI references: [torch 2.15.0](https://pypi.org/project/torch/2.15.0/) and [NumPy 2.4.6](https://pypi.org/project/numpy/2.4.6/).

Create the host environment (not executed as part of this documentation task):

```sh
python3.11 --version  # use Python 3.11.7 on arm64
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-policy.txt
```

In separate host terminals, start the policy service and viewer:

```sh
.venv/bin/python -m learning.inference \
  --checkpoint checkpoints/window-depth-v1.pt \
  --exchange artifacts/markerfree/policy-exchange
```

```sh
.venv/bin/python -m web.server
```

Open <http://127.0.0.1:8083>. Gazebo's noVNC display is at <http://127.0.0.1:8082/vnc.html?autoconnect=1&resize=scale>. Check that those local ports are available before starting the simulator.

The packaged checkpoint SHA-256 is `a1919ecdd2149515e4875f4694c52733e269345ef95cad6b074099bfdbff8b26`. Its manifest and validation history are in [policy-checkpoint.md](policy-checkpoint.md).

## Fresh hover and policy route

Run these commands from one host shell at the repository root so the unique UTC output paths remain available. The run evidence directory is created on first use; a fresh clone has no prior hover or score artifacts.

```sh
MARKERFREE_RUN_UTC=$(date -u +%Y%m%dT%H%M%SZ)
MARKERFREE_HOVER_DIR="artifacts/markerfree/quickstart-${MARKERFREE_RUN_UTC}-hover"
MARKERFREE_POLICY_DIR="artifacts/markerfree/quickstart-${MARKERFREE_RUN_UTC}-policy"
MARKERFREE_CHECKPOINT_SHA=a1919ecdd2149515e4875f4694c52733e269345ef95cad6b074099bfdbff8b26
MARKERFREE_CONTAINER=dronehacks-markerfree-sim-1
```

The examples below use synthetic 10 mm Gaussian depth noise, matching the noisy route setting; replace `gaussian-10mm` with `control` for a nominal-depth run. Use FALS7 survey normals for both the fresh hover and route, and wait for the hover mission to confirm landing/disarm before restarting the simulator project.

```sh
ln -sfn "$(basename "$MARKERFREE_HOVER_DIR")" artifacts/markerfree/current
docker --context colima-dronehacks-sim exec -w /workspace "$MARKERFREE_CONTAINER" \
  python3 -m mission.markerfree_evaluate \
  --output "$MARKERFREE_HOVER_DIR" -- \
  --task hover --launch-position -2.5146 0.9144 0.24 --launch-yaw-deg 90 \
  --use-px4-tilt --survey-normals fals7
```

Score the hover before proceeding:

```sh
docker --context colima-dronehacks-sim exec -w /workspace "$MARKERFREE_CONTAINER" \
  python3 -m tools.score_markerfree "$MARKERFREE_HOVER_DIR" \
  --output "$MARKERFREE_HOVER_DIR/score.json"
```

Require the hover summary to report completion, landing/disarm and at least three settled seconds. Its score must have the same run ID and pass all seven applicable checks: controller waypoint completion, complete truth coverage, ordered truth waypoint arrival, multipart clearance, transit-floor checking, PX4 ground/disarm confirmation and touchdown inside the pad. The window-crossing check is expected to be false for a hover. A scorer process exit alone is not a hover acceptance decision; inspect those fields and retain any failed or incomplete recording before continuing.

Recreate only this named project between the verified hover and route using the same image and noise setting. This clears the simulator state while preserving the host-mounted evidence and policy exchange:

```sh
SIM_PX4_IMAGE=dronehacks-markerfree:hrt-v1.16.2 \
  SIM_RGBD_DEPTH_NOISE=gaussian-10mm ./runtime/run-markerfree.sh
```

Then start the learned policy route:

```sh
ln -sfn "$(basename "$MARKERFREE_POLICY_DIR")" artifacts/markerfree/current
docker --context colima-dronehacks-sim exec -w /workspace "$MARKERFREE_CONTAINER" \
  python3 -m mission.markerfree_evaluate \
  --output "$MARKERFREE_POLICY_DIR" -- \
  --task policy --course mission/course-window-x500.json \
  --policy-dir artifacts/markerfree/policy-exchange \
  --policy-sha256 "$MARKERFREE_CHECKPOINT_SHA" --policy-depth-mode raw \
  --heading-lookahead 1.8 \
  --hover-summary "$MARKERFREE_HOVER_DIR/telemetry.summary.json" \
  --launch-position -2.5146 0.9144 0.24 --launch-yaw-deg 90 \
  --use-px4-tilt --survey-normals fals7
```

The current gates default to a 120 ms maximum accepted capture gap and 300 ms maximum capture age. Verify `vision_max_capture_gap_s: 0.12` and `vision_max_capture_age_s: 0.3` in the resulting telemetry metadata and summary before comparing against the recorded FALS7 runs.

The waypoint baseline does not need the host policy service. Replace `--task policy` with `--task course` and omit the policy-specific options to run that controller with the same hover summary and FALS7 setting.

Score the independent truth/contact observations after the route:

```sh
docker --context colima-dronehacks-sim exec -w /workspace "$MARKERFREE_CONTAINER" \
  python3 -m tools.score_markerfree "$MARKERFREE_POLICY_DIR" \
  --output "$MARKERFREE_POLICY_DIR/score.json"
```

The truth observer supplies offline scoring only. A generated score and run directory are new evidence, not files included in a fresh clone. Contact silence is not proof of no collision; use the scorer's geometry and contact coverage fields as documented.

The `current` link above lets the viewer follow each flight as it runs.

## Stop only this simulator project

This quickstart avoids the older transition scripts that stop `sim` in the `dronehacks-sim` or `dronehacks-px4` Compose project. To stop this marker-free instance only:

```sh
docker --context colima-dronehacks-sim compose \
  -p dronehacks-markerfree \
  -f compose.px4.yaml -f compose.markerfree.yaml stop sim
```

The simulator and learned policy are software tests using Gazebo rendering, a synthetic depth sensor, and a modeled x500 airframe. They do not establish physical camera or aircraft transfer.
