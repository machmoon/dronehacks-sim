# Stereo edge visibility and finite mismatch replay

The [preregistered protocol](protocol.json) fixed eight input views, five perturbation modes and three eastward prior offsets before fitting. All 120 trials ran in the dedicated runtime container with OpenCV 4.6 and NumPy 1.26.4. No pose or actor request went to Gazebo or PX4. The forward depth transform assumes a 75 mm stereo baseline on the simulated camera grid. The baseline and 1/8-pixel disparity step are diagnostic inputs. The active OAK calibration, confidence handling, matcher and stereo transport were outside this test. [Luxonis disparity and depth configuration](https://docs.luxonis.com/hardware/platform/depth/configuring-stereo-depth)

The native MAZE input is the float32 policy-06 rejection at 50.852 s with its recorded sensor prior. The existing synthetic 69° alias image uses the same native sensor position and tilt with a fixed synthetic yaw. Its saved reference position is read only for scoring. The policy-07 window inputs are six 5 Hz depth PNGs at frames 359, 371, 383, 395, 407 and 419. These represent two images each in approach, align and crossing. Each uses an independently archived preceding native sensor EV pose, ToF and a PX4 tilt sample at the capture time. The PX4 vision timestamps required a documented 295–423 µs correction to the 4 ms Gazebo acquisition grid. All eight cases use a 52 ms interval. Archive image processing age is set to zero for this offline fit. The prepared inputs and hashes are in [sources.json](sources.json).

Each trial reports raw survey registration separately from acceptance after the current timing, motion and absolute uncertainty gates. It does not replay relative depth or RGB fallback because the native previous images are unavailable. It does not simulate EKF response, policy actions or future captures. Survey pose error uses independent observer truth after all fits. Policy admission is the raw preprocessing result and is measured once per modified image, independently of the three survey seeds. The captured RGB JPEGs are hashed provenance for the archive frames. They do not enter the survey fit.

| Perturbation | Survey fits passing runtime gates | Worst accepted error | Policy inputs admitted |
|---|---:|---:|---:|
| Control | 24/24 | 7.52 mm | 8/8 |
| Disparity quantized | 24/24 | 7.26 mm | 8/8 |
| Geometric occlusion holes | 18/24 | 7.90 mm | 0/8 |
| Occlusion with 2 px horizontal dilation | 18/24 | 7.89 mm | 0/8 |
| Finite far-edge mismatch | 24/24 | 33.17 mm | 8/8 |

Both hole conditions removed the independent support from the native and synthetic maze views. The survey rejected each of those cases at all three offsets, while the other six inputs remained observable. The raw policy rejected every image with a preserved hole. The separate negative control deliberately gave those holes the simulator `+Inf` meaning. The actor then reported full coverage and mapped up to 105 previously near rays to maximum range. That invalid conversion was evaluated only in preprocessing.

No accepted survey pose exceeded the preregistered 60 mm counterexample threshold. Six poses exceeded 30 mm, all three offsets for the same two corrupted window views. The zero-offset cases at frame 395 in ALIGN and frame 407 in CROSS shifted about 33 mm north despite plane RMS of 11.1 and 9.7 mm respectively. The finite mismatch in ALIGN frame 383 also moved 14 of 384 actor bins from at most 1.5 m to at least 3 m. These were outside the central region and the policy reported full coverage. The finite substitutions model a deliberately adverse edge correspondence and do not establish its probability in a physical scene. The [complete result](results.json) retains rejects, fits, per-frame policy observations and timing.

## Reproduction

Exact preparation requires the original local policy-06/policy-07 camera, telemetry and ULog archives, which are not checked into Git. This folder preserves the protocol, scripts, prepared inputs where available, complete results and input hashes. Restore compressed raw observer files only when a tool requires the original path. The local archive manifests contain byte-verified restoration commands.

Use the existing archives and a host Python with NumPy, OpenCV and PyULog. The second command runs a read-only offline input replay in the dedicated container and writes only its result JSON.

```sh
original=docs/experiments/stereo-stress-01
replay=$(mktemp -d artifacts/markerfree/stereo-stress-replay.XXXXXX)
cp "$original"/protocol.json "$original"/prepare.py "$original"/probe.py "$replay/"
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "$replay/prepare.py"
docker --context colima-dronehacks-sim exec \
  -e PYTHONPATH=/workspace -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 \
  dronehacks-markerfree-sim-1 /usr/bin/python3 "/workspace/$replay/probe.py"
```
