# PX4 POSIX HRT backport candidate

Policy-04 stopped when the battery and system-power publications became stale. The pinned SITL binary stopped running those publishers and the CPU-load publisher around arming. Read-only probes still saw the simulated HRT interrupt count, IMU timestamps, and event-driven PX4 work advance. Gazebo and PX4 timestamps remained monotonic around the fault. The local evidence is in the policy-04 `power-feed-diagnostic.json` and `hrt-scheduler-diagnostic.json` artifacts.

The pinned POSIX implementation releases the HRT queue lock while invoking a callback. Another thread can mutate the list in that interval, before a periodic callback is reinserted. The upstream change describes queue chains becoming torn and leaving entries unreachable under lockstep. That mechanism is consistent with the observed selective stall. The available traces do not expose the list itself, so an A/B test is still needed. [Upstream PX4 fix](https://github.com/PX4/PX4-Autopilot/commit/9b62a1701de211fe8d9021f291d5668e8fb38b4e) · [Pinned POSIX implementation](https://github.com/PX4/PX4-Autopilot/blob/54f0455ffcd755534539a7cf33a09a20bf71d29d/platforms/posix/src/px4/common/drv_hrt.cpp)

The patch adapts only the POSIX HRT part of upstream `9b62a1701de211fe8d9021f291d5668e8fb38b4e` to PX4 v1.16.2 at `54f0455ffcd755534539a7cf33a09a20bf71d29d`. It uses a POSIX recursive pthread mutex so a callback on the HRT worker can reenter the HRT scheduling API while concurrent callers wait. It leaves the separate WorkQueue lifetime change out of this experiment. The original PX4 BSD-3-Clause source header and license text are preserved. [Distributed PX4 license](../patches/PX4-LICENSE) · [Pinned upstream license](https://github.com/PX4/PX4-Autopilot/blob/54f0455ffcd755534539a7cf33a09a20bf71d29d/LICENSE)

## Verify and build in an isolated PX4 checkout

From the dronehacks worktree, set `PX4_SRC` to the isolated checkout. The apply check and SITL compilation have passed against the exact pinned upstream source. The derived image passed binary provenance and shared-library checks. Scheduler behavior requires the runtime checks below.

```sh
test "$(git -C "$PX4_SRC" rev-parse HEAD)" = 54f0455ffcd755534539a7cf33a09a20bf71d29d
git -C "$PX4_SRC" apply --check --whitespace=error-all "$PWD/patches/px4-posix-hrt-serialization.patch"
git -C "$PX4_SRC" apply "$PWD/patches/px4-posix-hrt-serialization.patch"
git -C "$PX4_SRC" diff --check
git -C "$PX4_SRC" diff --stat
CMAKE_BUILD_PARALLEL_LEVEL=1 make -C "$PX4_SRC" px4_sitl
```

The installed PX4 image contains the stripped binary and no build tree. Docker had no cached build objects, so the first build compiles C++ from scratch. The isolated build reuses the runtime image, preserves the pinned build and install scripts, and stages the sparse checkout and object files in a bounded tmpfs. The package estimate on that image was 76.9 MB of archives and 316 MB installed. Check host disk and VM memory before reproducing the build.

```sh
docker --context colima-dronehacks-sim run -d --name dronehacks-markerfree-hrt-builder \
  --memory 5g --memory-swap 5g --cpus 2 --pids-limit 512 \
  --tmpfs /tmp:rw,exec,nosuid,nodev,size=3g,mode=1777 \
  --tmpfs /var/lib/apt/lists:rw,nosuid,nodev,size=128m \
  --tmpfs /var/cache/apt/archives:rw,nosuid,nodev,size=256m \
  --mount type=bind,source="$PWD",target=/workspace,readonly \
  --entrypoint /bin/sh dronehacks-markerfree:base -c 'sleep infinity'
docker --context colima-dronehacks-sim exec dronehacks-markerfree-hrt-builder \
  bash /workspace/px4/build-hrt-backport.sh dependencies
docker --context colima-dronehacks-sim exec dronehacks-markerfree-hrt-builder \
  bash /workspace/px4/build-hrt-backport.sh compile

mkdir -p artifacts/px4/hrt-backport-build
bash px4/package-hrt-backport.sh artifacts/px4/hrt-backport-build/image
```

The manifest records the base revision, runtime base image ID, patch and script hashes, build adjustments, and hash of the installed stripped binary. Packaging verifies that the image layers start with those of the builder's runtime base and that the embedded binary hash matches the manifest. Verify the binary hash again before a run.

On a fresh, disarmed test instance, `px4-hrt_test start` exercises recursive locking by having its callback schedule the same HRT entry again. Verify six `Test` messages and the final `HRT_CALL - 1` and `HRT_CALL + 1` lines. Run it before the timed soak so its callbacks stay outside the publisher cadence measurement. [Pinned PX4 HRT test](https://github.com/PX4/PX4-Autopilot/blob/54f0455ffcd755534539a7cf33a09a20bf71d29d/src/systemcmds/tests/hrt_test/hrt_test.cpp)

Compare the existing image with the patched image using the same simulator seed and settings. Capture periodic publisher timestamps and HRT and publisher counters across the unarmed soak and arming transition. Keep the battery and power watchdog checks active for the guarded flight. A successful run shows those publications kept advancing in that run. Repetition is needed to assess the intermittent failure.

## First runtime checks

The patched image passed the six-callback recursive-timer smoke and a 60.256-second simulated publisher soak with 186 samples and no stale battery, system-power or CPU-load publication. The timer test overlapped the beginning of that soak. A subsequent camera-guided hover landed and disarmed after 19.000 simulated seconds; independent scoring certified its modeled multipart path and touchdown inside the pad. These finite checks do not establish that the intermittent scheduler fault is eliminated. The [runtime evidence](px4-hrt-results.json) preserves the binary and artifact hashes.

Start this explicitly named image; a missing requested image now causes the launcher to fail instead of retagging the base image:

```sh
SIM_PX4_IMAGE=dronehacks-markerfree:hrt-v1.16.2 \
  SIM_RGBD_DEPTH_NOISE=control ./runtime/run-markerfree.sh
docker --context colima-dronehacks-sim exec dronehacks-markerfree-sim-1 \
  python3 -m tools.soak_px4_publishers \
  --output /workspace/artifacts/markerfree/hrt-soak-NEW/telemetry.jsonl \
  --sim-seconds 60 --wall-seconds 400
```

The soak is passive and refuses to run outside the isolated simulator partition. It does not arm the drone, change parameters or replace the flight watchdog. Each new flight records the actual executable hash and checks it against the image's build manifest before arming.

## Post-flight publisher soak

After policy-09 landed and disarmed, the same patched PX4 process passed a separate 600.108-second simulated publisher soak. All 1,869 samples passed the timestamp checks for battery, system power and CPU load. The observation took 2,001.591 wall-clock seconds at the configured slow simulator rate. No arming or parameter changes occurred during this soak. This extends the finite evidence for publisher continuity after a completed noisy-depth flight; it does not test every arming transition or establish that the intermittent fault is eliminated. The [runtime evidence](px4-hrt-results.json) includes the executable identity, sample-log hash and recomputed assessment.
