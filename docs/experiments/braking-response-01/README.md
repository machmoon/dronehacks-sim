# Braking response trace diagnostic

This reproducible, standard-library-only analysis compares recorded policy-run command and PX4 velocity traces from runs 05, 07, 08, and 09. Run it from the repository root with:

```sh
python3 docs/experiments/braking-response-01/reproduce.py
```

It emits `summary.json` with the analysis-script SHA256, telemetry input hashes, and recorded/current source hashes. The script also runs deterministic hand checks for a unit command step at 0.5 s: with a 1 s response constant, the predicted velocity at 1 s is `1 - exp(-0.5)` at zero delay and `1 - exp(-0.3)` with 0.2 s delay. The later command in the fixture is ignored.

The telemetry samples commands at roughly 10 Hz. The already-sent ENU command is timestamped by telemetry `sim_time`; measured PX4 ENU velocity uses `position_stamp_sim_s`. The fit uses exact first-order integration across the delayed zero-order-held command changes, initializes each contiguous selected route segment from its first measured velocity, and resets at excluded phases, missing history, or time gaps. It does not interpolate across gaps or use commands that become effective after a measured sample.

The coarse-grid fits were about 0.44–0.48 s response and 0.12–0.16 s delay, with 0.0155–0.0156 m/s fit RMSE. Each fit is measured on the same trace used to choose its parameters, without a held-out test. The configured 0.8 s / 0.4 s pair had 0.084–0.086 m/s RMSE in this descriptive model. These values describe the recorded closed-loop traces; they do not identify isolated airframe dynamics or worst-case stopping distance. The command-to-estimate path includes policy feedback, PX4 velocity control, estimator behavior, and setpoint slew, and the command sample is only about 10 Hz.

The guard's 0.4 s lead combines assumed command delay, estimate age, and maximum setpoint interval. It is not a measured pure command delay. These trace fits do not justify reducing the guard or establish a safety bound.

The input archives are local files under `artifacts/markerfree/` and are not shipped in the public repository. Their hashes identify the data used here; reproducing the numerical results requires those archives.
