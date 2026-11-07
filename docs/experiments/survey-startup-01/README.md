# Optional survey startup gate

This test evaluates the opt-in `--require-survey-startup` prearm gate from source revision `f2cbe06`. The option remains disabled by default. It requires at least 3 seconds of consecutive accepted absolute-survey captures, no capture gap over 120 ms, and a current final capture.

- **Control hover** `8a24b47cb4424831add1d987dbc5ccd1`: passed all 14 registered checks. PX4 armed with the gate ready after 315 accepted captures spanning 15.7 simulated seconds; the hover settled for 3 seconds and landed/disarmed.
- **Net-visual-removed treatment** `b5bd32b04123445f9274cea6c2cb5017`: passed all 12 negative-control checks by refusing to arm with `absolute_survey_startup_unqualified`. Its 702-record capture ledger contained 701 `rgbd_odometry` and one `stationary_launch` sample, with no qualified captures, no arm request, and no armed PX4 samples.

This verifies one control hover and one prearm refusal. It does not demonstrate treatment navigation, full-route performance, or continued observability after takeoff. The treatment keeps the estimator's analytic net map while removing rendered net visuals; contact-topic coverage was partial. No physical aircraft or sensor was used.

See the [machine-readable results](results.json) and [registered protocol](protocol.json) for checks and hashes.
