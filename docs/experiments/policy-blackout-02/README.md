# Learned-policy camera-aid loss

This preregistered diagnostic cut camera aiding 18.008 simulated seconds after arming. The packaged v1 policy had accepted 25 actions. Its freshness watchdog stopped the mission 356 ms after injection with the expected `stale_policy_action` fault. PX4 entered Auto Land, touched the floor, and disarmed. The closed ULog contains no visual-odometry update after the cutoff and spans the abort without reported dropout or corruption.

The diagnostic **did not pass**: eight of nine checks passed, but the independent truth recording has a 204 ms gap from 50.384 to 50.588 simulated seconds. That exceeds the unchanged 120 ms interpolation limit inside the abort interval, which ends at the controller's disarm confirmation at 50.612 s. PX4 had disarmed at 50.032 s. The observed trajectory segments had no unresolved obstacle separation, but missing samples prevent certification of the complete abort path. Received contact messages contain no obstacle or transit-floor contact; their partial topic coverage cannot establish zero physical contacts.

The recorder used a four-message truth queue without overflow counters. That is vulnerable to writer stalls, but the saved data cannot establish whether queue eviction, transport delivery, or another pause caused this gap. Any instrumentation change and new trial must retain this result and the original scoring limits.

| Event | Simulation time |
| --- | ---: |
| Armed in Offboard | 18.504 s |
| Camera aiding suppressed | 36.512 s |
| Expected policy freshness fault | 36.868 s |
| PX4 Auto Land | 36.892 s |
| First recorded floor contact | 38.348 s |
| PX4 disarmed | 50.032 s |
| Controller confirmed touchdown and disarm | 50.612 s |

The [registration](protocol.json) predates the run. The [result](results.json) includes all nine checks, registration compliance, exact runtime/checkpoint/source hashes, and the recording gap. The original local archives are under `artifacts/markerfree/blackout-02`; no physical vehicle was involved. Sensors and recorders continued running during injection: this tests suppression of camera aiding, not a stopped render or disconnected camera.
