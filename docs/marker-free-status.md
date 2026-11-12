# Marker-free PX4 status

## Latest verified run

Course-08 completed the **waypoint baseline** through all 15 waypoints. It armed and then landed/disarmed after 83.000 simulated seconds. Independent Gazebo truth scoring verified a forward full-envelope window crossing with 0.106299 m minimum aperture clearance after the 15 mm model inflation, plus 0.313969 m horizontal envelope clearance inside the modeled landing pad. The multipart CAD-proxy path check was clear over takeoff, transit, and landing.

The estimator/controller used forward 384×256 rendered RGB-D registration against the supplied static course survey, downward ToF, and PX4 EKF2. The navigation audit passed: no artificial landmark models and zero publishers on the Gazebo truth odometry topic used by the separate observer. Recorded simulator truth did not feed navigation or control. The course-08 contact report has partial topic coverage: base-link contact messages were observed, while rotor topics advertised publishers but delivered no messages. That silence does not prove absence of collisions.

Policy-01 failed during preflight with `invalid_policy_exchange_record:response.json`; elapsed flight time was zero and the vehicle did not arm. No learned-policy flight occurred, so that result remains unverified. Do not interpret course-08 as a learned-policy result. Earlier marker-free course runs failed observability and safely landed after abort; their failures remain part of the evidence. The course-08 replay and compact score are linked from the [README](../README.md#dronehacks-sim).

## Limits

The route, window, and pad clearances are simulated checks against the pinned x500 multipart collision model and the supplied CAD proxy. They use linear translation, shortest-arc attitude interpolation, and 15 mm inflation. The model envelope has not been verified against the physical Skybeta airframe and propellers. Simulated depth, supplied survey geometry, and the PX4 roll/pitch prior do not validate a real OAK camera estimator or real flight. The course score is not official competition scoring.

See the [RGB-D runbook and historical evidence](markerfree-rgbd.md), [compact machine-readable results](markerfree-results.json), and [OAK metric-depth handoff](oak-depth-handoff.md).
