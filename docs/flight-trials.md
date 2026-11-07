# Marker-free flight trials

This inventory contains 39 unique run IDs across 40 flight telemetry archive paths; `current` is an identical alias of `survey-startup-treatment-01`. No overall reliability percentage is inferred across changing code, configuration, and diagnostic protocols.

## Full-route attempts

| Archive | Controller / checkpoint | Outcome / fault | Controller waypoints | Independent truth | Landing |
|---|---|---|---:|---:|---|
| clean-export-policy-01 | learned policy / `a1919ecd` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| course-01 | waypoint course controller / `—` | odometry_jump | 1/15 | not scored | not confirmed in archived telemetry/summary |
| course-02 | waypoint course controller / `—` | rgbd_pose_unavailable:tracking_lost:survey:unobservable_depth_geometry:0.0035633;depth:unobservable_depth_geometry:3.4445e-11;rgb:insufficient_correspondences | 1/15 | 1/15 | confirmed touchdown and disarm |
| course-03 | waypoint course controller / `—` | rgbd_pose_unavailable:tracking_lost:survey:unobservable_depth_geometry:0.0039908;depth:unobservable_depth_geometry:5.5805e-11;rgb:insufficient_correspondences | 1/15 | 1/15 | confirmed touchdown and disarm |
| course-04 | waypoint course controller / `—` | rgbd_pose_unavailable:tracking_lost:survey:unobservable_depth_geometry:0.0026728;depth:unobservable_depth_geometry:0.00073811;rgb:insufficient_texture | 12/15 | 10/15 | confirmed touchdown and disarm |
| course-05 | waypoint course controller / `—` | px4_power_health:battery_status_timestamp_not_advancing | 1/15 | 1/15 | confirmed touchdown and disarm |
| course-06 | waypoint course controller / `—` | px4_power_health:battery_status_listener_timeout | 9/15 | 9/15 | confirmed touchdown and disarm |
| course-07 | waypoint course controller / `—` | rgbd_pose_unavailable:tracking_lost:survey:unobservable_depth_geometry:0.0028614;depth:unobservable_depth_geometry:0.0016497;rgb:insufficient_texture | 1/15 | 1/15 | confirmed touchdown and disarm |
| course-08 | waypoint course controller / `—` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| course-09 | waypoint course controller / `—` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| course-10 | waypoint course controller / `—` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| policy-01 | learned policy / `872ba44c` | invalid_policy_exchange_record:response.json | 0/15 | not scored | no armed flight; touchdown not applicable |
| policy-02 | learned policy / `872ba44c` | course_boundary_breach | 6/15 | 6/15 | confirmed touchdown and disarm |
| policy-03 | learned policy / `3602a898` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| policy-04 | learned policy / `3602a898` | px4_power_health:battery_status_stale_sim_timestamp | 0/15 | 0/15 | confirmed touchdown and disarm |
| policy-05 | learned policy / `a1919ecd` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| policy-06 | learned policy / `a1919ecd` | rgbd_pose_unavailable:tracking_lost:survey:unobservable_depth_geometry:0.0028333;depth:insufficient_depth_overlap;rgb:insufficient_correspondences | 5/15 | 5/15 | confirmed touchdown and disarm |
| policy-07 | learned policy / `a1919ecd` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| policy-08 | learned policy / `a1919ecd` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| policy-09 | learned policy / `a1919ecd` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| policy-10 | learned policy / `a1919ecd` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |
| policy-11 | learned policy / `a1919ecd` | completed | 15/15 | 15/15 ordered | confirmed touchdown and disarm |

## Independent score details

Only a score file whose run ID matches its telemetry archive is joined. Abort-landing clearance and contact coverage are reported separately from route completion.

- **clean-export-policy-01:** truth 15/15 ordered=True.
- **course-02:** truth 1/15 ordered=False.
- **course-03:** truth 1/15 ordered=False.
- **course-04:** truth 10/15 ordered=False.
- **course-05:** truth 1/15 ordered=False.
- **course-06:** truth 9/15 ordered=False.
- **course-07:** truth 1/15 ordered=False.
- **course-08:** truth 15/15 ordered=True.
- **course-09:** truth 15/15 ordered=True.
- **course-10:** truth 15/15 ordered=True.
- **policy-02:** truth 6/15 ordered=False.
- **policy-03:** truth 15/15 ordered=True.
- **policy-04:** truth 0/15 ordered=False; abort landing 49.192→59.46 s, CAD clear through disarm=True, contact coverage=partial_topic_coverage.
- **policy-05:** truth 15/15 ordered=True.
- **policy-06:** truth 5/15 ordered=False; abort landing 50.896→56.556 s, CAD clear through disarm=True, contact coverage=partial_topic_coverage.
- **policy-07:** truth 15/15 ordered=True.
- **policy-08:** truth 15/15 ordered=True.
- **policy-09:** truth 15/15 ordered=True.
- **policy-10:** truth 15/15 ordered=True.
- **policy-11:** truth 15/15 ordered=True.

## Diagnostic hovers

| Archive | Outcome | Fault | Landing | Independent score |
|---|---|---|---|---|
| clean-export-hover-01 | completed | none | confirmed touchdown and disarm | matched |
| hover-01 | completed | none | confirmed touchdown and disarm | not scored |
| hover-02 | completed | none | confirmed touchdown and disarm | matched |
| hover-03 | completed | none | confirmed touchdown and disarm | matched |
| hover-04 | failed | rgbd_pose_unavailable:tracking_lost:survey:insufficient_depth_overlap;depth:insufficient_depth_overlap;rgb:insufficient_inliers | confirmed touchdown and disarm | matched |
| hover-05 | completed | none | confirmed touchdown and disarm | matched |
| hover-06 | completed | none | confirmed touchdown and disarm | matched |
| hover-07 | completed | none | confirmed touchdown and disarm | matched |
| hover-08 | completed | none | confirmed touchdown and disarm | matched |
| hover-09 | completed | none | confirmed touchdown and disarm | matched |
| hover-10 | completed | none | confirmed touchdown and disarm | matched |
| hover-11 | failed | rgbd_pose_unavailable:tracking_lost:survey:unobservable_depth_geometry:-4.2137e-18;depth:insufficient_depth_overlap;rgb:insufficient_inliers | confirmed touchdown and disarm | matched |
| survey-startup-control-01 | completed | none | confirmed touchdown and disarm | matched |
| survey-startup-treatment-01 | failed | absolute_survey_startup_unqualified | no armed flight; touchdown not applicable | not scored |

## Blackout diagnostic

| Archive | Outcome | Injection/fault | Controller waypoints | Independent truth | Landing |
|---|---|---|---:|---:|---|
| blackout-01 | failed | rgbd_pose_unavailable:stale_capture | 1/15 | 1/15 | confirmed touchdown and disarm |
| blackout-02 | failed | stale_policy_action | 1/15 | 1/15 | confirmed touchdown and disarm |
| blackout-03 | failed | stale_policy_action | 1/15 | 1/15 | confirmed touchdown and disarm |

## Interpretation and provenance

The JSON preserves each run ID, checkpoint and braking-guard metadata, event and summary simulation times, matching independent-score checks, and SHA-256 fingerprints of source records. Missing scores remain unknown; the inventory does not infer truth waypoints from controller progress.

All results are simulator evidence. Contact-topic coverage may be partial; absence of unexpected observed contacts is not a zero-physical-collision claim.
