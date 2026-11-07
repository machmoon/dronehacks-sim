# +40 mm launch-prior diagnostic

This is one preregistered simulation pair: a hover prerequisite followed by one learned-policy route. The protocol and machine-readable evidence are in [protocol.json](protocol.json) and [results.json](results.json). The [94.88-second RGB/depth replay](../../media/window-policy-launch-prior.mp4) includes captures before arming and after landing; the measured flight took 78 simulated seconds.

## Outcome

| Run | Configuration | Result |
| --- | --- | --- |
| Hover-09 prerequisite | +40 mm declared launch prior; FALS7 normals; Gaussian 10 mm depth noise | 23/23 protocol checks passed. Hover settled at 3.004 simulated seconds. Maximum position errors over the full HOVER_CHECK phase were 4.06 mm PX4 and 4.39 mm camera. |
| Policy-10 | Same launch prior and FALS7/noise configuration; raw policy depth | Completed all 15 waypoints and landed/disarmed. The scorer passed 8/8 checks. Armed duration was 78.0 simulated seconds, including 73.096 seconds from arm to route completion. Minimum full-envelope window clearance was 94.75 mm; 369 policy actions were received. |

The closest registered FALS7 Gaussian-noise policy reference, policy-09, also took 78.0 simulated seconds and had 95.34 mm (95.3 mm rounded) window clearance, with 363 actions. These are individual runs; they do not establish a speed, reliability, or generalization difference. The hover-08 nominal-launch reference had a 3.036-second settle time and full-HOVER_CHECK maximum errors of 4.08 mm PX4 and 5.53 mm camera. Hover errors are phase-wide maxima, not post-settle-only figures.

## What changed

The declared launch prior moved +40 mm in ENU X from the simulated spawn. The CLI launch value seeds the vision estimator and is also used for startup position qualification and the exact-match hover-proof contract, so this is a combined estimator-seed and declared-launch perturbation. Independent truth at policy-10 arming remained at the original simulated spawn XY, and the first course waypoint retained that same XY. The simulated vehicle spawn and course geometry were not moved.

Before either run, observer buffering/accounting was amended after a prior blackout diagnostic exposed an unscorable truth gap. The updated observer was used for hover-09 and policy-10; policy-09 used the earlier observer. The amendment changed evidence recording only; controller, estimator, policy checkpoint, geometry, and score thresholds were unchanged. The registered source hashes and run input hashes are recorded in `results.json`.

## Evidence limits

Policy-10 observer recording completed with zero queue overflow or pending items and consistent receive/dequeue accounting. Contact evidence is scoped to messages received: only the base-link contact topic delivered messages. The four rotor topics were advertised but silent. Therefore, zero unexpected contact pairs in the received messages is not proof that no physical contact occurred.

This is a single paired simulation diagnostic. It is not a reliability estimate, generalization test, safety guarantee, or physical-flight validation. The configured predictive boundary brake also carries no formal safety guarantee.
