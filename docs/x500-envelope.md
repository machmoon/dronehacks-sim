# PX4 x500 collision envelope audit

This is a geometry audit of the simulator's PX4 x500 model, not of the physical Skybeta aircraft. It derives dimensions from the SDF collision solids and joint poses. The rotor sweep below is the sweep of the modeled collision rectangles, not a measurement of the propeller visual meshes.

## Source and provenance

The flight-evidence variant is copied from PX4-Autopilot v1.16.2, commit [`54f0455ffcd755534539a7cf33a09a20bf71d29d`](https://github.com/PX4/PX4-Autopilot/tree/54f0455ffcd755534539a7cf33a09a20bf71d29d), as built into the pinned simulator image. The variant's [README](../models/flight_x500/README.md) records the original stock SDF hashes and states that the variant changes only the model include and adds read-only contact sensors. The copied license is [BSD 3-Clause](../models/flight_x500/LICENSE), copyright Rudis Laboratories 2022.

The audited files are [`flight_x500/model.sdf`](../models/flight_x500/model.sdf) and [`flight_x500_base/model.sdf`](../models/flight_x500_base/model.sdf). The base model's stock collisions are a body box, two tilted leg boxes, two skid boxes, and one thin rectangular collision per rotor. Contact sensors added to those collision elements do not change their geometry. The rotor visuals reference `model://x500_base/meshes/1345_prop_{cw,ccw}.stl`; those mesh files are not copied into this repository, and their render bounds are not used in the calculations below.

## Dimensions relative to `base_link`

The following bounds are in model coordinates with `base_link` at the model origin. The model itself is spawned at z = +0.24 m; add that offset for world-z coordinates at level attitude.

| Component | SDF definition | Derived bounds |
|---|---|---|
| Main body collision | `base_link_collision_0`, pose `(0,0,+.007)`, box `.353553 × .353553 × .05` m | x/y ±0.17678 m; z −0.018 to +0.032 m |
| Legs | collisions 1/2, centers `(0,±.098,−.123)` m, roll ∓/+0.35 rad, box `.015 × .015 × .21` m | x ±0.0075 m; outer y ±0.14105 m; z −0.22421 to −0.02179 m |
| Skids | collisions 3/4, centers `(0,±.132,−.2195)` m, box `.25 × .015 × .015` m | x ±0.125 m; outer y ±0.1395 m; z −0.227 to −0.212 m |
| Rotor collision centers | four rotor links at `(±.174,±.174,+.06)` m | center radial distance from base_link: `sqrt(.174²+.174²) = 0.24607 m` |
| Each rotor collision blade | box `.27923077 × .01692308 × .00084615` m, centered on link and rotated about its z-axis | swept-disc radius `sqrt((.27923077/2)²+(.01692308/2)²) = 0.13987 m`; z = +0.05958 to +0.06042 m |

The rotor link joints are revolute about z. The farthest swept rotor corner from the body origin is therefore approximately `0.24607 + 0.13987 = 0.38594 m`. Its axis-aligned x/y reach is `±(0.174 + 0.13987) = ±0.31387 m`. These are different measures: using `sqrt(2) × 0.174 + rotor_radius` gives the radial reach to a diagonal rotor tip; the axis-aligned bounds are smaller.

For each rotor rectangle, the exact radial sweep uses the rectangle-corner hypot above. The simpler half-length-plus-half-width sum (0.14808 m) is a conservative upper bound on its corner radius, not the exact radius.

## What the 0.36 m sphere means

A sphere centered at `base_link` with radius 0.36 m overstates vertical body/gear extent: at level attitude the modeled collision solids reach z −0.227 m down and +0.06042 m up, while the sphere extends 0.36 m in either direction. But it is not a fully conservative circumscribed envelope for the modeled rotor sweep: the diagonal rotor-tip radius is about 0.38594 m, roughly 2.6 cm beyond the sphere. A sphere that encloses the entire modeled rotor sweep would need radius at least 0.386 m, which would further exaggerate vertical extent.

This supports using an attitude-aware union of body/leg/skid solids and four rotor-centered swept discs for geometric passage analysis, with an explicit clearance margin. It does not justify shrinking the scorer's guard without a reviewed patch and its own tests. The current 0.36 m radius is a spherical proxy for the PX4 x500 only; neither this model nor the available source defines the real Skybeta's measured body, guards, propellers, camera, or landing gear. The forward camera is mounted at +0.34 m in the generated scene but has no collision shape in the audited SDF, so include its physical housing separately once dimensions are known. No claim here establishes physical-aircraft clearance.
