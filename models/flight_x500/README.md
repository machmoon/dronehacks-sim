# PX4 x500 flight-evidence variant

This variant is based on PX4-Autopilot commit
`54f0455ffcd755534539a7cf33a09a20bf71d29d` (tag `v1.16.2`), as built by
this repository. The source files were copied from `/opt/px4/models/x500` and
`/opt/px4/models/x500_base` in the pinned simulator image.

The only model-physics-file changes are:

- `model://x500_base` resolves to `model://flight_x500_base`.
- Read-only Gazebo contact sensors are added to the existing `base_link` and
  rotor links in `../flight_x500_base/model.sdf`.

The stock collision geometries, poses, inertias, masses, joints, and motor
plugins are preserved. Sensor topics group the existing collision elements by
link at `/model/course_quad/contact/<link>`. `worlds/course.sdf` already loads
Gazebo's contact system. Mesh references remain `model://x500_base/...`, so no
mesh copies or edits are needed.

Stock source SHA-256 values before the variant edits:

- `x500/model.sdf`: `76d8668c9c1a10348f9b7011d892203f319e5051dc8c0eb06c62e52db8e6714e`
- `x500/model.config`: `52fdda8e66faa5095434d54e4dcaad5a6a8833cd014e0a5b439349ed8b4a1b67`
- `x500_base/model.sdf`: `d97e53de8c50f167b0b3a4f86dc21413236387f7b72cb817549362b63534e83c`
- `x500_base/model.config`: `4856afbdeb52f19468f56ddaf4be57c04688f347b8cef2f609daacd2d014d437`

The stock SDF checksums are listed above. The variant is selected by a world
include of `model://flight_x500`; stock
`model://x500` remains available unchanged. The copied PX4 BSD 3-Clause license
is in `LICENSE`.
