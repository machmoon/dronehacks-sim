# Clean tracked-export validation

This run used a `git archive` export of revision `37696f60537d8a69ea029e603d3821c01f7e045c`. Its 36 runtime Python files matched the registered source inventory, and the exported tree contained no untracked helpers. The simulator launcher, Compose configuration, default world, and the three runtime model trees also matched the reused checkout byte for byte.

This validates execution from tracked exported source against the existing Colima simulator image and host Python environment. It does not demonstrate a clean-machine install, a new container-image build, or independent image reproducibility. The exported checkout supplied the mission, observer, scorer, and host policy inference code; the existing simulator launcher and image were reused.

## Results

- **Passive collector smoke:** the exported collector recorded 32 synchronized RGB-D frames on the treatment world. The registered sensor-world, PX4 binary, source, native array, observer duration, closure, and overflow checks passed. This is an integration smoke test, not a second A/B experiment.
- **Hover:** exported-source hover `fc60b9d437bd4d2b8b8a5597979e058e` passed its check, settled for 3.004 simulated seconds, and had maximum PX4 and camera position errors of 4.822 mm and 4.441 mm.
- **Policy route:** exported-source policy run `26f5ee4841b84e3db77ad1f9b4a07917` passed all 19 registered checks. It completed 15/15 waypoints in 78 simulated seconds, landed and disarmed, and had 98.152 mm minimum modeled window clearance. The separate inference service used the packaged v1 checkpoint and its session matched the run telemetry.

Watch the [exported-policy RGB-D replay](../../media/window-policy-clean-export.mp4). The encoded timeline is 94.48 seconds and includes the surrounding recording; the completed route itself took 78 simulated seconds.

Contact-topic coverage was partial: only the base-link contact topic delivered messages. Zero obstacle pairs on that received stream and silence on rotor topics do not establish zero physical contact. The navigation process did not use simulator pose; the independent observer used truth only for offline scoring.

The [machine-readable result](results.json) includes source, model, launcher, PX4, checkpoint, inference-session, observer, scoring, and artifact hashes. Raw run artifacts are retained locally and are not included in a fresh source export.
