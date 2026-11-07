# Keyboard RC in local Gazebo

From the simulator repository on the laptop:

```sh
cd /Users/patliu/Desktop/dronehacks/dronehacks-sim
bash runtime/run-keyboard.sh
```

Open <http://127.0.0.1:8087>. Keep the launcher terminal open if it is serving the
panel. Wait for Ready, click **Enable keyboard RC**, then **Take off**. Enabling
keys alone never arms. The simulated drone climbs to 0.95 m before accepting
movement keys.

- W/S move forward/back relative to the drone's current heading.
- A/D move left/right relative to its heading.
- Up/down arrows change altitude. Q/E turn left/right.
- Releasing movement keys or pressing Space holds position.
- Land requests PX4 landing at the current position.
- Escape, disabling RC, switching tabs or losing window focus clears inputs and
  requests landing. A missing input heartbeat also triggers landing.

This is assisted velocity control through PX4 Offboard, not a physical RC radio
or raw motor control. The simulated x500 uses rendered RGB-D map localization.
It does not avoid obstacles for the pilot. It cannot send commands to hardware:
MAVLink is fixed to container loopback, and both the simulator-only and keyboard
runtime flags are required. The web service binds only to laptop loopback.

The separate Compose project is `dronehacks-keyboard`, with its Gazebo display
on port 8086, control panel on 8087, and its local control connection on 8088. Files are under `artifacts/keyboard`.
The launcher reuses an already-running keyboard session rather than resetting
an aircraft in flight. To reset, click Land and wait for disarm first, then:

```sh
docker --context colima-dronehacks-sim compose -p dronehacks-keyboard \
  -f compose.px4.yaml -f compose.markerfree.yaml -f compose.keyboard.yaml down
bash runtime/run-keyboard.sh
```

If the Docker VM is stopped, start the existing profile first:

```sh
colima start --profile dronehacks-sim --activate=false
```

The launcher requires the `dronehacks-markerfree:hrt-v1.16.2` image. A deleted
Colima VM also deletes its images; see [Mac setup](mac-quickstart.md) and the
[HRT build](px4-hrt-backport.md) to rebuild them. Do not remove `~/.colima` while
running the simulator.

## Checked on this laptop

The rebuilt runtime completed a takeoff, forward command, hover, landing and
disarm test. A separate Chrome test confirmed the Enable and Take off buttons,
a held W key, Space, and Escape followed by touchdown/disarm. Three unit tests
cover body-relative directions, diagonal speed limiting, stale input expiry and
out-of-order packet rejection. [Validation record](keyboard-rc-validation.json).

The initial file-based control experiments were replaced by local TCP after
shared-file read gaps canceled flight. A receiver-clock race was also corrected.
The successful final tests used TCP; no claim is made that the earlier versions
worked. The rebuilt runtime reuses the saved, hash-verified HRT-patched PX4
binary, with fresh Ubuntu/Gazebo runtime packages. It has not rerun the earlier
seven-run autonomous course benchmark.
