#!/usr/bin/env bash
# Starts a dedicated local simulator. It cannot connect to the physical drone.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
context=${SIM_DOCKER_CONTEXT:-colima-dronehacks-sim}
export SIM_PX4_IMAGE=dronehacks-markerfree:hrt-v1.16.2
export SIM_GUI=1 GZ_GUI_PORT=8086 SIM_RGBD_DEPTH_NOISE=off
if ! docker --context "$context" info >/dev/null 2>&1; then
  if [[ "$context" == colima-dronehacks-sim ]]; then
    colima start --profile dronehacks-sim --activate=false
  else
    echo "Start Docker context $context before launching keyboard RC." >&2
    exit 1
  fi
fi
if ! docker --context "$context" image inspect "$SIM_PX4_IMAGE" >/dev/null 2>&1; then
  echo "Simulator image is missing. Follow docs/mac-quickstart.md and docs/px4-hrt-backport.md to rebuild it." >&2
  exit 1
fi
mkdir -p artifacts/keyboard
# Leave an already-running controller and its input state alone.
if ! docker --context "$context" inspect -f '{{.State.Running}}' dronehacks-keyboard-sim-1 2>/dev/null | grep -qx true; then
  printf '%s\n' '{"enabled":false,"keys":[],"action":"none","action_id":0,"updated_at":0}' > artifacts/keyboard/input.json
  docker --context "$context" compose -p dronehacks-keyboard -f compose.px4.yaml -f compose.markerfree.yaml -f compose.keyboard.yaml up --no-build -d
fi
if ! curl -fsS http://127.0.0.1:8087/api/rc/status >/dev/null 2>&1; then
  printf "Keyboard RC panel: http://127.0.0.1:8087 — keep this terminal open.\n"
  exec python3 -m web.rc_server
fi
printf '\nKeyboard RC panel: http://127.0.0.1:8087\nEnable keyboard RC, then click Take off after the simulator is ready.\n'
