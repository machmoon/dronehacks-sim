#!/usr/bin/env bash
set -euo pipefail

pids=()
cleanup() {
    trap - EXIT
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 0' INT TERM

start() {
    "$@" &
    pids+=("$!")
}

args=(-r -v 3)
if [[ -n ${SIM_MARKERFREE_GZ_SEED:-} ]]; then
    if [[ ${GZ_PARTITION:-} != dronehacks_markerfree || ! $SIM_MARKERFREE_GZ_SEED =~ ^[1-9][0-9]*$ ]]; then
        echo 'Invalid marker-free Gazebo seed or partition.' >&2
        exit 1
    fi
    args+=(--seed "$SIM_MARKERFREE_GZ_SEED")
fi
world=${SIM_WORLD:-worlds/course.sdf}
gui_args=()
if [[ ${SIM_GUI:-1} == 1 ]]; then
    install -d -m 700 "$XDG_RUNTIME_DIR"
    # Match the actual Qt window to the virtual display, otherwise a smaller
    # Xvfb screen clips the default 1280x800 Gazebo window and its 3D view.
    python3 - "${SIM_DESKTOP_SIZE:-1200x1000}" "$world" <<'PY'
import re
import shutil
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
match = re.fullmatch(r'([0-9]{3,4})x([0-9]{3,4})', sys.argv[1])
if not match:
    raise SystemExit('SIM_DESKTOP_SIZE must be WIDTHxHEIGHT')
config = Path.home() / '.gz/sim/8/gui.config'
config.parent.mkdir(parents=True, exist_ok=True)
if not config.exists():
    shutil.copy('/usr/share/gz/gz-sim8/gui/gui.config', config)
# Gazebo GUI configs are XML fragments (window and plugin are siblings),
# not necessarily a single-root XML document. Preserve all other fragments.
text = config.read_text()
window_match = re.search(r'<window(?:\s[^>]*)?>.*?</window>', text, re.S)
window = ET.fromstring(window_match.group()) if window_match else ET.Element('window')
for key, value in zip(('width', 'height'), match.groups()):
    field = window.find(key)
    if field is None:
        field = ET.SubElement(window, key)
    field.text = value
updated = ET.tostring(window, encoding='unicode')
if window_match:
    text = text[:window_match.start()] + updated + text[window_match.end():]
else:
    text = updated + '\n' + text
config.write_text(text)
# Explicit GUI config has higher precedence than the world's plugin block.
# Copy that block into it so both window dimensions and course camera apply.
if len(sys.argv) > 2:
    gui = ET.parse(sys.argv[2]).getroot().find('world/gui')
    if gui is not None:
        text = updated + '\n' + '\n'.join(
            ET.tostring(plugin, encoding='unicode') for plugin in gui.findall('plugin'))
config.with_name('dronehacks-gui.config').write_text(text)
PY
    gui_args=(--gui-config "$HOME/.gz/sim/8/dronehacks-gui.config")
    start Xvfb "$DISPLAY" -screen 0 "${SIM_DESKTOP_SIZE:-1200x1000}x24" -nolisten tcp
    for ((attempt = 0; attempt < 100; attempt++)); do
        if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
    if ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
        echo "Xvfb did not start on $DISPLAY" >&2
        exit 1
    fi
    start x11vnc -display "$DISPLAY" -listen 127.0.0.1 -rfbport 5900 -nopw \
        -forever -shared -noxdamage -quiet
    start websockify --web /usr/share/novnc --heartbeat 30 0.0.0.0:6080 127.0.0.1:5900
else
    args+=(-s)
    if [[ ${SIM_HEADLESS_RENDERING:-0} == 1 ]]; then
        args+=(--headless-rendering)
    fi
fi

if [[ ${SIM_GUI:-1} == 1 && ${SIM_SEPARATE_GUI:-0} == 1 ]]; then
    # Start the server independently so rendering and GUI discovery cannot
    # block the server's initial Gazebo transport handshake.
    start stdbuf -oL -eL gz sim -s --headless-rendering "${args[@]}" "$world"
    server_pid=$!
    world_ready=0
    for ((attempt = 0; attempt < 30; attempt++)); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            break
        fi
        if gz service -l | grep -E '^/world/[^/]+/scene/info$' >/dev/null; then
            world_ready=1
            break
        fi
        sleep 0.25
    done
    if (( ! world_ready )); then
        echo "Gazebo server did not advertise a scene" >&2
        exit 1
    fi
    start stdbuf -oL -eL gz sim -g -v 3 "${gui_args[@]}"
else
    start gz sim "${args[@]}" "${gui_args[@]}" "$world"
fi
services=("${pids[@]}")
if [[ ${SIM_MISSION:-1} == 1 ]]; then
    mkdir -p artifacts/gazebo
    start python3 -m mission.run \
        --course="${SIM_COURSE:-mission/course.json}" \
        --speed-scale="${SIM_SPEED_SCALE:-0.65}" \
        --output /workspace/artifacts/gazebo/telemetry.jsonl
fi

wait -n "${services[@]}"
