#!/usr/bin/env bash
set -euo pipefail
cd /workspace
bash runtime/px4-start.sh &
sim_pid=$!
python3 -m mission.keyboard_rc &
rc_pid=$!
cleanup() {
  trap - EXIT
  kill -TERM "$rc_pid" 2>/dev/null || true
  wait "$rc_pid" 2>/dev/null || true
  kill -TERM "$sim_pid" 2>/dev/null || true
  wait "$sim_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 0' TERM INT
wait -n "$sim_pid" "$rc_pid"
