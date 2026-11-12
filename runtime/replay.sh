#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
docker --context colima-dronehacks-sim compose -f compose.px4.yaml stop sim
docker --context colima-dronehacks-sim compose up --no-build -d --force-recreate sim
