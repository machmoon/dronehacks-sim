#!/usr/bin/env bash
# Package only the verified binary and manifest onto the builder's runtime base.
set -euo pipefail

scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
context=${PX4_DOCKER_CONTEXT:-colima-dronehacks-sim}
builder=${PX4_HRT_BUILDER:-dronehacks-markerfree-hrt-builder}
base=${PX4_HRT_RUNTIME_BASE:-dronehacks-markerfree:base}
target=${PX4_HRT_IMAGE:-dronehacks-markerfree:hrt-v1.16.2}
stage=${1:?Pass a new staging directory}
docker=(docker --context "$context")

base_id=$("${docker[@]}" container inspect --format '{{.Image}}' "$builder")
test "$("${docker[@]}" image inspect --format '{{.Id}}' "$base")" = "$base_id"
[[ $base_id =~ ^sha256:[0-9a-f]{64}$ ]]
base_pin="dronehacks-markerfree:hrt-base-${base_id:7:64}"
"${docker[@]}" image tag "$base_id" "$base_pin"
test "$("${docker[@]}" image inspect --format '{{.Id}}' "$base_pin")" = "$base_id"

mkdir "$stage"
"${docker[@]}" cp "$builder:/opt/px4/bin/px4" "$stage/px4"
"${docker[@]}" cp "$builder:/opt/px4/build-manifest.json" "$stage/build-manifest.json"
python3 - "$stage" "$base_id" "$scripts" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

stage, scripts = Path(sys.argv[1]), Path(sys.argv[3])
manifest_path = stage/'build-manifest.json'
manifest = json.loads(manifest_path.read_text())
digest = hashlib.sha256((stage/'px4').read_bytes()).hexdigest()
if manifest['px4_binary_sha256'] != digest:
    raise ValueError('staged PX4 binary disagrees with build manifest')
manifest['runtime_base_image_id'] = sys.argv[2]
manifest['package_script_sha256'] = hashlib.sha256((scripts/'package-hrt-backport.sh').read_bytes()).hexdigest()
manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
PY

"${docker[@]}" build --network none --build-arg "PX4_RUNTIME_BASE=$base_pin" \
    -f "$scripts/Dockerfile.hrt-backport" -t "$target" "$stage"
target_id=$("${docker[@]}" image inspect --format '{{.Id}}' "$target")
base_layers=$("${docker[@]}" image inspect --format '{{json .RootFS.Layers}}' "$base_id")
target_layers=$("${docker[@]}" image inspect --format '{{json .RootFS.Layers}}' "$target_id")
python3 - "$base_layers" "$target_layers" <<'PY'
import json
import sys

base, target = map(json.loads, sys.argv[1:])
if len(target) <= len(base) or target[:len(base)] != base:
    raise ValueError('patched image does not derive from the recorded runtime base')
PY
"${docker[@]}" run --rm --network none --entrypoint /usr/bin/python3 "$target_id" -c '
import hashlib,json,subprocess,sys
from pathlib import Path

px4=Path("/opt/px4/bin/px4")
manifest=json.loads(Path("/opt/px4/build-manifest.json").read_text())
digest=hashlib.sha256(px4.read_bytes()).hexdigest()
if manifest["runtime_base_image_id"] != sys.argv[1] or manifest["px4_binary_sha256"] != digest:
    raise SystemExit("packaged PX4 provenance mismatch")
links=subprocess.run(["ldd",str(px4)],capture_output=True,text=True,check=True)
if "not found" in links.stdout:
    raise SystemExit(links.stdout)
print("Verified packaged PX4 binary SHA-256", digest)
' "$base_id"
printf 'PX4 image %s %s\nRuntime base image %s\n' "$target" "$target_id" "$base_id"
