#!/usr/bin/env bash
# Run inside a disposable ARM64 builder from the existing marker-free runtime image.
set -euo pipefail

scripts=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(dirname "$scripts")
source_dir=/tmp/PX4-Autopilot
venv=/tmp/px4-hrt-venv
revision=54f0455ffcd755534539a7cf33a09a20bf71d29d
patch_path=patches/px4-posix-hrt-serialization.patch
patch_sha=3d7560bf9154091b0c7a2ce5230ac06af719cb51a15e4aa6ab0b65a1fbe0d61c

if [[ ${1:-} == dependencies ]]; then
    apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 update -qq
    DEBIAN_FRONTEND=noninteractive apt-get -y --no-install-recommends install \
        build-essential cmake cppzmq-dev git libeigen3-dev libgz-sim8-dev \
        libunwind-dev libxml2-utils ninja-build pkg-config protobuf-compiler python3-venv rsync
    exit 0
fi
if [[ ${1:-} != compile || $# != 1 ]]; then
    printf 'Usage: %s {dependencies|compile}\n' "$0" >&2
    exit 2
fi

test "$(sha256sum "$repo/$patch_path" | cut -d' ' -f1)" = "$patch_sha"
if [[ ! -d $source_dir/.git ]]; then
    git -c advice.detachedHead=false clone --depth 1 --filter=blob:none --sparse --branch v1.16.2 \
        https://github.com/PX4/PX4-Autopilot.git "$source_dir"
fi
test "$(git -C "$source_dir" rev-parse HEAD)" = "$revision"
git -C "$source_dir" sparse-checkout set --no-cone '/*' '!/docs/'
if ! git -C "$source_dir" apply --reverse --check "$repo/$patch_path" >/dev/null 2>&1; then
    git -C "$source_dir" apply --check --whitespace=error-all "$repo/$patch_path"
    git -C "$source_dir" apply "$repo/$patch_path"
fi
git -C "$source_dir" diff --check

if [[ ! -x $venv/bin/python ]]; then
    python3 -m venv "$venv"
fi
"$venv/bin/python" -m pip --disable-pip-version-check install --no-cache-dir \
    -r "$scripts/requirements-build.txt"
# The pinned Makefile can miss the job count in a noninteractive container.
PATH="$venv/bin:$PATH" GZ_DISTRO=harmonic PX4_BUILD_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1 \
    bash "$scripts/build.sh"

python3 - "$repo" "$revision" "$patch_path" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

repo = Path(sys.argv[1])
patch = sys.argv[3]
px4 = Path('/opt/px4')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    'schema_version': 1,
    'base_revision': sys.argv[2],
    'patches': [{'path': patch, 'sha256': sha(repo/patch)}],
    'px4_binary_sha256': sha(px4/'bin/px4'),
    'build_adjustments': ['mavlink_loopback_only', 'unused_gz_optical_flow_plugin_disabled'],
    'build_script_sha256': sha(repo/'px4/build-hrt-backport.sh'),
    'base_build_script_sha256': sha(repo/'px4/build.sh'),
    'build_target': 'px4_sitl',
    'compiler_jobs': 1,
}
(px4/'build-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
