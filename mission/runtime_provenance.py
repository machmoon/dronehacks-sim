"""Record the actual SITL executable and verify an optional build manifest."""
import hashlib
import json
from pathlib import Path


def px4_runtime_provenance(root=Path('/opt/px4')):
    root = Path(root)
    executable = root/'bin/px4'
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    manifest_path = root/'build-manifest.json'
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
                or manifest.get('px4_binary_sha256') != digest):
            raise RuntimeError('PX4 build manifest does not match the running image executable')
    return {
        'executable_path': str(executable),
        'px4_binary_sha256': digest,
        'build_manifest': manifest,
        'build_manifest_sha256': (hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                                  if manifest is not None else None),
        'manifest_verified': manifest is not None,
        'scope': 'On-disk SITL executable; launch script starts this path. Not hardware firmware.',
    }
