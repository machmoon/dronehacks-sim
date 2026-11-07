import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from mission.runtime_provenance import px4_runtime_provenance


class RuntimeProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root/'bin').mkdir()
        (self.root/'bin/px4').write_bytes(b'executable fixture')
        self.digest = hashlib.sha256(b'executable fixture').hexdigest()

    def test_base_without_manifest_still_identifies_binary(self):
        result = px4_runtime_provenance(self.root)
        self.assertEqual(result['px4_binary_sha256'], self.digest)
        self.assertFalse(result['manifest_verified'])
        self.assertIsNone(result['build_manifest'])

    def test_changed_binary_cannot_use_old_manifest(self):
        manifest = {'schema_version': 1, 'px4_binary_sha256': self.digest}
        (self.root/'build-manifest.json').write_text(json.dumps(manifest))
        self.assertTrue(px4_runtime_provenance(self.root)['manifest_verified'])
        (self.root/'bin/px4').write_bytes(b'different build')
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            px4_runtime_provenance(self.root)

    def test_unrecognized_manifest_is_rejected(self):
        for manifest in ({'schema_version': 2, 'px4_binary_sha256': self.digest}, [], {}):
            with self.subTest(manifest=manifest):
                (self.root/'build-manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(RuntimeError):
                    px4_runtime_provenance(self.root)
