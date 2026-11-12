import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web import server


class ViewerEvidenceTests(unittest.TestCase):
    def _state_for_run(self, root, run_id, score_run_id):
        selected = root / 'selected-run'
        selected.mkdir(parents=True)
        telemetry = selected / 'telemetry.jsonl'
        rows = [
            {'type': 'metadata', 'run_id': run_id, 'navigation_source': 'RGBD + down-facing ToF'},
            {'type': 'sample', 'position': [0, 0, 1], 'sim_time': 1.0},
        ]
        telemetry.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (selected / 'score.json').write_text(json.dumps({'run_id': score_run_id, 'full_route_verified_by_geometry': True}))
        camera = selected / 'camera'
        camera.mkdir()
        (camera / 'latest.json').write_text(json.dumps({'source_run': 'selected-run'}))
        (camera / 'latest.jpg').write_bytes(b'selected-camera')
        (camera / 'depth.jpg').write_bytes(b'selected-depth')
        unrelated = root / 'old-current'
        unrelated.mkdir()
        (unrelated / 'score.json').write_text(json.dumps({'run_id': 'unrelated-run', 'full_route_verified_by_geometry': True}))
        old_camera = unrelated / 'camera'
        old_camera.mkdir()
        (old_camera / 'latest.json').write_text(json.dumps({'source_run': 'unrelated-run'}))
        (old_camera / 'latest.jpg').write_bytes(b'unrelated-camera')
        (old_camera / 'depth.jpg').write_bytes(b'unrelated-depth')
        return telemetry, unrelated

    def _read_state(self, telemetry, unrelated_dir, vision_dir=None):
        with patch.object(server, 'telemetry_path', return_value=telemetry), \
             patch.object(server, 'MARKERFREE', unrelated_dir), \
             patch.object(server, 'PREVIEW', unrelated_dir / 'preview'), \
             patch.object(server, 'VISION', vision_dir or (unrelated_dir / 'vision')):
            return server.state()

    def test_score_comes_from_selected_telemetry_directory(self):
        with tempfile.TemporaryDirectory() as td:
            telemetry, unrelated = self._state_for_run(Path(td), 'selected-run', 'selected-run')
            state = self._read_state(telemetry, unrelated)
            self.assertEqual(state['score']['run_id'], 'selected-run')
            self.assertEqual(state['source'], str(telemetry))
            self.assertEqual(state['camera']['source_run'], 'selected-run')
            self.assertEqual(state['camera_url'], '/rgbd-camera.jpg')
            self.assertEqual(state['depth_url'], '/rgbd-depth.jpg')
            with patch.object(server, 'telemetry_path', return_value=telemetry):
                self.assertEqual(server.selected_camera_dir(), telemetry.parent / 'camera')


    def test_markerfree_run_does_not_fall_back_to_legacy_camera(self):
        with tempfile.TemporaryDirectory() as td:
            telemetry, unrelated = self._state_for_run(Path(td), 'selected-run', 'selected-run')
            (telemetry.parent / 'camera' / 'latest.jpg').unlink()
            vision = Path(td) / 'vision'
            vision.mkdir()
            (vision / 'camera.jpg').write_bytes(b'legacy annotated image')
            state = self._read_state(telemetry, unrelated, vision)
            self.assertIsNone(state['camera_url'])

    def test_score_with_mismatched_run_id_is_not_exposed(self):
        with tempfile.TemporaryDirectory() as td:
            telemetry, unrelated = self._state_for_run(Path(td), 'selected-run', 'other-run')
            state = self._read_state(telemetry, unrelated)
            self.assertIsNone(state['score'])

    def test_score_is_not_exposed_when_metadata_has_no_run_id(self):
        with tempfile.TemporaryDirectory() as td:
            telemetry, unrelated = self._state_for_run(Path(td), None, None)
            state = self._read_state(telemetry, unrelated)
            self.assertIsNone(state['score'])


if __name__ == '__main__':
    unittest.main()
