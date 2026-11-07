"""Image-level geometry, resource bound, and failure/recovery tests."""
import unittest

import cv2
import numpy as np

from tools.benchmark_sparse_slam import SyntheticRoom, trajectory
from vision.sparse_slam import Config, SparseMap
from tools.prepare_dpvo_clip import resize_calibration


class SparseMapTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(1);cv2.setRNGSeed(42)
        self.room=SyntheticRoom()
        self.map=SparseMap(self.room.k)

    def test_metric_translation_and_rotation_from_images(self):
        for i in range(20):
            pose=trajectory(i,120);rgb,depth=self.room.render(pose,.002)
            result=self.map.estimate(rgb,depth,1+i/30)
            self.assertTrue(result['valid'],result)
            predicted=np.array(result['map_from_camera'])
            self.assertLess(np.linalg.norm(predicted[:3,3]-pose[:3,3]),.025)
            self.assertLess(np.linalg.norm(cv2.Rodrigues(predicted[:3,:3].T@pose[:3,:3])[0]),.025)

    def test_blank_frame_has_no_pose_and_same_place_relocalizes(self):
        rgb,depth=self.room.render(np.eye(4))
        self.assertTrue(self.map.estimate(rgb,depth,1)['valid'])
        failure=self.map.estimate(np.zeros_like(rgb),depth,1.033)
        self.assertFalse(failure['valid']);self.assertIsNone(failure['map_from_camera'])
        recovered=self.map.estimate(rgb,depth,1.067)
        self.assertTrue(recovered['valid'],recovered);self.assertTrue(recovered['relocalized'])
        np.testing.assert_allclose(recovered['map_from_camera'],np.eye(4),atol=1e-5)
        self.assertFalse(recovered['flight_ready'])

    def test_wrong_depth_does_not_create_valid_motion(self):
        rgb,depth=self.room.render(np.eye(4));self.map.estimate(rgb,depth,1)
        result=self.map.estimate(rgb,depth*.5,1.033)
        self.assertFalse(result['valid']);self.assertIsNone(result['map_from_camera'])

    def test_unrelated_scene_cannot_relocalize(self):
        rgb,depth=self.room.render(np.eye(4));self.map.estimate(rgb,depth,1)
        self.map.estimate(np.zeros_like(rgb),depth,1.033)
        other,_=SyntheticRoom(seed=99).render(np.eye(4))
        result=self.map.estimate(other,depth,1.067)
        self.assertFalse(result['valid']);self.assertIsNone(result['map_from_camera'])

    def test_keyframe_and_track_memory_is_bounded_and_launch_anchor_retained(self):
        cfg=Config(features=96,recovery_features=192,keyframes=3,keyframe_period_s=.05)
        tracker=SparseMap(self.room.k,cfg)
        first=None
        for i in range(30):
            rgb,depth=self.room.render(trajectory(i,120))
            result=tracker.estimate(rgb,depth,1+i/30)
            self.assertTrue(result['valid'],result)
            if first is None:first=tracker.frames[0]
            self.assertLessEqual(len(tracker.frames),3)
            self.assertLessEqual(len(tracker.previous[1]),96)
            self.assertLessEqual(tracker.map_bytes,3*(192*(3*8+32)+16*8))
        self.assertIs(tracker.frames[0],first)
        self.assertEqual(len(tracker.frames),3)

    def test_resize_keeps_pixel_center_geometry_consistent(self):
        k=self.room.k.copy()
        new=resize_calibration(k,(320,240),(160,120))
        point=np.array([.2,-.1,2.])
        old_pixel=k@point;old_pixel=old_pixel[:2]/old_pixel[2]
        new_pixel=new@point;new_pixel=new_pixel[:2]/new_pixel[2]
        np.testing.assert_allclose(new_pixel,(old_pixel+.5)*.5-.5)

    def test_bad_timestamps_and_intrinsics_fail(self):
        rgb,depth=self.room.render(np.eye(4));self.map.estimate(rgb,depth,1)
        self.assertFalse(self.map.estimate(rgb,depth,.9)['valid'])
        with self.assertRaises(ValueError):SparseMap(np.zeros((3,3)))
        with self.assertRaises(ValueError):Config(features=8)
        with self.assertRaises(ValueError):Config(features=300,recovery_features=200)


if __name__=='__main__':unittest.main()
