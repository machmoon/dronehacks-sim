import math
import unittest
from unittest.mock import patch
from mission.keyboard_rc import movement
from mission.keyboard_input import InputMailbox

class KeyboardRCTests(unittest.TestCase):
    def test_body_forward_rotates_with_yaw(self):
        x,y,z,r=movement({'KeyW'},math.pi/2)
        self.assertAlmostEqual(x,0)
        self.assertAlmostEqual(y,.6)
        self.assertEqual((z,r),(0,0))
    def test_diagonal_is_speed_limited_and_opposites_cancel(self):
        x,y,*_=movement({'KeyW','KeyA'},0)
        self.assertAlmostEqual(math.hypot(x,y),.6)
        self.assertEqual(movement({'KeyW','KeyS','KeyQ','KeyE'},0),(0,0,0,0))
    def test_expiry_and_out_of_order_packets_cannot_extend_control(self):
        box=InputMailbox(0)
        try:
            row=dict(updated_at=100.,enabled=True,keys=['KeyW'],action='takeoff',action_id=1)
            self.assertTrue(box.accept(row,10))
            self.assertTrue(box.poll(10.5)['enabled'])
            with patch('mission.keyboard_input.time.monotonic',return_value=10.1):
                self.assertTrue(box.poll()['enabled'])
            self.assertFalse(box.accept(row,10.5))
            self.assertIsNone(box.poll(10.7))
            self.assertFalse(box.accept(dict(row,updated_at=float('nan')),10.8))
            self.assertFalse(box.accept(dict(row,keys=[{}]),10.8))
            self.assertTrue(box.accept(dict(row,updated_at=101,enabled=False,action='land',action_id=2),11))
            self.assertFalse(box.poll(11)['enabled'])
        finally:box.close()

if __name__=='__main__':unittest.main()
