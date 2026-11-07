import copy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from vehicle.skybeta import G, ROOT, build_model, load, properties, px4_parameters
from vision.scene import generate


class SkyBetaPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile, cls.geometry = load()

    def test_airframe_has_four_cad_motor_positions_and_one_jetson(self):
        p = properties(self.profile, self.geometry)
        self.assertEqual(len(p['rotors']), 4)
        self.assertEqual(sum('Jetson Orin' in c['name'] for c in p['parts']), 1)
        # Measured STEP motor centres, independently checked against raw CAD.
        np.testing.assert_allclose([r['position'][:2] for r in p['rotors']],
            [[.083133728,-.159653530],[-.147453250,.135127566],
             [.083133728,.159653530],[-.147453250,-.135127566]], atol=2e-5)

    def test_total_mass_has_no_duplicate_sensor_or_propeller_mass(self):
        sdf,p=build_model(self.profile,self.geometry)
        sdf_mass=sum(float(e.text) for e in sdf.findall('.//inertial/mass'))
        expected=sum(c['mass_kg'] for c in p['parts'])+self.profile['unmodelled_mass_kg']+4*self.profile['propeller']['mass_kg']
        self.assertAlmostEqual(sdf_mass,expected)
        self.assertAlmostEqual(p['mass'],expected)
        eigenvalues=np.linalg.eigvalsh(p['inertia'])
        self.assertTrue(np.all(eigenvalues>0))
        self.assertLessEqual(eigenvalues[-1],sum(eigenvalues[:2]))

    def test_hover_balances_force_and_all_three_torques(self):
        p=properties(self.profile,self.geometry)
        forces=np.zeros((4,3)); forces[:,2]=p['hover_thrust']
        arms=np.array([r['position']-p['com'] for r in p['rotors']])
        torque=np.cross(arms,forces).sum(axis=0)
        torque[2]+=sum(r['reaction_sign']*f*self.profile['motor']['moment_ratio_m']
                       for r,f in zip(p['rotors'],p['hover_thrust']))
        np.testing.assert_allclose(torque,0,atol=1e-12)
        self.assertAlmostEqual(sum(p['hover_thrust']),p['mass']*G)

    def test_px4_geometry_is_frd_about_com_and_matches_rotor_order(self):
        p=properties(self.profile,self.geometry); params=px4_parameters(self.profile,p)
        for row in p['rotors']:
            i=row['index']
            np.testing.assert_allclose([params[f'CA_ROTOR{i}_P{a}'] for a in 'XYZ'],
                                       (row['position']-p['com'])*[1,-1,-1])
            self.assertEqual(np.sign(params[f'CA_ROTOR{i}_KM']),-row['reaction_sign'])
        self.assertGreater(params['MPC_THR_HOVER'],0)
        self.assertLess(params['MPC_THR_HOVER'],1)

    def test_heavier_battery_moves_com_and_increases_hover_power(self):
        old=properties(self.profile,self.geometry)
        profile=copy.deepcopy(self.profile); profile['component_mass_kg']['OVONIC']*=1.3
        new=properties(profile,self.geometry)
        self.assertGreater(new['mass'],old['mass'])
        self.assertGreater(np.linalg.norm(new['hover_speed']),np.linalg.norm(old['hover_speed']))
        self.assertGreater(np.linalg.norm(new['com']-old['com']),.001)

    def test_impossible_thrust_profile_fails_closed(self):
        profile=copy.deepcopy(self.profile); profile['motor']['thrust_coefficient']=1e-9
        with self.assertRaisesRegex(ValueError,'cannot hover'):
            properties(profile,self.geometry)

    def test_default_scene_is_skybeta_with_forward_stereo_and_no_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'scene.sdf';generate(ROOT/'worlds/course.sdf',out,0)
            root=ET.parse(out)
            self.assertFalse(root.findall('.//include[uri="model://x500"]'))
            self.assertEqual(len(root.findall('.//plugin[@name="gz::sim::systems::MulticopterMotorModel"]')),4)
            self.assertIsNone(root.find('.//model[@name="vision_landmarks"]'))
            sensors=[root.find(f'.//sensor[@name="oak_{side}"]') for side in ('left','right')]
            poses=[np.array(list(map(float,s.findtext('pose').split()))) for s in sensors]
            self.assertGreater(poses[0][0],.10)
            self.assertAlmostEqual(poses[0][1]-poses[1][1],.075)
            np.testing.assert_array_equal(poses[0][3:],0)

    def test_course_collision_and_visual_names_are_unique(self):
        root=ET.parse(ROOT/'models/course_v3/model.sdf')
        for link in root.findall('.//link'):
            names=[e.get('name') for e in link if e.get('name')]
            self.assertEqual(len(names),len(set(names)))

    def test_legacy_downward_camera_remains_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'scene.sdf'; generate(ROOT/'worlds/course.sdf',out,0,airframe='x500')
            root=ET.parse(out)
            self.assertIsNotNone(root.find('.//include[uri="model://x500"]'))
            self.assertIsNotNone(root.find('.//sensor[@name="down_camera"]'))
            with self.assertRaisesRegex(ValueError,'diagnostic requires'):
                generate(ROOT/'worlds/course.sdf',out,0,with_test_markers=True,airframe='skybeta')


if __name__=='__main__': unittest.main()
