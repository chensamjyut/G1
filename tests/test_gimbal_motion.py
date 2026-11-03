import unittest
from planner.gimbal_motion import move_angles


class GimbalMotion(unittest.TestCase):
    def test_pitch_sweep_retains_yaw_and_exact_five_degree_endpoint(self):
        writes=[];ticks=[]
        trace=move_angles((3.,-10.),(3.,-5.),dt=.02,write=lambda *q:writes.append(q),tick=lambda:ticks.append(True))
        self.assertEqual(len(ticks),25)
        self.assertEqual(writes[-1],(3.,-5.))
        self.assertTrue(all(q[0]==3. for q in writes))
        self.assertAlmostEqual(trace[-1]['elapsed_sim_s'],.5)
        self.assertTrue(all(abs(b[1]-a[1])<=.20000001 for a,b in zip([(3.,-10.)]+writes,writes)))

    def test_yaw_sweep_retains_pitch(self):
        writes=[]
        move_angles((0.,-5.),(-5.,-5.),dt=.02,write=lambda *q:writes.append(q),tick=lambda:None)
        self.assertEqual(writes[-1],(-5.,-5.))
        self.assertTrue(all(q[1]==-5. for q in writes))

    def test_invalid_command_rejected_before_writing(self):
        for target,dt in [((0.,10.),.02),((float('nan'),0.),.02),((0.,5.),0.)]:
            writes=[]
            with self.subTest(target=target,dt=dt),self.assertRaises(ValueError):
                move_angles((0.,0.),target,dt=dt,write=lambda *q:writes.append(q),tick=lambda:None)
            self.assertEqual(writes,[])

    def test_failed_tick_never_continues_to_next_command(self):
        writes=[]
        def fail():raise RuntimeError('collision')
        with self.assertRaises(RuntimeError):
            move_angles((0.,0.),(0.,5.),dt=.02,write=lambda *q:writes.append(q),tick=fail)
        self.assertEqual(len(writes),1)
