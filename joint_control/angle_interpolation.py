'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''

import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'software_installation'))

from pid import PIDAgent
from keyframes import hello, wipe_forehead
from scipy import interpolate
from spark_agent import Perception



class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def init_splines(self):
        joint_angles = list(map(lambda joint : list(map(lambda key : key[0], joint)), self.keyframes[2]))
        self.splines = list(map(lambda times, angles : interpolate.splrep(times, angles, s=0), self.keyframes[1], joint_angles)) 

    def angle_interpolation(self, keyframes, perception: Perception):
        if not hasattr(self, 'start_time'):
            self.start_time = perception.time
            
        target_angles = list(map(lambda spline : interpolate.splev([perception.time - self.start_time], spline)[0], self.splines))

        return dict(zip(keyframes[0], target_angles))

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = wipe_forehead()  # CHANGE DIFFERENT KEYFRAMES
    agent.init_splines()
    agent.run()
