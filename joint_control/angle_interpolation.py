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
from keyframes import hello, rightBackToStand
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

    def start_animation(self, keyframes):
        self.start_time = -1
        self.keyframes = keyframes
        joint_angles = list(map(lambda joint : list(map(lambda key : key[0], joint)), self.keyframes[2]))
        self.splines = list(map(lambda times, angles : interpolate.splrep(times, angles, s=0), self.keyframes[1], joint_angles))
        print('started animation')

    def stop_animation(self):
        del self.start_time
        self.keyframes = ([],[],[])
        print('stopped animation')

    def is_animating(self):
        return hasattr(self, 'start_time')

    def angle_interpolation(self, keyframes, perception):
        if not self.is_animating():
            return {}

        if self.start_time == -1:
            self.start_time = perception.time

        time = perception.time - self.start_time
            
        finished_keyframes = list(map(lambda times : times[-1] < time, keyframes[1]))

        last_angles = list(map(lambda keys : keys[-1][0], keyframes[2]))
        target_angles = list(map(lambda spline, finished, last_angle : interpolate.splev([time], spline)[0] if not finished else last_angle, self.splines, finished_keyframes, last_angles))

        if all(finished_keyframes):
            self.stop_animation()

        target_joints = dict(zip(keyframes[0], target_angles))
        target_joints["RHipYawPitch"] = target_joints["LHipYawPitch"]
        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()     
    agent.start_animation(rightBackToStand()) # CHANGE DIFFERENT KEYFRAMES
    agent.run()
