'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    the local_trans has to consider different joint axes and link parameters for different joints
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

from numpy.matlib import matrix, identity
import numpy as np

from angle_interpolation import AngleInterpolationAgent


class ForwardKinematicsAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: identity(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {
            'Head': ['HeadYaw', 'HeadPitch'],
            'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll'],
            'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
            'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'],
            'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll']
        }

        self.joint_limits = {
            'HeadYaw': [0, 0, 126.5, 1],
            'HeadPitch': [0, 0, 0, 1],
            'RHipYawPitch': [0, -50, -85, 1],
            'RHipRoll': [0, 0, 0, 1],
            'RHipPitch': [0, 0, 0, 1],
            'RKneePitch': [0, 0, -100, 1],
            'RAnkleRoll': [0, 0, 0, 1],
            'RAnklePitch': [0, 0, -102.9, 1],
            'LHipYawPitch': [0, 50, -85, 1],
            'LHipRoll': [0, 0, 0, 1],
            'LHipPitch': [0, 0, 0, 1],
            'LKneePitch': [0, 0, -100, 1],
            'LAnkleRoll': [0, 0, 0, 1],
            'LAnklePitch': [0, 0, -102.9, 1],
            'RShoulderPitch': [0, -98, 100, 1],
            'RShoulderRoll': [0, 0, 0, 1],
            'RElbowYaw': [105, -15, 0, 1],
            'RElbowRoll': [0, 0, 0, 1],
            'LShoulderPitch': [0, 98, 100, 1],
            'LShoulderRoll': [0, 0, 0, 1],
            'LElbowYaw': [105, 15, 0, 1],
            'LElbowRoll': [0, 0, 0, 1],
        }


    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, joint_angle):
        '''calculate local transformation of one joint

        :param str joint_name: the name of joint
        :param float joint_angle: the angle of joint in radians
        :return: transformation
        :rtype: 4x4 matrix
        '''
        
        sin = np.sin(joint_angle)
        cos = np.cos(joint_angle)
        limits = self.joint_limits[joint_name]

        if 'Pitch' in joint_name:
            return matrix([
                [cos, 0, sin, 0],
                [0, 1, 0, 0],
                [-1 * sin, 0, cos, 0],
                limits
            ])
        elif 'Roll' in joint_name:
            return matrix([
                [1, 0, 0, 0],
                [0, cos, -1 * sin, 0],
                [0, sin, cos, 0],
                limits
            ])
        elif 'Yaw' in joint_name:
            return matrix([
                [cos, sin, 0, 0],
                [-1 * sin, cos, 0, 0],
                [0, 0, 1, 0],
                limits
            ])

        return identity(4)

    def forward_kinematics(self, joints):
        '''forward kinematics

        :param joints: {joint_name: joint_angle}
        '''
        T = identity(4)
        for joint in joints:
            angle = joints[joint]
            Tl = self.local_trans(joint, angle)
            # YOUR CODE HERE

            self.transforms[joint] = np.dot(T, Tl)

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
