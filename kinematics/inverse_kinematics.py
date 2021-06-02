'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''

from math import atan2, sqrt, pi, pow
import numpy as np

from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity

MINIMAL_ERROR = 1e-3


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''
        chain = self.chains[effector_name]
        joints = {joint_name: self.perception.joint[joint_name] for joint_name in self.chains[effector_name]}
        end_joint = chain[-1]
        transform_target = self.transform_values_of(transform)

        joint_angles = np.zeros(len(joints))

        iterations = 0
        error_unsatisfied = True
        while iterations < 1000 and error_unsatisfied:

            self.forward_kinematics(joints)

            transform_current = self.transform_values_of(self.transforms[end_joint])
            transform_error = transform_target - transform_current

            jacobian = self.calc_jacobian(transform_target, joints)
            delta_angles = self.jacobian_transpose(jacobian, transform_error)

            joint_angles += delta_angles

            for i, key in enumerate(joints.keys()):
                joints[key] += delta_angles[i]

            error_unsatisfied = np.inner(transform_error, transform_error) >= MINIMAL_ERROR
            iterations += 1

        return joint_angles

    def calc_jacobian(self, transform_target, joints):
        jacobian = np.zeros((6, len(joints)))
        for i, (x, t) in enumerate(self.transforms.iteritems()):
            x, y, z = self.coordinates_of(t)
            jacobian[: i] = (transform_target - np.array([x, y, z, 0, 0, 0]))
        for i, joint in enumerate(joints):
            index = 0
            if 'Pitch' in joint:
                index = 3
            if 'Roll' in joint:
                index = 5
            if 'Yaw' in joint:
                index = 4
            jacobian[index, i] = 1
        return jacobian

    def jacobian_transpose(self, jacobian, transform_error):
        jacobian_transposed_error_skalar = np.dot(np.dot(jacobian, jacobian.T), transform_error)
        alpha = float(np.inner(transform_error, jacobian_transposed_error_skalar)) / float(np.inner(jacobian_transposed_error_skalar, jacobian_transposed_error_skalar))
        return alpha * np.dot(jacobian.T, transform_error)

    def coordinates_of(self, transform):
        x = transform[-1, 0]
        y = transform[-1, 1]
        z = transform[-1, 2]

        return x, y, z

    def transform_values_of(self, transform):
        x, y, z = self.coordinates_of(transform)

        if(transform[2, 0] > MINIMAL_ERROR):
            omega_x = atan2(transform[2, 1], transform[2, 2])
            omega_y = atan2(-transform[2, 0], sqrt(pow(transform[2, 1], 2) + pow(transform[2, 2], 2)))
            omega_z = atan2(transform[1, 0], transform[0, 0])
        else:
            if(abs(transform[2, 0] + 1) < MINIMAL_ERROR):
                omega_x = atan2(transform[0, 1], transform[0, 2])
                omega_y = pi / 2
            else:
                omega_x = atan2(-transform[0, 1], -transform[0, 2])
                omega_y = -pi / 2
            omega_z = 0
        return np.array([x, y, z, omega_x, omega_y, omega_z])

    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        names = []
        times = []
        keys = []
        self.forward_kinematics(self.perception.joint)
        angles = self.inverse_kinematics(effector_name, transform)
        for i, joint in enumerate(self.chains[effector_name]):
            names.append(joint)
            times.append([1.0, 3.0])
            keys.append([[angles[i] - 0.01, [3, 0, 0], [3, 0, 0]],
                        [angles[i], [3, 0, 0], [3, 0, 0]]])

        self.start_animation((names, times, keys))


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = identity(4)
    T[-1, 1] = 0.05
    T[-1, 2] = 0.26
    agent.set_transforms('LLeg', T)
    agent.run()
