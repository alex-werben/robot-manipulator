import pybullet as p
import time
import pybullet_data
import random, numpy, math, time
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import display, HTML
import os

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11 #8
pandaNumDofs = 7
ll = [-7]*pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7]*pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7]*pandaNumDofs
jointPositions=[0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]

rp = jointPositions
class Environment:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -1)
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.panda = p.loadURDF("franka_panda/panda.urdf",
                                useFixedBase=True,
                                basePosition=[0, 0, 0.5],
                                baseOrientation=orientation)
        self.plane = p.loadURDF("plane.urdf")
        self.lego = p.loadURDF("lego/lego.urdf", basePosition=[0.5, 0.5, 0.6])
        self.gripper_height = 0.2 + 0.5
        self.state = 0
        self.control_dt = 1. / 240.
        self.finger_target = 0
        self.offset = [0, 0, 0]
        c = p.createConstraint(self.panda,
                               9,
                               self.panda,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        p.setAdditionalSearchPath(os.getcwd())

        self.surface = p.loadURDF("assets/cube.urdf", basePosition=[0.25, 0, 0.25], useFixedBase=True)
        self.end_position = p.loadURDF("assets/position.urdf", basePosition=[0.6, 0.5, 0.525], useFixedBase=True)
        self.start_position = p.loadURDF("assets/position.urdf", basePosition=[0.6, -0.5, 0.525], useFixedBase=True)
        p.changeVisualShape(self.end_position, -1, rgbaColor=[0, 1, 0, 0.3])
        p.changeVisualShape(self.start_position, -1, rgbaColor=[1, 0, 0, 0.3])
        self.border = p.loadURDF("assets/border.urdf", basePosition=[0.25, 0, 0.525], useFixedBase=True)

        index = 0

        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)
            # print("info=",info)
            jointName = info[1]
            jointType = info[2]
            if (jointType == p.JOINT_PRISMATIC):
                p.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
            if (jointType == p.JOINT_REVOLUTE):
                p.resetJointState(self.panda, j, jointPositions[index])
                index = index + 1
        self.t = 0.

    def reset(self):
        pass

    def update_state(self):
        keys = p.getKeyboardEvents()
        if len(keys) > 0:
            for k, v in keys.items():
                if v & p.KEY_WAS_TRIGGERED:
                    if k == ord('1'):
                        self.state = 1
                    if k == ord('2'):
                        self.state = 2
                    if k == ord('3'):
                        self.state = 3
                    if k == ord('4'):
                        self.state = 4
                    if k == ord('5'):
                        self.state = 5
                    if k == ord('6'):
                        self.state = 6
                if v & p.KEY_WAS_RELEASED:
                    self.state = 0

    def step(self):
        p.stepSimulation()
        self.update_state()
        time.sleep(1. / 240.)

        alpha = 0.9
        if self.state == 1 or self.state == 2 or self.state == 3 or self.state == 4 or self.state == 7:
            # gripper_height = 0.034
            self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.03
            if self.state == 2 or self.state == 3 or self.state == 7:
                self.gripper_height = alpha * self.gripper_height + (1. - alpha) * 0.2

            t = self.t
            self.t += self.control_dt
            pos = [self.offset[0] + 0.2 * math.sin(1.5 * t), self.offset[1] + self.gripper_height,
                   self.offset[2] + -0.6 + 0.1 * math.cos(1.5 * t)]
            if self.state == 3 or self.state == 4:
                pos, o = p.getBasePositionAndOrientation(self.lego)
                pos = [pos[0], self.gripper_height, pos[2]]
                self.prev_pos = pos
            if self.state == 7:
                pos = self.prev_pos
                diffX = pos[0] - self.offset[0]
                diffZ = pos[2] - (self.offset[2] - 0.6)
                self.prev_pos = [self.prev_pos[0] - diffX * 0.1, self.prev_pos[1], self.prev_pos[2] - diffZ * 0.1]

            orn = p.getQuaternionFromEuler([math.pi / 2., 0., 0.])
            p.submitProfileTiming("IK")
            jointPoses = p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll, ul,
                                                                       jr, rp, maxNumIterations=20)
            p.submitProfileTiming()
            for i in range(pandaNumDofs):
                p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, jointPoses[i],
                                                         force=5 * 240.)
            # target for fingers
        for i in [9, 10]:
            p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, self.finger_target,
                                                     force=10)
        p.submitProfileTiming()


env = Environment()
env.reset()
while (1):
    env.step()