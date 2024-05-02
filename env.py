import math
import os
import time
import numpy as np
import gym
from gym import spaces
import pybullet as p
import pybullet_data
import random
from typing import List

useNullSpace = 1
ikSolver = 0
pandaEndEffectorIndex = 11
pandaNumDofs = 7
ll = [-7] * pandaNumDofs
#upper limits for null space (todo: set them to proper range)
ul = [7] * pandaNumDofs
#joint ranges for null space (todo: set them to proper range)
jr = [7] * pandaNumDofs
joint_positions = [0, 0, 0, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]

rp = joint_positions


class Environment(gym.Env):
    def __init__(self):
        p.connect(p.GUI)
        self.reset()

        # self.action_space = spaces.Box(low=np.array([0.1, -0.3, 0.5]),
        #                                high=np.array([0.7, 0.3, 0.9]))
        # self.observation_space = spaces.Box(low=np.array([0.1, -0.3, 0.5]),
        #                                     high=np.array([0.7, 0.3, 0.9]))

        self.action_space = spaces.Box(np.array([-1]*4), np.array([1]*4))
        self.observation_space = spaces.Box(np.array([-1]*5), np.array([1]*5))

    def reset(self, seed=23):
        p.resetSimulation()
        self.t = 0
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(solverResidualThreshold=0)

        orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.panda = p.loadURDF("franka_panda/panda.urdf",
                                useFixedBase=True,
                                basePosition=[0, 0, 0.5],
                                baseOrientation=orientation)
        self.plane = p.loadURDF("plane.urdf")
        # self.lego = p.loadURDF("lego/lego.urdf", basePosition=[0.5, 0, 0.6])
        self.gripper_height = 0.2 + 0.7
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
        index = 0

        self.surface = p.loadURDF("assets/cube.urdf", basePosition=[0.25, 0, 0.25], useFixedBase=True)
        self.end_position = p.loadURDF("assets/position.urdf", basePosition=[0.6, 0.5, 0.525], useFixedBase=True)
        self.start_position = p.loadURDF("assets/position.urdf", basePosition=[0.6, -0.5, 0.525], useFixedBase=True)
        p.changeVisualShape(self.end_position, -1, rgbaColor=[0, 1, 0, 0.3])
        p.changeVisualShape(self.start_position, -1, rgbaColor=[1, 0, 0, 0.3])
        self.border = p.loadURDF("assets/border.urdf", basePosition=[0.25, 0, 0.525], useFixedBase=True)
        state_object = [random.uniform(0.3, 0.6), random.uniform(-0.4, 0.4), 0.6]
        self.lego = p.loadURDF("assets/block.urdf", state_object)

        # Reset joint state
        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)
            joint_type = info[2]
            if joint_type == p.JOINT_PRISMATIC:
                p.resetJointState(self.panda, j, joint_positions[index])
                index = index + 1
            if joint_type == p.JOINT_REVOLUTE:
                p.resetJointState(self.panda, j, joint_positions[index])
                index = index + 1
        # Debug print axes
        p.addUserDebugText('X', [1, 0, 0.5], [0, 0, 0])
        p.addUserDebugText('Y', [0, 1, 0.5], [0, 0, 0])

        state_robot = p.getLinkState(self.panda, pandaEndEffectorIndex)[0]
        state_fingers = (p.getJointState(self.panda, 9)[0], p.getJointState(self.panda, 10)[0])
        observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation

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
                    if k == ord('7'):
                        self.state = 7
                    if k == ord('8'):
                        self.state = 8

                if v & p.KEY_WAS_RELEASED:
                    self.state = 0

    def step(self, action: List):
        """

        :param action:
        :return: observation, reward, done, info
        """
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        orientation = p.getQuaternionFromEuler([2 * math.pi / 2., 0, 0.])

        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        current_position = p.getLinkState(self.panda, pandaEndEffectorIndex)[0]

        new_position = [current_position[0] + dx,
                        current_position[1] + dy,
                        current_position[2] + dz]

        joint_poses = p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, new_position,
                                                   orientation, ll, ul, jr, rp, maxNumIterations=20)

        for i in range(pandaNumDofs):
            p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, joint_poses[i], force=120.)
        p.setJointMotorControlArray(self.panda, [9, 10], p.POSITION_CONTROL, [fingers] * 2)
        # p.setJointMotorControlArray(self.panda, list(range(7)) + [9, 10], p.POSITION_CONTROL,
        #                             list(joint_poses) + 2 * [fingers])


        p.stepSimulation()
        time.sleep(1 / 240.)

        state_object = np.array(p.getBasePositionAndOrientation(self.lego)[0])

        state_robot = p.getLinkState(self.panda, pandaEndEffectorIndex)[0]
        state_fingers = (p.getJointState(self.panda, 9)[0], p.getJointState(self.panda, 10)[0])
        if state_object[2] > 0.85:
            reward = 1
            done = True
        else:
            reward = 0
            done = False
        info = state_object
        observation = state_robot + state_fingers

        return observation, reward, done, info

    def step_manual(self):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
        self.update_state()
        # time.sleep(1. / 240.)
        alpha = 0.9

        if self.state != 0:
            if self.state == 5 or self.state == 6:
                finger_targets = {
                    5: 0.01,
                    6: 0.04
                }
                self.finger_target = finger_targets[self.state]
            elif self.state == 1 or self.state == 2 or self.state == 3 or self.state == 4 or self.state == 7 or self.state == 8:
                if self.state == 1:
                    pos = [0.5, 0, 0.8]
                if self.state == 2 or self.state == 3 or self.state == 4:
                    indexes = {
                        2: self.start_position,
                        3: self.end_position,
                        4: self.lego
                    }
                    pos, o = p.getBasePositionAndOrientation(indexes[self.state])
                    current_height = p.getLinkState(self.panda, pandaEndEffectorIndex)[0][2]
                    pos = [pos[0], pos[1], current_height]
                if self.state == 7 or self.state == 8:
                    z_pos = {
                        7: 0.5,
                        8: 0.8
                    }
                    pos = p.getLinkState(self.panda, pandaEndEffectorIndex)[0]
                    pos = [pos[0], pos[1], z_pos[self.state]]

                orn = p.getQuaternionFromEuler([2 * math.pi / 2., 0, 0.])
                jointPoses = p.calculateInverseKinematics(self.panda, pandaEndEffectorIndex, pos, orn, ll, ul,
                                                          jr, rp, maxNumIterations=20)
                self.prev_pos = pos
                for i in range(pandaNumDofs):
                    p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, jointPoses[i], force=120.)
        for i in [9, 10]:
            p.setJointMotorControl2(self.panda, i, p.POSITION_CONTROL, self.finger_target, force=1000)

        p.stepSimulation()
        self.t += 1

        pos_goal, _ = p.getBasePositionAndOrientation(self.lego)
        pos_tsp = p.getLinkState(self.panda, pandaEndEffectorIndex)[0]

        return self.t, pos_goal, pos_tsp

    def render(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0.7, 0, 0.8],
                                                          distance=.7,
                                                          yaw=90,
                                                          pitch=-70,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(960) / 720,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960,
                                            height=720,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def close(self):
        p.disconnect()

# env = Environment()
# env.reset()
# # while (1):
# for _ in range(100000):
#     # env.render()
#     action = env.action_space.sample()
#     print(action)
#     env.step(action)
#     # env.step(1)
# env.close()
