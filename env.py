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

use_null_space = 1
ik_solver = 0
panda_end_effector_index = 11
panda_num_dofs = 7

MODE = p.GUI  # p.GUI or p.DIRECT - with or without rendering
DIM_OBS = 8  # no. of dimensions in observation space
DIM_ACT = 4  # no. of dimensions in action space
MAX_EPISODE_LEN = 500


class Environment(gym.Env):
    def __init__(self):
        self.step_counter = 0
        self.observation = None
        p.connect(MODE)
        self.reset()

        self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        self.observation_space = spaces.Box(np.array([-1] * 5), np.array([1] * 5))

    def reset(self, seed=23):
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -10)
        p.setPhysicsEngineParameter(solverResidualThreshold=0)

        # Generate panda, plane, other objects
        orientation = p.getQuaternionFromEuler([0, 0, 0])
        # panda
        self.panda = p.loadURDF("franka_panda/panda.urdf",
                                useFixedBase=True,
                                basePosition=[0, 0, 0],
                                baseOrientation=orientation)
        joint_positions = [0, 0, 0, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]
        index = 0
        for j in range(p.getNumJoints(self.panda)):
            p.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.panda, j)
            joint_type = info[2]
            if joint_type == p.JOINT_PRISMATIC or joint_type == p.JOINT_REVOLUTE:
                p.resetJointState(self.panda, j, joint_positions[index])
                index = index + 1

        # plane
        self.plane = p.loadURDF("plane.urdf", basePosition=[0, 0, -0.5])


        # surface, other objects
        p.setAdditionalSearchPath(os.getcwd())
        self.surface = p.loadURDF("assets/cube.urdf", basePosition=[0.25, 0, -0.25], useFixedBase=True)
        # self.end_position = p.loadURDF("assets/position.urdf", basePosition=[0.6, 0.5, 0.025], useFixedBase=True)
        # self.start_position = p.loadURDF("assets/position.urdf", basePosition=[0.6, -0.5, 0.025], useFixedBase=True)
        # p.changeVisualShape(self.end_position, -1, rgbaColor=[0, 1, 0, 0.3])
        # p.changeVisualShape(self.start_position, -1, rgbaColor=[1, 0, 0, 0.3])
        # self.border = p.loadURDF("assets/border.urdf", basePosition=[0.25, 0, 0.025], useFixedBase=True)
        # state_object = [random.uniform(0.3, 0.6), random.uniform(-0.4, 0.4), 0.05]
        state_object = [0.4, 0.1, 0.05]
        self.object = p.loadURDF("assets/block.urdf", basePosition=state_object)

        # Debug print axes
        p.addUserDebugText('X', [1, 0, 0], [0, 0, 0])
        p.addUserDebugText('Y', [0, 1, 0], [0, 0, 0])

        state_robot = p.getLinkState(self.panda, panda_end_effector_index)[0]
        state_fingers = (p.getJointState(self.panda, 9)[0], p.getJointState(self.panda, 10)[0])
        observation = state_robot + state_fingers
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation

    def step(self, action: List):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        # Execute action
        orientation = p.getQuaternionFromEuler([2 * math.pi / 2., 0, 0.])
        dv = 0.005
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        current_position = p.getLinkState(self.panda, panda_end_effector_index)[0]
        new_position = [current_position[0] + dx,
                        current_position[1] + dy,
                        current_position[2] + dz]

        joint_poses = p.calculateInverseKinematics(self.panda, panda_end_effector_index, new_position,
                                                   orientation)[0:7]

        p.setJointMotorControlArray(self.panda, list(range(7)) + [9, 10], p.POSITION_CONTROL,
                                    list(joint_poses) + 2 * [fingers])
        p.stepSimulation()
        # time.sleep(1/ 240.)

        # Calculate reward and observation
        state_object = np.array(p.getBasePositionAndOrientation(self.object)[0])
        state_robot = p.getLinkState(self.panda, panda_end_effector_index)[0]
        state_fingers = (p.getJointState(self.panda, 9)[0], p.getJointState(self.panda, 10)[0])

        # reward
        done = False
        tip = state_robot
        obj = state_object
        result = [abs(tip[i] - obj[i]) for i in range(len(tip))]
        reward = -sum(result)
        if state_object[2] > 0.45:
            reward += 1
            done = True
        # End episode 
        self.step_counter += 1
        if self.step_counter > MAX_EPISODE_LEN or reward > -0.09:
            done = True

        # TODO: Detect Collisions of Panda and Obstacle
        # cost = 0
        # contacts = p.getContactPoints(self.panda, self.obstacle1)
        # if contacts:
        #     print("Collision detected! ", "[", self.cost_counter, "]", end="\r")
        #     self.cost_counter += 1
        #     cost = 1

        # print("REWARD: ",reward)
        # info = {'cost': cost}  #
        info = state_object

        self.observation = state_robot + state_fingers  # + state_robot

        return np.array(self.observation).astype(np.float32), reward, done, info

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
