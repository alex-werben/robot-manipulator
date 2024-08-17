import math
import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import random
from typing import List





class PandaReachObjEnv(gym.Env):
    metadata = {"render_mode": ["human"], "render_fps": 30}

    def __init__(self, render_mode):
        self.MAX_EPISODE_LEN = 500

        self.panda_end_effector_index = 11
        super().__init__()
        self.object = None
        self.c1 = 10
        self.surface = None
        self.plane = None
        self.step_counter = 0
        self.threshold = 0.05
        self.observation = None
        if render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.reset()

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    def reset(self, seed=None, options=None):
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
        joint_positions = [0, 0, 0, -2.5, -0.30, 2.66, 2.32, 0.02, 0.02]
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
        state_object = np.array([0.5, 0.1, 0.05]).astype(np.float32)
        self.object = p.loadURDF("assets/block.urdf", basePosition=state_object)

        # Debug print axes
        p.addUserDebugText('X', [1, 0, 0], [0, 0, 0])
        p.addUserDebugText('Y', [0, 1, 0], [0, 0, 0])

        state_robot = np.array(p.getLinkState(self.panda, self.panda_end_effector_index)[0]).astype(np.float32)
        state_fingers = (p.getJointState(self.panda, 9)[0], p.getJointState(self.panda, 10)[0])
        observation = state_robot + state_object
        info = {}
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return observation, info

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        state_grasp_prev = p.getLinkState(self.panda, self.panda_end_effector_index)
        pos_prev = np.array(state_grasp_prev[0])

        # Execute action
        orientation = p.getQuaternionFromEuler([2 * math.pi / 2., 0, 0.])
        dv = 0.5
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        new_position = [pos_prev[0] + dx,
                        pos_prev[1] + dy,
                        pos_prev[2] + dz]
        # print(new_position)
        joint_poses = p.calculateInverseKinematics(self.panda, self.panda_end_effector_index, new_position,
                                                   orientation)[0:7]

        p.setJointMotorControlArray(self.panda, list(range(7)), p.POSITION_CONTROL,
                                    list(joint_poses))
        p.stepSimulation()

        # Calculate observation
        pos_obj_new = np.array(p.getBasePositionAndOrientation(self.object)[0])
        state_grasp = p.getLinkState(self.panda, self.panda_end_effector_index)
        pos_new = state_grasp[0]

        # reward
        r_g = -np.linalg.norm(pos_new - pos_obj_new)

        reward = self.c1 * r_g
        # result = [abs(pos_new[i] - pos_obj_new[i]) for i in range(len(pos_new))]
        # reward = -sum(result)

        terminated = False
        truncated = False

        if np.linalg.norm(pos_new - pos_obj_new) < 0.005:
            terminated = True
            truncated = True
        # End episode
        self.step_counter += 1
        if self.step_counter > self.MAX_EPISODE_LEN or reward > -0.09:
            terminated = True
            truncated = True

        observation = np.array(pos_obj_new + pos_new).astype(np.float32)
        info = {}
        return observation, reward, terminated, truncated, info

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
