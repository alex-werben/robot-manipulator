from typing import Optional, Tuple, Dict, Any

import numpy as np
from gymnasium.utils import seeding

from gym_envs.envs.core import RobotTaskEnv
from gym_envs.envs.robots.panda import Panda
from gym_envs.envs.tasks.grasp import Grasp
from gym_envs.envs.tasks.grasp_avoid_reach import GraspAvoidReach
from gym_envs.envs.tasks.pick_place_avoid import PickPlaceAvoid
from gym_envs.pybullet import PyBullet


class PandaPickPlaceAvoidEnv(RobotTaskEnv):
	"""Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """

	def __init__(
		self,
		render_mode: str = "rgb_array",
		reward_type: str = "sparse",
		control_type: str = "ee",
		renderer: str = "Tiny",
		render_width: int = 720,
		render_height: int = 480,
		render_target_position: Optional[np.ndarray] = None,
		render_distance: float = 1.4,
		render_yaw: float = 45,
		render_pitch: float = -30,
		render_roll: float = 0,
	) -> None:
		sim = PyBullet(render_mode=render_mode, renderer=renderer)
		robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
		task = PickPlaceAvoid(sim,
							  reward_type=reward_type,
							  get_ee_position=robot.get_ee_position)
		super().__init__(
			robot,
			task,
			render_width=render_width,
			render_height=render_height,
			render_target_position=render_target_position,
			render_distance=render_distance,
			render_yaw=render_yaw,
			render_pitch=render_pitch,
			render_roll=render_roll,
		)


class PandaGraspEnv(RobotTaskEnv):
	"""Pick and Place task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        renderer (str, optional): Renderer, either "Tiny" or OpenGL". Defaults to "Tiny" if render mode is "human"
            and "OpenGL" if render mode is "rgb_array". Only "OpenGL" is available for human render mode.
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
        render_target_position (np.ndarray, optional): Camera targetting this postion, as (x, y, z).
            Defaults to [0., 0., 0.].
        render_distance (float, optional): Distance of the camera. Defaults to 1.4.
        render_yaw (float, optional): Yaw of the camera. Defaults to 45.
        render_pitch (float, optional): Pitch of the camera. Defaults to -30.
        render_roll (int, optional): Rool of the camera. Defaults to 0.
    """

	def __init__(
		self,
		render_mode: str = "rgb_array",
		reward_type: str = "sparse",
		control_type: str = "ee",
		renderer: str = "Tiny",
		render_width: int = 720,
		render_height: int = 480,
		render_target_position: Optional[np.ndarray] = None,
		render_distance: float = 1.4,
		render_yaw: float = 45,
		render_pitch: float = -30,
		render_roll: float = 0,
	) -> None:
		sim = PyBullet(render_mode=render_mode, renderer=renderer)
		robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
		task = Grasp(sim,
					 reward_type=reward_type,
					 robot=robot)
		super().__init__(
			robot,
			task,
			render_width=render_width,
			render_height=render_height,
			render_target_position=render_target_position,
			render_distance=render_distance,
			render_yaw=render_yaw,
			render_pitch=render_pitch,
			render_roll=render_roll,
		)

	def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
		self.robot.set_action(action)
		self.sim.step()
		# obstacle_name = "obstacle"
		observation = self._get_obs()

		# An episode is terminated iff the agent has reached the target
		# terminated = bool(self.task.is_success(observation["achieved_goal"], self.task.get_desired_goal()))
		terminated = bool(self.task.is_success(observation['achieved_goal'], observation['desired_goal']))
		truncated = False
		info = {"is_success": terminated,
				"pos_obj": self.sim.get_base_position("object"),
				"pos_tcp": self.robot.get_ee_position(),
				"grasp": action[-1],
				"collisions": self.check_collision(self.robot.body_name, "object")}
		reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_desired_goal(), info))

		return observation, reward, terminated, truncated, info

	def reset(
		self, seed: Optional[int] = None, options: Optional[dict] = None
	) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
		super().reset(seed=seed, options=options)
		self.grasp = False
		self.task.np_random, seed = seeding.np_random(seed)
		with self.sim.no_rendering():
			self.robot.reset()
			self.task.reset()
		observation = self._get_obs()
		info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_desired_goal()),
				}
		return observation, info


class PandaGraspAvoidReachEnv(RobotTaskEnv):
	def __init__(
		self,
		render_mode: str = "rgb_array",
		reward_type: str = "sparse",
		control_type: str = "ee",
		renderer: str = "Tiny",
		render_width: int = 720,
		render_height: int = 480,
		render_target_position: Optional[np.ndarray] = None,
		render_distance: float = 1.4,
		render_yaw: float = 45,
		render_pitch: float = -30,
		render_roll: float = 0,
	) -> None:
		sim = PyBullet(render_mode=render_mode, renderer=renderer)
		robot = Panda(sim, block_gripper=False, base_position=np.array([-0.6, 0.0, 0.0]), control_type=control_type)
		task = GraspAvoidReach(sim,
							   reward_type=reward_type,
							   robot=robot)
		self.grasp = False
		super().__init__(
			robot,
			task,
			render_width=render_width,
			render_height=render_height,
			render_target_position=render_target_position,
			render_distance=render_distance,
			render_yaw=render_yaw,
			render_pitch=render_pitch,
			render_roll=render_roll,
		)

	def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
		ee_position = self.robot.get_ee_position()
		self.robot.set_action(action)
		self.sim.step()
		observation = self._get_obs()

		# An episode is terminated iff the agent has reached the target
		terminated = bool(self.task.is_success(observation['achieved_goal'], observation['desired_goal']))
		truncated = False
		info = {"is_success": terminated,
				"pos_tcp": self.robot.get_ee_position(),
				"grasp": action[-1],}
		reward = float(self.task.compute_reward(observation["achieved_goal"], self.task.get_desired_goal(), info))

		return observation, reward, terminated, truncated, info

	def reset(
		self, seed: Optional[int] = None, options: Optional[dict] = None
	) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
		super().reset(seed=seed, options=options)
		self.grasp = False
		self.task.np_random, seed = seeding.np_random(seed)
		with self.sim.no_rendering():
			self.robot.reset()
			self.task.reset()
		observation = self._get_obs()
		info = {"is_success": self.task.is_success(observation["achieved_goal"], self.task.get_desired_goal()),}
		return observation, info
