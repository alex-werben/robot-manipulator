from typing import Any, Dict, Callable

import numpy as np

from gym_envs.envs.core import Task
from gym_envs.pybullet import PyBullet
from gym_envs.utils import distance


class PickPlaceAvoid(Task):
    def __init__(
        self,
        sim: PyBullet,
        reward_type: str = "sparse",
        get_ee_position: Callable = None,
        check_collision: Callable = None,
        distance_threshold: float = 0.05,
        goal_xy_range: float = 0.1,
        goal_z_range: float = 0.,
        obj_xy_range: float = 0.1,
    ) -> None:
        super().__init__(sim)
        self.get_ee_position = get_ee_position
        self.check_collision = check_collision
        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.object_size = 0.04
        self.goal_range_low = np.array([-goal_xy_range / 2, -goal_xy_range / 2, 0])
        self.goal_range_high = np.array([goal_xy_range / 2, goal_xy_range / 2, goal_z_range])
        self.obj_range_low = np.array([-obj_xy_range / 2, -obj_xy_range / 2, 0])
        self.obj_range_high = np.array([obj_xy_range / 2, obj_xy_range / 2, 0])
        with self.sim.no_rendering():
            self._create_scene()

    def _create_scene(self) -> None:
        """Create the scene."""
        self.sim.create_plane(z_offset=-0.4)
        self.sim.create_table(length=1.1, width=0.7, height=0.4, x_offset=-0.3)
        self.sim.create_box(
            body_name="object",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=1.0,
            position=np.array([0.0, -0.2, self.object_size / 2]),
            rgba_color=np.array([0.1, 0.9, 0.1, 1.0]),
        )
        self.obstacle_position = np.array([0.1, 0., 0.1])
        self.sim.create_cylinder(
            body_name="obstacle",
            radius=0.05,
            height=0.05,
            mass=1.0,
            position=self.obstacle_position,
            rgba_color=np.array([0.9, 0.1, 0.1, 1.0]),
        )
        self.sim.create_box(
            body_name="target",
            half_extents=np.ones(3) * self.object_size / 2,
            mass=0.0,
            ghost=True,
            position=np.array([0.0, 0.2, 0]),
            rgba_color=np.array([0.1, 0.9, 0.1, 0.3]),
        )

    def get_obs(self) -> np.ndarray:
        # position, rotation of the object
        object_position = self.sim.get_base_position("object")
        object_rotation = self.sim.get_base_rotation("object")
        object_velocity = self.sim.get_base_velocity("object")
        object_angular_velocity = self.sim.get_base_angular_velocity("object")
        observation = np.concatenate([object_position, object_rotation, object_velocity, object_angular_velocity])
        return observation

    def get_achieved_goal(self) -> np.ndarray:
        """Returns achieved goal. [0-2] - pos_obj, [3-5] - pos_tcp"""
        pos_obj = np.array(self.sim.get_base_position("object"))
        # pos_tcp = self.sim.get_link_position("panda", 11)
        # pos_tcp = self.get_ee_position()
        # pos_target = np.array(self.sim.get_base_position("target"))

        # achieved_goal = np.concatenate([pos_obj, pos_tcp])
        achieved_goal = np.array(pos_obj)
        return achieved_goal

    def get_desired_goal(self) -> np.ndarray:
        """Return the current goal."""
        if self.goal is None:
            raise RuntimeError("No goal yet, call reset() first")
        else:
            desired_goal = np.array(self.goal.copy())
            return desired_goal

    def reset(self) -> None:
        self.goal = self._sample_goal()
        object_position = self._sample_object()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("object", object_position, np.array([0.0, 0.0, 0.0, 1.0]))
        self.sim.set_base_pose("obstacle", self.obstacle_position, np.array([0.0, 0.0, 0.0, 1.0]))

    def _sample_goal(self) -> np.ndarray:
        """Sample a goal."""
        goal = np.array([0.0, 0.2, self.object_size / 2])  # z offset for the cube center
        noise = self.np_random.uniform(self.goal_range_low, self.goal_range_high)
        if self.np_random.random() < 0.3:
            noise[2] = 0.0
        goal += noise
        return goal

    def _sample_object(self) -> np.ndarray:
        """Randomize start position of object."""
        object_position = np.array([0.0, -0.2, self.object_size / 2])
        noise = self.np_random.uniform(self.obj_range_low, self.obj_range_high)
        object_position += noise
        return object_position

    def is_success(self, achieved_goal: np.ndarray, desired_goal: np.ndarray) -> np.ndarray:
        d = distance(achieved_goal, desired_goal)
        return np.array(d < self.distance_threshold, dtype=bool)

    def compute_reward(self, achieved_goal, desired_goal, info: Dict[str, Any]) -> np.ndarray:
        reward = 0.0
        try:
            pos_tcp = np.array([dd["pos_tcp"] for dd in info])
            pos_obstacle = np.array([dd["pos_obstacle"] for dd in info])
            action_gripper = np.array([dd["action_gripper"] for dd in info])
        except:
            pos_tcp = info["pos_tcp"]
            pos_obstacle = info["pos_obstacle"]
            action_gripper = info["action_gripper"]
        tcp_to_obj = distance(pos_tcp, achieved_goal)
        obj_to_target = distance(achieved_goal, desired_goal)

        reward = -tcp_to_obj -obj_to_target
        # reward for open tcp if it's far from obj
        # if tcp_to_obj > self.distance_threshold and action_gripper > 0:
        reward += 0.5 * np.count_nonzero((tcp_to_obj > self.distance_threshold) & (action_gripper > 0))
        # if tcp_to_obj < self.distance_threshold and action_gripper < 0:
        reward += 0.5 * np.count_nonzero((tcp_to_obj < self.distance_threshold) & (action_gripper < 0))

        # reward = -tcp_to_obj -obj_to_target
        # if self.reward_type == "sparse":
        #     return -np.array(d > self.distance_threshold, dtype=np.float32)
        # else:
        return reward.astype(np.float32)

