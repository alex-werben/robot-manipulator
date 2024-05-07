import pybullet as p
from gymnasium.envs.registration import register

register(
     id="PandaReachObjEnv-v0",
     entry_point="gym_envs.envs:PandaReachObjEnv",
     kwargs={
          'render_mode': "rgb_array"
     }
)
