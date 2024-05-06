import gymnasium as gym
import panda_gym
import pandas as pd
from stable_baselines3 import PPO
# import stable_baselines3
# from envs.env_reach_obj_gymnasium import Environment
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv
import os
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import multiprocessing

from envs.env_reach_obj_gymnasium import Environment

num_cpu = multiprocessing.cpu_count()


class SaveOnBestTrainingRewardCallback(BaseCallback):
	"""
	Callback for saving a model (the check is done every ``check_freq`` steps)
	based on the training reward (in practice, we recommend using ``EvalCallback``).

	:param check_freq: (int)
	:param log_dir: (str) Path to the folder where the model will be saved.
		It must contain the file created by the ``Monitor`` wrapper.
	:param verbose: (int)
	"""

	def __init__(self, check_freq: int, log_dir: str, verbose=1):
		super().__init__(verbose)
		self.check_freq = check_freq
		self.log_dir = log_dir
		self.save_path = os.path.join(log_dir, "best_model")
		self.best_mean_reward = -np.inf

	def _on_step(self) -> bool:
		if self.n_calls % self.check_freq == 0:

			# Retrieve training reward
			x, y = ts2xy(load_results(self.log_dir), "timesteps")
			if len(x) > 0:
				# Mean training reward over the last 100 episodes
				mean_reward = np.mean(y[-100:])

				self.df.loc[len(self.df)] = [self.n_calls, mean_reward]
				self.df.to_csv(self.log_dir + "metrics.csv", index=False)

				if self.verbose > 0:
					print(f"Num timesteps: {self.num_timesteps}")
					print(
						f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
					)

				# New best model, you could save the agent here
				if mean_reward > self.best_mean_reward:
					self.best_mean_reward = mean_reward
					# Example for saving best model
					if self.verbose > 0:
						print(f"Saving new best model to {self.save_path}.zip")
					self.model.save(self.save_path)

		return True


def make_env(env_id: str, rank: int, seed: int = 0):
	def _init() -> gym.Env:
		# env = gym.make(env_id)
		env = Environment(p.DIRECT)
		# env.reset(seed=seed + rank)
		env = Monitor(env, os.getcwd() + f"/metrics/env_reach_obj_gymnasium/{rank}")
		return env

	set_random_seed(seed)
	return _init


def main():
	# Create log dir
	log_dir = os.getcwd() + "/metrics/panda_env/"
	os.makedirs(log_dir, exist_ok=True)

	env = SubprocVecEnv(
		[make_env("PandaReach-v3", i) for i in range(num_cpu)])
	# Create RL model
	model = PPO('MlpPolicy', env, verbose=0)
	# Train the agent
	model.learn(total_timesteps=int(1e6), callback=None, progress_bar=True)
	model.save(log_dir + "model.zip")

def test():
	# pass
	log_dir = os.getcwd() + "/metrics/panda_env/"
	# env = gym.make("PandaReach-v3", render_mode='human')
	env = Environment(p.GUI)
	model = PPO('MlpPolicy', env=env)
	model.load(log_dir + "model.zip")
	obs, info = env.reset()

	done = False

	while not done:
		action, _state = model.predict(obs, deterministic=True)
		print(action)

		obs, reward, done, truncated, info = env.step(action)


if __name__ == '__main__':
	test()
