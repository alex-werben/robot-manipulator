import multiprocessing
import os
import time

import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.callback import SaveOnBestTrainingRewardCallback
from src.utils import prepare_directory_for_results, prepare_model, make_env

import panda_gym
import gym_envs
# from panda_gym.envs import PandaReachEnv


def train(env_id: str = "PandaReachObjEnv-v0", model_name: str = "PPO"):
	num_cpu = multiprocessing.cpu_count()
	log_dir, model_dir = prepare_directory_for_results(os.getcwd(), env_id, model_name)

	env = SubprocVecEnv([make_env(env_id, i, log_dir) for i in range(num_cpu)])

	callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_dir=model_dir, verbose=1)

	model_cls = prepare_model(model_name)
	model = model_cls('MlpPolicy', env)

	model.learn(total_timesteps=int(1e5), callback=callback, progress_bar=True)
	model.save(model_dir + "/end_model.zip")


def test(env_id: str = "PandaReachObjEnv-v0", model_name: str = "PPO"):
	_, model_dir = prepare_directory_for_results(os.getcwd(), env_id, model_name)
	env = gym.make(env_id, render_mode="human")

	model_cls = prepare_model(model_name)
	model = model_cls('MlpPolicy', env)
	model.load(model_dir + "/end_model.zip")

	deterministic = True
	# evaluate_policy(
	# 	model,
	# 	env,
	# 	n_eval_episodes=10,
	# 	render=True,
	# 	deterministic=deterministic
	# )
	observations, state = env.reset()
	observations = observations
	states = None
	while True:
		actions, states = model.predict(
			observations,
			state=states,
			deterministic=deterministic,
		)
		new_observations, rewards, terminated, truncated, infos = env.step(actions)
		observations = new_observations


if __name__ == '__main__':
	# train("P-v2", "DDPG")
	# test("CartPole-v1", "A2C")
	# test("PandaReach-v2", "DDPG")
	# train("PandaReachObjEnv-v0", "PPO")
	test("PandaReachObjEnv-v0", "PPO")
