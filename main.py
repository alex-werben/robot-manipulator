import multiprocessing
import os
import time

import gymnasium as gym
from stable_baselines3 import PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.callback import SaveOnBestTrainingRewardCallback, HParamCallback
from src.utils import prepare_directory_for_results, prepare_model, make_env

import panda_gym
import gym_envs


def train_her(env_id: str = "PandaReachObjEnv-v0", model_name: str = "PPO",
			  build_name: str = "build",
			  train_from_scratch: bool = True,
			  total_timesteps: int = 1_000_000):
	num_cpu = multiprocessing.cpu_count()
	log_dir, model_dir = prepare_directory_for_results(os.getcwd(), env_id, model_name, build_name)

	# model_cls = prepare_model(model_name)
	callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_dir=model_dir, verbose=1)
	env = SubprocVecEnv([make_env(env_id, i, log_dir, train_from_scratch) for i in range(num_cpu)])

	model = DDPG(
		policy='MultiInputPolicy',
		env=env,
		batch_size=2048,
		buffer_size=1_000_000,
		tau=0.05,
		gamma=0.95,
		learning_rate=0.001,
		learning_starts=1000,
		replay_buffer_class=HerReplayBuffer,
		replay_buffer_kwargs=dict(
			n_sampled_goal=4,
			goal_selection_strategy="future",
			# online_sampling=True,
		),
		policy_kwargs=dict(
			net_arch=[512, 512, 512],
			n_critics=2
		),
		tensorboard_log=log_dir
	)
	model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True, tb_log_name=build_name)
	model.save(model_dir + "/end_model.zip")


def test_her(env_id: str = "PandaReachObjEnv-v0", model_name: str = "PPO"):
	_, model_dir = prepare_directory_for_results(os.getcwd(), env_id, model_name)

	env = gym.make(env_id, render_mode="human")
	model_cls = prepare_model(model_name)
	model = model_cls.load(model_dir + "/end_model.zip", env=env)
	# model = PPO.load(model_dir + "/best_model.zip", env=env)
	deterministic = True
	evaluate_policy(
		model,
		env,
		n_eval_episodes=100,
		render=True,
		deterministic=deterministic
	)


def train(env_id: str = "PandaReachObjEnv-v0", model_name: str = "PPO",
		  build_name: str = "my_build",
		  train_from_scratch: bool = True,
		  total_timesteps: int = 1_000_000):
	num_cpu = multiprocessing.cpu_count()
	log_dir, model_dir = prepare_directory_for_results(os.getcwd(), env_id, model_name, build_name)

	env = SubprocVecEnv([make_env(env_id, i, log_dir, train_from_scratch) for i in range(num_cpu)])

	callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir, model_dir=model_dir, verbose=1)
	model_cls = prepare_model(model_name)
	if train_from_scratch:
		# model = model_cls('MultiInputPolicy', env)
		model = model_cls(
			'MultiInputPolicy',
			env,
			batch_size=2048,
			learning_starts=1000,
			gamma=0.95,
			tau=0.05,
			verbose=0,
			tensorboard_log=log_dir
		)
		model.learn(total_timesteps=total_timesteps, callback=HParamCallback(),
					progress_bar=True, tb_log_name=build_name)
	else:
		model = model_cls.load(log_dir + "/end_model.zip", env=env)
		model.learn(total_timesteps=total_timesteps, callback=callback,
					progress_bar=True, tb_log_name="test_run",
					reset_num_timesteps=False)
	model.save(model_dir + "/end_model.zip")


def test(env_id: str = "PandaReachObjEnv-v0", model_name: str = "PPO"):
	_ = prepare_directory_for_results(os.getcwd(), env_id, model_name, "aa")

	env = gym.make("PandaSlide-v3", render_mode="human")
	model_cls = prepare_model(model_name)
	# model = model_cls.load(model_dir + "/best_model.zip")
	model = model_cls('MultiInputPolicy', env)
	deterministic = False
	evaluate_policy(
		model,
		env,
		n_eval_episodes=100,
		render=True,
		deterministic=deterministic
	)

if __name__ == '__main__':
	train_her("PandaSlide-v3", "DDPG", build_name="first_build", train_from_scratch=True)
	# test("PandaSlide-v3")
	# train("PandaReach-v3", model_name="DDPG", train_from_scratch=True, total_timesteps=20_000)
# train("PandaPush-v3", model_name="PPO", train_from_scratch=True, total_timesteps=1_000_000)
# test_her("PandaPush-v3", "DDPG")
