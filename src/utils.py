import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, DDPG, A2C, SAC, HerReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm


def prepare_model(model_name: str):
	model_dict = {
		'PPO': PPO,
		'DDPG': DDPG,
		'A2C': A2C,
		'SAC': SAC,
		'HerReplayBuffer': HerReplayBuffer,
	}

	return model_dict[model_name]


def make_env(env_id: str, rank: int, log_dir: str, train_from_scratch: bool = True, seed: int = 0):
	def _init() -> gym.Env:
		env = gym.make(env_id, reward_type="dense")
		env = Monitor(env, f"{log_dir}/{rank}", override_existing=train_from_scratch)
		# env = Monitor(env)
		return env

	set_random_seed(seed)
	return _init


def prepare_directory_for_results(path: str, env_id: str, model_name: str, build_name: str) -> tuple[str, str]:
	results_path = os.path.join(path, "results", env_id, model_name)
	log_dir = results_path
	model_dir = os.path.join(results_path, build_name)
	os.makedirs(log_dir, exist_ok=True)

	return log_dir, model_dir
