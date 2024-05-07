import os
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3 import PPO, DDPG, HER, A2C


def prepare_model(model_name: str):
	model_dict = {
		'PPO': PPO,
		'DDPG': DDPG,
		'HER': HER,
		'A2C': A2C,
	}

	return model_dict[model_name]


def make_env(env_id: str, rank: int, log_dir: str, seed: int = 0):
	def _init() -> gym.Env:
		env = gym.make(env_id)
		env = Monitor(env, f"{log_dir}/{rank}")
		return env

	set_random_seed(seed)
	return _init


def prepare_directory_for_results(path: str, env_id: str, model_name) -> tuple[str, str]:
	results_path = os.path.join(path, "results", env_id, model_name)
	log_dir = os.path.join(results_path, "metrics")
	model_dir = os.path.join(results_path, "models")
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(model_dir, exist_ok=True)

	return log_dir, model_dir
