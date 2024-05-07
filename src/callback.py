from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.callbacks import BaseCallback
import os
import numpy as np


class SaveOnBestTrainingRewardCallback(BaseCallback):
	"""
	Callback for saving a model (the check is done every ``check_freq`` steps)
	based on the training reward (in practice, we recommend using ``EvalCallback``).

	:param check_freq: (int)
	:param log_dir: (str) Path to the folder where the model will be saved.
		It must contain the file created by the ``Monitor`` wrapper.
	:param verbose: (int)
	"""

	def __init__(self, check_freq: int, log_dir: str, model_dir: str, verbose=1):
		super().__init__(verbose)
		self.check_freq = check_freq
		self.log_dir = log_dir
		self.model_dir = model_dir
		self.best_mean_reward = -np.inf
		self.save_path = os.path.join(self.model_dir, 'best_model.zip')

	def _on_step(self) -> bool:
		if self.n_calls % self.check_freq == 0:

			# Retrieve training reward
			x, y = ts2xy(load_results(self.log_dir), "timesteps")
			if len(x) > 0:
				# Mean training reward over the last 100 episodes
				mean_reward = np.mean(y[-100:])

				# New best model, you could save the agent here
				if mean_reward > self.best_mean_reward:
					self.best_mean_reward = mean_reward
					# Example for saving best model
					if self.verbose > 0:
						print(f"Saving new best model to {self.save_path}, current timestep: {self.n_calls * 12}")
					self.model.save(self.save_path)

		return True
