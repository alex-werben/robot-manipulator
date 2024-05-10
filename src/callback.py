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
        self.best_mean_reward = -np.inf
        self.model_dir = model_dir
        self.save_path = os.path.join(self.model_dir, 'best_model')

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
                    self.model.save(self.save_path + f"_{self.n_calls * 12}.zip")

        return True

from stable_baselines3.common.logger import HParam


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """
    def __init__(self):
        super().__init__()

    def _on_training_start(self) -> None:
        self.hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        self.metric_dict = {
            "rollout/ep_len_mean": 0,
        }
        self.logger.record(
            "hparams",
            HParam(self.hparam_dict, self.metric_dict),
            # exclude=("log", "json", "csv"),
        )
        # logs_path = os.path.join(self.logdir, self.model.__class__.__name__ + "_0")
        # # save these hyperparameters as logs
        # with tf.summary.create_file_writer(logs_path).as_default():
        #     hp.hparams(hparams)
        # self._done = True

    def _on_step(self) -> bool:
        self.logger.record(
            "hparams",
            HParam(self.hparam_dict, self.metric_dict),
            # exclude=("log", "json", "csv"),
        )
        return True

