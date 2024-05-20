import multiprocessing
import os

import torch
import yaml

import gymnasium as gym
from stable_baselines3 import PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.arguments import get_args
from src.callback import SaveOnBestTrainingRewardCallback
from src.utils import prepare_directory_for_results, prepare_model, make_env

# import panda_gym
import gym_envs


def train(env_id: str = "PandaReach-v3",
          model_name: str = "DDPG",
          train_from_scratch: bool = True,
          model_to_load_path: str = None,
          params: dict = None):
    log_dir, model_dir = prepare_directory_for_results(os.getcwd(), env_id, model_name, params['build_name'])

    # Multiprocessing
    num_cpu = multiprocessing.cpu_count()
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    # Vectorized environments
    env = SubprocVecEnv([make_env(env_id, i, log_dir, train_from_scratch) for i in range(num_cpu)])
    # env = gym.make(env_id, reward_type="dense")
    # Callback to save best model during learning
    save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000,
                                                     log_dir=log_dir,
                                                     model_dir=model_dir,
                                                     verbose=1)
    # Prepare model classes
    model_cls = prepare_model(params['model_name'])

    replay_buffer_class = prepare_model(params['replay_buffer_class']) if 'replay_buffer_class' in params else None
    # Train from scratch or keep training with existing model
    # print(train_from_scratch)
    if train_from_scratch:
        model = model_cls(env=env,
                          tensorboard_log=log_dir,
                          device=device,
                          replay_buffer_class=replay_buffer_class,
                          **params['model_params'])
        reset_num_timesteps = True
    else:
        print("loading model")
        model = model_cls.load(model_dir + model_to_load_path, env=env, device=device)
        model.load_replay_buffer(model_dir + "/end_replay_buffer")
        reset_num_timesteps = False

    model.learn(**params['learn_params'],
                callback=save_callback,
                progress_bar=True,
                reset_num_timesteps=reset_num_timesteps)
    model.save(model_dir + "/end_model.zip")
    model.save_replay_buffer(model_dir + "/end_replay_buffer.pkl")


def test(env_id: str = "PandaReachObjEnv-v0", model_name: str = "PPO", params: dict = None):
    log_dir, model_dir = prepare_directory_for_results(os.getcwd(), env_id, model_name, params['build_name'])

    env = gym.make(env_id, render_mode="human", reward_type="dense")
    model_cls = prepare_model(model_name)
    model = model_cls.load(model_dir + "/best_model.zip", env=env)
    model.load_replay_buffer(model_dir + "/replay_buffer")
    deterministic = True
    evaluate_policy(
        model,
        env,
        n_eval_episodes=100,
        render=True,
        deterministic=deterministic
    )


def test_env(env_id: str = "PandaReachObjEnv-v0", model_name: str = "PPO"):
    env = gym.make(env_id, render_mode="human")
    model_cls = prepare_model(model_name)

    model = model_cls('MultiInputPolicy', env=env)
    # model.load_replay_buffer(model_dir + "/replay_buffer_504000.pkl")
    deterministic = False
    evaluate_policy(
        model,
        env,
        n_eval_episodes=100,
        render=True,
        deterministic=deterministic
    )


if __name__ == '__main__':
    args = get_args()
    config_path = args.config
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if args.mode == "train":
        train(env_id=args.env,
              model_name=args.model,
              train_from_scratch=True,
              model_to_load_path="/end_model",
              params=config)
    else:
        test(env_id=args.env,
             model_name=args.model,
             params=config)
