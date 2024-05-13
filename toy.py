import gymnasium as gym
import gym_envs

def main(env_id: str = "PandaStack-v0"):
    env = gym.make(env_id, render_mode="human", reward_type="dense")
    env.reset()
    # action = env.action_space.sample()
    print(env.action_space.shape)

    action = [0, 0, 0, 0]
    a = [-0.5, -0.1, 0, 0.1, 0.5]
    for i in range(1000):
        print(action)
        env.step(action)
        action[-1] = a[0]


if __name__ == '__main__':
    main(env_id="PandaPickPlaceAvoid-v3")
