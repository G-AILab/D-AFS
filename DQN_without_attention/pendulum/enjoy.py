from env import pendulum_att
# from env import SISO

from model import dqn
from model.dqn import mlp

def main():
    env = pendulum_att.PendulumEnv()
    act = dqn.learn(
        env,
        network=mlp(num_hidden=48, num_layers=1),
        total_timesteps=0,
        load_path="pendulum.cpkt"
    )


    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            # env.render()
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0])
            episode_rew += rew
            #done = env.render()
        print("Episode reward", episode_rew)

if __name__ == '__main__':
    main()
