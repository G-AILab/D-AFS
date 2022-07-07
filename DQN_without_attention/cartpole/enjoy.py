from env import cartpole
# from env import SISO

from model import dqn
from model.dqn import mlp

def main():
    env = cartpole.CartPoleEnv()
    act = dqn.learn(
        env,
        network=mlp(num_hidden=12, num_layers=1),
        total_timesteps=0,
        load_path="cartpole.cpkt"
    )


    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            env.render()
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0])
            episode_rew += rew
            #done = env.render()
        print("Episode reward", episode_rew)

if __name__ == '__main__':
    main()
