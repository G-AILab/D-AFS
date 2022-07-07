from env import mountain_car

from module import dqn
from module.dqn import mlp

def main():
    env = mountain_car.MountainCarEnv()
    act = dqn.learn(
        env,
        network=mlp(num_hidden=64, num_layers=2),
        total_timesteps=0,
        load_path="mountaincar.cpkt"
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
