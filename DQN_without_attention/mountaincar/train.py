from env import mountain_car
from module import dqn
from module.dqn import mlp

def main():
    env = mountain_car.MountainCarEnv()
    act = dqn.learn(
        env,
        network=mlp(num_hidden=64, num_layers=2),
        lr=1e-3,
        total_timesteps=150000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        checkpoint_freq=2000,
        seed=1234
    )
    print("Saving model to mountaincar.pkl")
    act.save("log/mountaincar.cpkt")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
#             env.render()
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0])
            episode_rew += rew
            #done = env.render()
        print("Episode reward", episode_rew)

if __name__ == '__main__':
    main()