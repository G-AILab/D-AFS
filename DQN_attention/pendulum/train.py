# from env import cartpole
from env import pendulum_att
from module import dqn
from module.dqn import mlp


def main():
    env = pendulum_att.PendulumEnv()
    act = dqn.learn(
        env,
        network=mlp(num_hidden=48, num_layers=1),
        lr=1e-3,
        total_timesteps=200000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        checkpoint_freq=1000
    )
    print("Saving model to pendulum_model.pkl")
    act.save("log/pendulum.cpkt")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            # env.render()
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0])
            episode_rew += rew

        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()