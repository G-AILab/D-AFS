from env import pendulum_att
from model import dqn
from model.dqn import mlp

def main():
    env = pendulum_att.PendulumEnv()
    act = dqn.learn(
        env,
        network=mlp(num_hidden=64, num_layers=1),
        lr=1e-4,
        total_timesteps=200000,
        buffer_size=10000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        checkpoint_freq=2000,
    )
    print("Saving model to pendulum_model.pkl")
    act.save("log/pendulum.cpkt")

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