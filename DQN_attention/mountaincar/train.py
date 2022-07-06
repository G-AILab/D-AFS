from env import mountain_car
from module import dqn
from module.dqn import mlp
import pandas as pd
import matplotlib.pyplot as plt


def main():
    env = mountain_car.MountainCarEnv()
    act = dqn.learn(
        env,
        network=mlp(num_hidden=64, num_layers=2),
        lr=1e-3,
        gamma=1,
        total_timesteps=200000,
        buffer_size=50000,
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        print_freq=10,
        checkpoint_freq=1000,
        target_network_update_freq=500,
        # seed=1234
    )
    print("Saving model to mountaincar_model.pkl")
    act.save("log/mountaincar.cpkt")

    plt.plot(env.episode_reward)
    plt.title("mountaincar reward")
    plt.xlabel('episode')
    plt.ylabel('mean 100 episode reward')
    plt.show()

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        attentions = []
        while not done:
            #env.render()
            attention_value = act(obs[None], stochastic=False)[1][0]
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0][0])
            episode_rew += rew
            attentions.append(attention_value)
        # data = pd.DataFrame(attentions, columns=['a1','a2','a3','a4'])
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
