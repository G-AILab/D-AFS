from module import dqn
from module.dqn import mlp
#from env import SISO
from env import cartpole
import pandas as pd
import matplotlib.pyplot as plt

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
        attentions = []
        step = 0
        """
    while not done:
        attention_values = act(obs[None], stochastic=False)[1][0]
        obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0][0])
        episode_rew += rew
        done = env.render()
        attentions.append(attention_values)
    print("Episode reward", episode_rew)
    pd.DataFrame(attentions).to_csv('attention.csv', header=['a1', 'a2', 'a3'], index=False)
        """
        while not done and step < 400:
            step += 1
            # env.render()
            attention_value = act(obs[None], stochastic=False)[1][0]
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0][0])
            episode_rew += rew
            attentions.append(attention_value)
        data = pd.DataFrame(attentions, columns=['a1', 'a2', 'a3', 'a4','a5', 'a6', 'a7', 'a8'])
        mean_a1 = data['a1'].mean()
        mean_a2 = data['a2'].mean()
        mean_a3 = data['a3'].mean()
        mean_a4 = data['a4'].mean()
        mean_a5 = data['a5'].mean()
        mean_a6 = data['a6'].mean()
        mean_a7 = data['a7'].mean()
        mean_a8 = data['a8'].mean()
        # mean_a9 = data['a9'].mean()
        # mean_a10 = data['a10'].mean()
        data.to_csv('weight.csv', index=False)
        """
        mean_a1 = data['a1'].mean()
        mean_a2 = data['a2'].mean()
        mean_a3 = data['a3'].mean()
        mean_a4 = data['a4'].mean()
        mean_a5 = data['a5'].mean()
        #mean_a6 = data['a6'].mean()
        plt.plot(data)
        plt.hlines(mean_a1, 0, len(data['a1']), colors="blue", linestyles="--")
        plt.hlines(mean_a2, 0, len(data['a1']), colors="orange", linestyles="--")
        plt.hlines(mean_a3, 0, len(data['a1']), colors="green", linestyles="--")
        plt.hlines(mean_a4, 0, len(data['a1']), colors="red", linestyles="--")
        plt.hlines(mean_a5, 0, len(data['a1']), colors="black", linestyles="--")
        #plt.hlines(mean_a6, 0, len(data['a1']), colors="yellow", linestyles="--")
        plt.legend(['a1', 'a2', 'a3', 'a4','a5','m_a1', 'm_a2', 'm_a3', 'm_a4','m_a5'])
        """
        plt.show()
        print("Episode reward", episode_rew)
        print([mean_a1,mean_a2,mean_a3,mean_a4,mean_a5,mean_a6,mean_a7,mean_a8])
if __name__ == '__main__':
    main()
