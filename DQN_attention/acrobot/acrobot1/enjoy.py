# from env import cartpole
from env import acrobot

from module import dqn
from module.dqn import mlp
# from env import SISO
import pandas as pd


def main():
    # env = cartpole.CartPoleEnv()
    env = acrobot.AcrobotEnv()
    act = dqn.learn(
        env,
        network=mlp(num_hidden=64, num_layers=1),
        total_timesteps=0,
        load_path="acrobot.cpkt"
    )

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        attentions = []
        step = 0
        while not done and step < 500:
#             env.render()
            attention_values = act(obs[None], stochastic=False)[1][0]
            obs, rew, done, _ = env.step(act(obs[None], stochastic=False)[0][0])
            episode_rew += rew
            attentions.append(attention_values)
            step+=1
        print("Episode reward", episode_rew)
        # data = pd.DataFrame(attentions, columns = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'a11', 'a12'])
        data = pd.DataFrame(attentions, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'])
        data.to_csv('attention.csv',index=False)
        m_a1 = data['a1'].mean()
        m_a2 = data['a2'].mean()
        m_a3 = data['a3'].mean()
        m_a4 = data['a4'].mean()
        m_a5 = data['a5'].mean()
        m_a6 = data['a6'].mean()
        m_a7 = data['a7'].mean()
        m_a8 = data['a8'].mean()
        m_a9 = data['a9'].mean()
        m_a10 = data['a10'].mean()
        # m_a11 = data['a11'].mean()
        # m_a12 = data['a12'].mean()
        # print(m_a1, m_a2, m_a3, m_a4, m_a5, m_a6, m_a7, m_a8, m_a9, m_a10, m_a11, m_a12)
        print(m_a1, m_a2, m_a3, m_a4, m_a5, m_a6, m_a7, m_a8, m_a9, m_a10)


if __name__ == '__main__':
    main()
