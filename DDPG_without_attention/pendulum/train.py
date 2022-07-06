import tensorflow as tf
import numpy as np
import pandas as pd
import gym
from replay_buffer import ReplayBuffer
from actor import ActorNetwork
from critic import CriticNetwork
from ou_noise import OUNoise
from pendulum_att import PendulumEnv

# np.set_printoptions(precision=4)


# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Soft target update param
TAU = 0.001
# Gym environment
ENV_NAME = 'Pendulum-v0'
RANDOM_SEED = 1234
EXPLORE = 70
DEVICE = '/cpu:0'


def trainer(epochs=1000, MINIBATCH_SIZE=64, GAMMA=0.99, epsilon=1.0, min_epsilon=0.01, BUFFER_SIZE=10000,
            train_indicator=True, render=False):
    with tf.Session() as sess:

        # configuring environment
        # env = gym.make(ENV_NAME)
        env = PendulumEnv()
        # configuring the random processes
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)
        # info of the environment to pass to the agent
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = np.array([env.max_torque])
        # Creating agent
        ruido = OUNoise(action_dim, mu=0.4)  # this is the Ornstein-Uhlenbeck Noise
        actor = ActorNetwork(sess, state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU, DEVICE)
        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars(),
                               DEVICE)

        sess.run(tf.global_variables_initializer())

        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()
        # Initialize replay memory
        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

        try:
            critic.recover_critic()
            actor.recover_actor()
            print('********************************')
            print('models restored succesfully')
            print('********************************')
        except tf.errors.NotFoundError:
            print('********************************')
            print('Failed to restore models')
            print('********************************')

        save_reward = None
        five_reward = []
        all_states = []
        all_rewards = []

        var = 3

        for i in range(epochs):

            state = env.reset()
            state = np.hstack(state)
            ep_reward = 0
            ep_ave_max_q = 0
            step = 0
            max_step = 200
            epsilon -= (epsilon / EXPLORE)
            epsilon = np.maximum(min_epsilon, epsilon)
            # attention_weight = [[1, 1, 1, 1]]
            # weights = []
            # five_weights = []
            states = []
            rewards = []
            four_states = []
            actions = []

            for j in range(max_step):
                if render:
                    env.render()

                action_original = actor.predict(np.reshape(state, (1, state_dim)))
                # + (10. / (10. + i))* np.random.randn(1)
                # action = action_original + max(epsilon, 0) * ruido.noise()
                # action = action_original + (1. / (1. + i + j))
                action = np.clip(np.random.normal(action_original, np.max([var, 0.03])), -env.max_torque, env.max_torque)
                actions.append(action)

                # remove comment if you want to see a step by step update
                # print(step,'a',action_original, action,'s', state[0], 'max state', max_state_episode)

                # 2. take action, see next state and reward :
                next_state, reward, done, info = env.step(action[0])

                if train_indicator:
                    # 3. Save in replay buffer:
                    replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                                      done, np.reshape(next_state, (actor.s_dim,)))

                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > MINIBATCH_SIZE:

                        # 4. sample random minibatch of transitions:
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)
                        var *= .995

                        # Calculate targets

                        # 5. Train critic Network (states,actions, R + gamma* V(s', a')):
                        # 5.1 Get critic prediction = V(s', a')
                        # the a' is obtained using the actor prediction! or in other words : a' = actor(s')
                        target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                        # 5.2 get y_t where:
                        y_i = []
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + GAMMA * target_q[k])

                        # 5.3 Train Critic!
                        predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))
                        # weights.append(attention_weight[0])

                        ep_ave_max_q += np.amax(predicted_q_value)

                        # 6 Compute Critic gradient (depends on states and actions)
                        # 6.1 therefore I first need to calculate the actions the current actor would take.
                        a_outs = actor.predict(s_batch)
                        # 6.2 I calculate the gradients
                        grads = critic.action_gradients(s_batch, a_outs)
                        actor.train(s_batch, grads[0])

                        # Update target networks
                        actor.update_target_network()
                        critic.update_target_network()

                states.append(state)
                state = next_state
                rewards.append(reward)
                ep_reward = ep_reward + reward
                step += 1
            if step > (max_step - 1):
                five_reward.append(ep_reward)
#                 five_weights.append(weights)
#                 five_weights = np.array(five_weights).reshape((len(weights), 5))
#                 four_states.append(states)
#                 four_states = np.array(four_states).reshape((len(states), 5))
                actions = np.array(actions).reshape((len(actions), 1))
                all_states.append(states)
                all_rewards.append(rewards)

                mean_reward = np.mean(five_reward[-50:])
                if save_reward == None or mean_reward > save_reward:
                    save_reward = mean_reward
                    # critic.save_critic()
                    # actor.save_actor()
                    print('better five mean reward:', save_reward)
#                     pd.DataFrame(five_weights).to_csv('attention_weights.csv', header=['a1', 'a2', 'a3', 'a4', 'a5'], index=False)
#                     pd.DataFrame(four_states).to_csv('state.csv', header=['a1', 'a2', 'a3', 'a4', 'a5'], index=False)
                    # pd.DataFrame(actions).to_csv('action.csv', header=['action'], index=False)
                ruido.reset()

            print('th', i + 1, 'n steps', step, 'R:', np.round(ep_reward, 3))
            # if render:
            #     env.render()
        # all_states = np.array(all_states).reshape(40000, 6)
        # all_rewards = np.array(all_rewards).reshape(40000, 1)
        # pd.DataFrame(all_states).to_csv('all_states.csv', header=['a1', 'a2', 'a3', 'a4', 'a5','a6'], index=False)
        # pd.DataFrame(all_rewards).to_csv('all_rewards.csv', header=['rewards'], index=False)


if __name__ == '__main__':
    trainer(epochs=300, epsilon=1., render=False)
