from logging import critical
from os import stat
from numpy import random
import tensorflow as tf
import numpy as np
import gym
from replay_buffer import ReplayBuffer
from ou_noise import OUNoise
from actor import ActorNetwork
from critic import CriticNetwork
from cartpole import CartPoleEnv


# ==========================
#   Training Parameters
# ==========================
# Maximum episodes run
MAX_EPISODES = 1000
# Max episode length
MAX_EP_STEPS = 1000
# Episodes with noise
NOISE_MAX_EP = 200

# Base learning rate for the Actor network
ACTOR_LEARNING_RATE = 0.0001
# Base learning rate for the Critic Network
CRITIC_LEARNING_RATE = 0.001
# Discount factor
GAMMA = 0.99
# Soft target update param
TAU = 0.001
EXPLORE = 70

# ===========================
#   Utility Parameters
# ===========================
# Render gym env during training
# RENDER_ENV = False
# Use Gym Monitor
# GYM_MONITOR_EN = False
# Gym environment
# ENV_NAME = 'CartPole-v0' # Discrete: Reward factor = 0.1
#ENV_NAME = 'CartPole-v1' # Discrete: Reward factor = 0.1
#ENV_NAME = 'Pendulum-v0' # Continuous: Reward factor = 0.01
# Directory for storing gym results
# MONITOR_DIR = './results/' + ENV_NAME
# Directory for storing tensorboard summary results
# SUMMARY_DIR = './results/tf_ddpg'
RANDOM_SEED = 1234
# Size of replay buffer

# ===========================
#   Tensorflow Summary Ops
# ===========================
# def build_summaries():
#     episode_reward = tf.Variable(0.)
#     tf.summary.scalar("Reward", episode_reward)
#     episode_ave_max_q = tf.Variable(0.)
#     tf.summary.scalar("Qmax Value", episode_ave_max_q)

#     summary_vars = [episode_reward, episode_ave_max_q]
#     summary_ops = tf.summary.merge_all()

#     return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================
def train(epochs=1000, MINIBATCH_SIZE=64, GAMMA=GAMMA, epsilon=1.0, min_epsilon=0.01, BUFFER_SIZE=1e5, train_indicator=True, render=False):
    with tf.Session() as sess:
        env = CartPoleEnv()
        np.random.seed(RANDOM_SEED)
        tf.set_random_seed(RANDOM_SEED)
        env.seed(RANDOM_SEED)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        action_bound = env.action_space.high

        noise = OUNoise(action_dim, mu=0.2)
        actor = ActorNetwork(sess,state_dim, action_dim, action_bound, ACTOR_LEARNING_RATE, TAU)
        critic = CriticNetwork(sess, state_dim, action_dim, CRITIC_LEARNING_RATE, TAU, actor.get_num_trainable_vars())

        sess.run(tf.global_variables_initializer())

        actor.update_target_network()
        critic.update_target_network()

        replay_buffer = ReplayBuffer(BUFFER_SIZE, RANDOM_SEED)

        all_states = []
        # all_actions = []
        all_rewards = []
        mean_rewards = []
        save_reward = None


        for i in range(epochs):
            state = env.reset()
            state = np.hstack(state)
            ep_reward = 0
            ep_ave_max_q = 0
            step = 0
            max_step = 1000
            epsilon -= (epsilon / EXPLORE)
            epsilon = max(min_epsilon, epsilon)
            
            states = []
            rewards = []
            actions = []
            done = False

            while not done:
                if render:
                    env.render
                
                action_original = actor.predict(np.reshape(state, (1, state_dim)))
                action = np.clip(action_original + np.max(epsilon, 0) * noise.noise(), -env.max_action, env.max_action)
                actions.append(action)

                next_state, reward, done, info = env.step(action[0])

                if train_indicator:
                    replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward, done, np.reshape(next_state, (actor.s_dim,)))

                    if replay_buffer.size() > MINIBATCH_SIZE:
                        s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(MINIBATCH_SIZE)

                        target_q = critic.predict_target(s2_batch, actor.predict_target(s2_batch))

                        y_i = []
                        for k in range(MINIBATCH_SIZE):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + GAMMA * target_q[k])

                        predicted_q_value, _ = critic.train(s_batch, a_batch, np.reshape(y_i, (MINIBATCH_SIZE, 1)))

                        ep_ave_max_q += np.amax(predicted_q_value)

                        a_outs = actor.predict(s_batch)

                        grads = critic.action_gradients(s_batch, a_outs)
                        actor.train(s_batch, grads[0])

                        actor.update_target_network()
                        critic.update_target_network()

                states.append(state)
                state = next_state
                rewards.append(reward)
                ep_reward = ep_reward + reward
                step += 1
            
            if step > (max_step -1):
                mean_rewards.append(ep_reward)
                actions = np.array(actions).reshape((len(actions), 1))
                all_states.append(states)
                all_rewards.append(rewards)

                ave_reward = np.mean(mean_rewards[-50:])
                if save_reward == None or ave_reward > save_reward:
                    save_reward = ave_reward
                    critic.save_critic()
                    actor.save_actor()
                    print('th', i + 1, 'n steps', step, 'R:', np.round(ep_reward, 3))


if __name__ == '__main__':
    train(epochs=3000, epsilon=1., render=False)
