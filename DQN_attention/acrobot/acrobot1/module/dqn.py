import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from replay_buff.replay_buff import ReplayBuffer
from module.utils import get_session, set_global_seed, fc, ObservationInput, LinearSchedule
from module.build_graph import build_train
from module import utils as U
import pandas as pd


class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params
        self.initial_state = None

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, save_path=None, sess=None, saver=None):
        sess = sess or get_session()
        saver = saver or tf.train.Saver(max_to_keep=1)
        if save_path is not None:
            saver.save(sess, save_path)
        else:
            save_path = tf.train.latest_checkpoint("log/")
            saver.save(sess, save_path)


def build_attention(e_num=6, scope="attention", patten='multiple'):
    def attention_score(x):
        with tf.variable_scope(scope):
            w_num = x.get_shape()[1].value
            e = tf.nn.tanh(fc(x, 'ate_fc', nh=e_num, init_scale=np.sqrt(2)))
            if patten == 'single':
                attention_out = fc(e, 'ata_fc', nh=w_num, init_scale=np.sqrt(2))
                A = tf.nn.softmax(attention_out)
            elif patten == 'multiple':
                E = tf.layers.flatten(e)
                attention_out_list = []
                for i in range(w_num):
                    attention_FC = fc(E, 'ata_fc{}'.format(i), nh=2, init_scale=np.sqrt(2))
                    attention_out = tf.nn.softmax(attention_FC)

                    attention_out = tf.expand_dims(attention_out[:,1], axis=1)

                    attention_out_list.append(attention_out)
                A = tf.squeeze(tf.stack(attention_out_list, axis=1),axis=2)
            return A
    return attention_score


def mlp(num_layers=1, num_hidden=35, activation=tf.tanh, layer_norm=False):
    attention = build_attention()

    def network_fn(x):
        A = attention(x)
        h = tf.layers.flatten(tf.multiply(x, A))
        for i in range(num_layers):
            h = fc(h, 'mlp_fc{}'.format(i), nh=num_hidden, init_scale=np.sqrt(2))
            if layer_norm:
                h = layers.layer_norm(h, center=True, scale=True)
            h = activation(h)
        return h, A
    return network_fn


def build_q_func(network, hiddens=[64], dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network,str):
        network = mlp(**network_kwargs)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            latent, a_score = network(input_placeholder)
            latent = layers.flatten(latent)

            with tf.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    action_out = layers.fully_connected(action_out, num_outputs=hidden, activation_fn=None)
                    if layer_norm:
                        action_out = layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = layers.fully_connected(action_out, num_outputs=num_actions, activation_fn=None)

            if dueling:
                with tf.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = layers.fully_connected(state_out, num_outputs=hidden, activation_fn=None)
                        if layer_norm:
                            state_out = layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    state_scores = layers.fully_connected(state_out, num_outputs=1, activation_fn=None)
                action_scores_mean = tf.reduce_mean(action_scores,1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean,1)
                q_out = state_scores + action_scores_centered
            else:
                q_out = action_scores
            return q_out, a_score

    return q_func_builder


def learn(env, network,
          seed=234,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=64,
          print_freq=100,
          checkpoint_freq=5000,
          checkpoint_path=None,
          learning_starts=1000,
          gamma=1,
          target_network_update_freq=500,
          load_path=None,
          **network_kwargs):

    sess = get_session()
    set_global_seed(seed)
    q_func = build_q_func(network, **network_kwargs)

    observation_space = env.observation_space

    def make_obs_ph(name):
        return ObservationInput(observation_space, name=name)

    act, train, update_target, debug = build_train(
        make_obs_ph=make_obs_ph,
        q_func=q_func,
        num_actions=env.action_space.n,
        optimizer=tf.train.AdamOptimizer(learning_rate=lr),
        gamma=gamma,
        grad_norm_clipping=10,
    )

    act_params = {
        'make_obs_ph': make_obs_ph,
        'q_func': q_func,
        'num_actions': env.action_space.n,
    }

    act = ActWrapper(act, act_params)

    replay_buffer = ReplayBuffer(buffer_size)
    beta_schedule = None

    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=1.0,
                                 final_p=exploration_final_eps)

    saver = tf.train.Saver(max_to_keep=1)
    U.initialize()
    update_target()

    episode_rewards = [0.0]
    saved_mean_reward = None
    obs = env.reset()
    reset = True
    t_count = 0
    attention_episode = []
    mean_attention = []

    model_saved = False
    save_dir = "log/"

    if load_path is not None:
        saver.restore(sess, save_dir + load_path)

    for t in range(total_timesteps):
        kwargs = {}
        update_eps = exploration.value(t)
        update_param_noise_threshold = 0.

        action = act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0][0]
        attention_episode.append(act(np.array(obs)[None], update_eps=update_eps, **kwargs)[1][0])
        env_action = action
        reset = False
        new_obs, rew, done, _ = env.step(env_action)
        # Store transition in the replay buffer.
        replay_buffer.add(obs, action, rew, new_obs, float(done))
        obs = new_obs
        t_count += 1

        episode_rewards[-1] += rew
        if done:
            mean_attention.append(np.mean(attention_episode, 0).tolist())
            obs = env.reset()
            episode_rewards.append(0.0)
            reset = True
            t_count = 0
            attention_episode = []
        elif t_count > 500:
            mean_attention.append(np.mean(attention_episode, 0).tolist())
            obs = env.reset()
            episode_rewards.append(0.0)
            reset = True
            t_count = 0
            done = True
            attention_episode = []

        if t > learning_starts and t % train_freq == 0:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(batch_size)
            weights, batch_idxes = np.ones_like(rewards), None
            td_errors = train(obses_t, actions, rewards, obses_tp1, dones, weights)

        if t > learning_starts and t % target_network_update_freq == 0:
            # Update target network periodically.
            update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
        num_episodes = len(episode_rewards)
        if reset:
            env.episode_reward = np.append(env.episode_reward, mean_100ep_reward)

        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            print("-----------------------------------\n")
            print("%time spent exploring: {}\n".format(round(100 * exploration.value(t), 2)))
            print("episodes: {}\n".format(num_episodes))
            print("mean_100ep_reward: {}\n".format(mean_100ep_reward))
            print("step: {}\n".format(t))
            print("-----------------------------------\n")

        if (checkpoint_freq is not None and t > learning_starts and 
                num_episodes > 50 and t % checkpoint_freq == 0):
            if saved_mean_reward is None or mean_100ep_reward > saved_mean_reward:
                if print_freq is not None:
                    print("Saving model due to mean reward increase: {} -> {}".format(
                        saved_mean_reward, mean_100ep_reward))
                save_step = t
                saver.save(sess, save_dir + "model.cpkt", global_step=save_step)
                model_saved = True
                saved_mean_reward = mean_100ep_reward
    data = pd.DataFrame(mean_attention, columns=['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10'])
    # total_reward = pd.DataFrame(env.episode_reward, columns=['reward'])
    # total_reward.to_csv('reward.csv', index=False)
    data.to_csv('attention_weight.csv', index=False)

    if model_saved:
        if print_freq is not None:
            print("Restored model with mean reward: {}".format(saved_mean_reward))
        saver.restore(sess, save_dir + 'model.cpkt-' + str(save_step))
    return act