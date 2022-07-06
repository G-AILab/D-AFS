"""经验池，实现经验池存入与采样
"""
import numpy as np
import random

class ReplayBuffer(object):
    """经验池基础类
    实现经验池数据存入与采样
    """
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)
    
    def add(self, obs_t, action, reward, obs_tp1, done):
        """数据存入经验池
        Parameters
        ----------
        obs_t：当前状态值
        action：动作值
        reward：奖励值
        obs_tp1：下一时刻状态值
        """
        data = (obs_t, action, reward, obs_tp1, done)
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t,copy=False))
            actions.append(np.array(action,copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1,copy=False))
            dones.append(done)
        return np.array(obses_t),np.array(actions),np.array(rewards),np.array(obses_tp1),np.array(dones)

    def sample(self, batch_size):
        """数据采样
        Parameters:
        -----------
        batch_size：int
            采样batch_size组数据

        return：
        **arg：np.ndarray 
            随机采样获得得batch_size组数据
            
        """
        idxes = [random.randint(0, len(self._storage)-1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
