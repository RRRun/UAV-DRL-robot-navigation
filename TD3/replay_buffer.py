"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size, n_state, n_action):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.n_state = n_state
        self.n_action = n_action
        self.cursor = 0
        self.full = False

        self.states = np.zeros((buffer_size, n_state), dtype=np.float32)
        self.actions = np.zeros((buffer_size, n_action), dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)


    def add(self, state, action, reward, done_bool):
        self.states[self.cursor] = state
        self.actions[self.cursor] = action
        self.rewards[self.cursor] = reward
        self.dones[self.cursor] = done_bool

        self.cursor += 1
        if self.cursor >= self.buffer_size:
            self.cursor = 0
            self.full = True

    @property
    def size(self):
        return self.buffer_size if self.full else self.cursor

    def sample_batch(self, batch_size, hist_size):
        assert self.size > 0, "Replay memory is empty"
        assert hist_size >= 1, "History size must be >= 1"

        idx = np.zeros(batch_size, dtype=np.int32)
        count = 0

        while count < batch_size:

            #index是s_t的索引
            index = np.random.randint(hist_size - 1, self.size - 1)

            #检查光标是否越界
            if self.cursor <= index + 1 < self.cursor + hist_size:
                continue

            #s_t不能包含中间终止状态
            if np.any(self.dones[index - (hist_size - 1):index]):
                continue

            #防止负索引
            if index - (hist_size - 1) < 0:
                continue

            idx[count] = index
            count += 1

        all_indices = idx.reshape(-1, 1) + np.arange(-(hist_size - 1), 2)
        states = self.states[all_indices]
        actions = self.actions[all_indices[:, :-1]]
        rewards = self.rewards[all_indices[:, :-1]]
        dones = self.dones[all_indices[:, :-1]]

        # 检查 batch 大小
        assert idx.shape == (batch_size,)
        assert states.shape == (batch_size, hist_size + 1, self.n_state)
        assert actions.shape == (batch_size, hist_size, self.n_action)
        assert rewards.shape == (batch_size, hist_size)
        assert dones.shape == (batch_size, hist_size)

        return dict(
            states=states,
            actions=actions,
            rewards=rewards,
            dones=dones
        )

    def empty(self):
        self.cursor = 0
        self.full = False
