"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, buffer_size, n_state, n_action, max_ep):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.n_state = n_state
        self.n_action = n_action
        self.cursor = 0
        self.full = False
        self.max_ep = max_ep
        self.episodes = deque(maxlen=buffer_size)
        self.current_episode = []


    def add(self, state, action, reward, done_bool):
        self.current_episode.append(
            (state, action, reward, done_bool)
        )

    @property
    def size(self):
        return self.buffer_size if self.full else self.cursor

    def end_episode(self):
        ep = self.current_episode
        ep_len = len(ep)

        assert ep_len > 0

        # padding
        if ep_len < self.max_ep:
            last_state, last_action, _, _ = ep[-1]
            for _ in range(self.max_ep - ep_len):
                ep.append((
                    last_state,
                    last_action,
                    0.0,
                    1.0,
                ))

        self.episodes.append(ep)
        self.current_episode = []

    def get_latest_episode(self):
        assert len(self.episodes) > 0, "ReplayBuffer empty"

        ep = self.episodes[-1]

        states, actions, rewards, dones = zip(*ep)

        o_t = torch.FloatTensor(states).unsqueeze(0)
        actions = torch.FloatTensor(actions).unsqueeze(0)
        rewards = torch.FloatTensor(rewards).unsqueeze(0)
        dones = torch.FloatTensor(dones).unsqueeze(0)

        return o_t, actions, rewards, dones

    def empty(self):
        self.cursor = 0
        self.full = False
