"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np
import torch

class ReplayBuffer(object):
    def __init__(self, buffer_size, n_state, n_action, max_ep,  random_seed=123):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.n_state = n_state
        self.n_action = n_action
        self.full = False
        self.max_ep = max_ep
        self.episodes = deque(maxlen=buffer_size)
        self.current_episode = []
        random.seed(random_seed)


    def add(self, state, action, reward, done_bool):
        self.current_episode.append(
            (state, action, reward, done_bool, 1.0)
        )

    @property
    def size(self):
        return self.buffer_size if self.full else len(self.episodes)

    def end_episode(self):
        ep = self.current_episode
        ep_len = len(ep)

        assert ep_len > 0

        # padding
        if ep_len < self.max_ep:
            last_state, last_action, _, _, _ = ep[-1]
            for _ in range(self.max_ep - ep_len):
                ep.append((
                    last_state,
                    last_action,
                    0.0,
                    1.0,
                    0.0,
                ))

        elif ep_len > self.max_ep:
            ep = ep[-self.max_ep:]

        self.episodes.append(ep)
        self.current_episode = []

    def sample_batch(self, batch_size):
        if len(self.episodes) < batch_size:
            batch = random.sample(self.episodes, len(self.episodes))
        else:
            batch = random.sample(self.episodes, batch_size)

        states, actions, rewards, dones, masks = [], [], [], [], []

        for ep in batch:
            s, a, r, d, m = zip(*ep)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            dones.append(d)
            masks.append(m)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        masks = torch.FloatTensor(masks)

        return states, actions, rewards, dones, masks

    def empty(self):
        self.full = False
