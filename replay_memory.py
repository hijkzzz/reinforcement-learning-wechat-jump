#coding: utf-8
import random
from collections import namedtuple

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition',
                        ('state', 'action', 'mask', 'next_state', 'reward'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.count = [0, 1]

    def push(self, trans):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = trans
        self.position = (self.position + 1) % self.capacity

        self.count[int(trans.reward > 0)] += 1


    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_negative_rate(self):
        return 1 if self.count[1] > self.count[0] else 0.1

    def __len__(self):
        return len(self.memory)
