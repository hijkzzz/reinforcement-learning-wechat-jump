#coding: utf-8

import torch
import numpy as np

from network import DDPG
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
import wechat_jump_android as env

SEED = 4
BATCH_SIZE = 32
REPLAY_SIZE = 10000
NUM_EPISODES = 10 * 10000
GAMMA = 0.9
TAU = 0.1
EXPLORATION_END = 100
UPDATES_PER_STEP = 3

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    net = DDPG(GAMMA, TAU, torch.cuda.is_available())
    memory = ReplayMemory(REPLAY_SIZE)
    ounoise = OUNoise(1, scale=1, theta=0.15, sigma=1)

    state = env.get_init_state()
    updates = 0

    for i_episode in range(NUM_EPISODES):

        while True:
            action = net.select_action(state, ounoise)
            transition = env.step(action)
            memory.push(transition)

            # Game Over
            if transition.next_state is None:
                state = env.get_init_state()
            else:
                state = transition.next_state

            if len(memory) > BATCH_SIZE:
                for _ in range(UPDATES_PER_STEP):
                    transitions = memory.sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions)) # * 打散为参数
                    value_loss, policy_loss = net.update_parameters(batch)
                    updates += 1

                print(
                    "Episode: {}, Updates: {}, Value Loss: {}, Policy Loss: {}".
                    format(i_episode, updates, value_loss, policy_loss))


if __name__ == "__main__":
    main()
