#coding: utf-8

import os
import torch
import numpy as np

from network import DDPG
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
import wechat_jump_android as env

SEED = 4
BATCH_SIZE = 32
REPLAY_SIZE = 10000
NUM_EPISODES = 100000
GAMMA = 0.9
TAU = 0.1
EXPLORATION_END = 1000
UPDATES_PER_STEP = 2

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    net = DDPG(GAMMA, TAU, True)
    memory = ReplayMemory(REPLAY_SIZE)
    ounoise = OUNoise(1, scale=1)

    state = env.get_init_state()
    updates = 0

    if os.path.exists('models/ddpg_actor_'):
        net.load_model('models/ddpg_actor_', 'models/ddpg_critic_')

    for i_episode in range(NUM_EPISODES):

        while True:
            ounoise.reset()

            action = net.select_action(state, ounoise)
            transition = env.step(action)
            memory.push(transition)

            # Game Over
            if transition.reward == -2:
                state = env.get_init_state()
            else:
                state = transition.next_state

            if len(memory) > BATCH_SIZE:
                for _ in range(UPDATES_PER_STEP):
                    transitions = memory.sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = net.update_parameters(batch)
                    updates += 1

                print(
                    "Episode: {}, Updates: {}, Value Loss: {}, Policy Loss: {}".
                    format(i_episode, updates, value_loss, policy_loss))

            if (i_episode + 1) % 1000 == 0:
                net.save_model()

if __name__ == "__main__":
    main()
