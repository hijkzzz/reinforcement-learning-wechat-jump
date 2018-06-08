#coding: utf-8

import os
import torch
import numpy as np

from network import DDPG
from ounoise import OUNoise
from replay_memory import ReplayMemory, Transition
import wechat_jump_android as env

SEED = 4
BATCH_SIZE = 4
REPLAY_SIZE = 10000
NUM_EPISODES = 100000
GAMMA = 0.9
TAU = 0.1
EXPLORATION_END = 100
UPDATES_PER_STEP = 1

torch.manual_seed(SEED)
np.random.seed(SEED)


def main():
    net = DDPG(GAMMA, TAU, torch.cuda.is_available())
    memory = ReplayMemory(REPLAY_SIZE)
    ounoise = OUNoise(1, scale=0.8)
    env.init_state()

    if os.path.exists('models/ddpg_actor_'):
        net.load_model('models/ddpg_actor_', 'models/ddpg_critic_')


    for i_episode in range(NUM_EPISODES):
        ounoise.reset()

        while True:
            action = net.select_action(state, ounoise) \
                    if i_episode < EXPLORATION_END else net.select_action(env.state)
            transition = env.step(action)
            memory.push(transition)

            if len(memory) > BATCH_SIZE:
                for _ in range(UPDATES_PER_STEP):
                    transitions = memory.sample(BATCH_SIZE)
                    batch = Transition(*zip(*transitions))
                    value_loss, policy_loss = net.update_parameters(batch)

                print(
                    "Episode: {}, Value Loss: {}, Policy Loss: {}".
                    format(i_episode, value_loss, policy_loss))
                break

        if (i_episode + 1) % 1000 == 0:
            net.save_model()

if __name__ == "__main__":
    main()
