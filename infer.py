#coding: utf-8

import os
import torch
import numpy as np

from ddpg import DDPG
import wechat_jump_android as env


def main():
    ddpg = DDPG(0, 0, torch.cuda.is_available())
    env.init_state()

    if os.path.exists('models/ddpg_actor_'):
        ddpg.load_model()
    else:
        print("Please ensure models existing!")

    while True:
        action = ddpg.select_action(env.state)
        env.step(action)
        print(env.last_score)

if __name__ == "__main__":
    main()
