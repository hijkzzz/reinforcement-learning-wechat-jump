#coding: utf-8
import sys
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F


def soft_update(target, source, tau):
    """Update target network parameters
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def hard_update(target, source):
    """Update target network parameters
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class Actor(nn.Module):
    """Actor Network
    """

    def __init__(self):
        super(Actor, self).__init__()
        # 3 * 224 * 224
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 111 * 111
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 54 * 54
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 26 * 26
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 12 * 12
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 5 * 5
        self.layer6 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 1),
            nn.Tanh()
        )

    def forward(self, inputs):
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.layer6(out)

        return out


class Critic(nn.Module):
    """Critic Network
    """

    def __init__(self):
        super(Critic, self).__init__()
        # 3 * 224 * 224
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 111 * 111
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 54 * 54
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 26 * 26
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 12 * 12
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64, momentum=1),
            nn.ReLU(), nn.MaxPool2d(2))
        # 64 * 5 * 5 + 1
        self.layer6 = nn.Sequential(nn.Linear(64 * 5 * 5 + 1, 128), nn.ReLU())
        # 128 * 1
        self.layer7 = nn.Sequential(nn.Linear(128, 1), nn.ReLU())

    def forward(self, inputs, actions):

        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = torch.cat((out, actions), 1)
        out = self.layer6(out)
        out = self.layer7(out)

        return out


class DDPG(object):
    def __init__(self, gamma, tau, cuda=False):

        self.actor = Actor()
        self.actor_target = Actor()
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic()
        self.critic_target = Critic()
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-4)

        self.cuda = cuda
        self.gamma = gamma
        self.tau = tau

        if self.cuda:
            self.actor.cuda()
            self.actor_target.cuda()

            self.critic.cuda()
            self.critic_target.cuda()

        hard_update(self.actor_target,
                    self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    def select_action(self, state, action_noise=None):
        """Select action with noise
        """

        self.actor.eval()
        mu = self.actor(
            Variable(state).cuda() if self.cuda else Variable(state))

        self.actor.train()
        mu = mu.data

        if action_noise is not None:
            mu += torch.Tensor(action_noise.noise()).cuda() \
                if self.cuda else torch.Tensor(action_noise.noise())

        return mu.data[0].cpu().numpy() if self.cuda else mu.data[0].numpy()

    def update_parameters(self, batch):
        """Train actor network and critic network
        """

        state_batch = Variable(torch.cat(batch.state))
        action_batch = Variable(torch.cat(batch.action))
        reward_batch = Variable(torch.cat(batch.reward))
        mask_batch = Variable(torch.cat(batch.mask))  # 0 == GAME OVER
        next_state_batch = Variable(torch.cat(batch.next_state))

        if self.cuda:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            mask_batch = mask_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        next_action_batch = self.actor_target(next_state_batch)
        next_q_values = self.critic_target(next_state_batch, next_action_batch)

        # 对齐形状
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = mask_batch.unsqueeze(1)
        expected_q_batch = reward_batch + (
            self.gamma * mask_batch * next_q_values)

        # Train Critic Network
        self.critic_optim.zero_grad()
        q_batch = self.critic((state_batch), (action_batch))
        value_loss = F.mse_loss(q_batch, expected_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Train Actor Network
        self.actor_optim.zero_grad()
        # Maxmise E(Value)
        policy_loss = -self.critic((state_batch), self.actor((state_batch)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Copy parameters to target network
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_model(self, suffix=""):
        """Save Actor and Critic
        """

        if not os.path.exists('models/'):
            os.makedirs('models/')

        actor_path = "models/ddpg_actor_{}".format(suffix)
        critic_path = "models/ddpg_critic_{}".format(suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))

        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        """Load Actor and Critic
        """

        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
