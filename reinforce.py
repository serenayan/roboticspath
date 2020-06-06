#taken from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
import argparse
from gym_opt_env import GymOptEnv, Action, OptimState
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = GymOptEnv()
env.seed(args.seed)
torch.manual_seed(args.seed)


class RandomPolicy(nn.Module):
    def __init__(self):
        super(RandomPolicy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.dropout = nn.Dropout(p=0.6)
        self.affine2 = nn.Linear(4, 200)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, u, aset, iset_val):
        # x = self.affine1(x)
        # x = self.dropout(x)
        # x = F.relu(x)

        return F.softmax(aset, dim=1), torch.rand_like(iset_val)


policy = RandomPolicy()
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    x = torch.from_numpy(state.x)
    u = torch.from_numpy(state.u)
    aset = torch.from_numpy(state.aset).float()
    iset_val = torch.from_numpy(state.iset_val).float()

    aset_dist, iset_val_dist = policy(x, u, aset, iset_val)
    from torch.distributions import Bernoulli
    aset_m = Bernoulli(aset_dist)
    iset_val_m = Bernoulli(iset_val_dist)
    aset_sample = aset_m.sample()
    iset_val_sample = iset_val_m.sample()
    # aset_m.log_prob(new_aset)
    # action = m.sample()
    T = 200
    pick_one_m = Categorical(torch.ones(4) * 0.25)

    for t in range(200):
        if not torch.any(aset_sample[t].bool()) and\
            not torch.any(iset_val_sample[t].bool()):
            iset_val_sample[t, pick_one_m.sample()] = 1

    policy.saved_log_probs.append((aset_m.log_prob(aset_sample),
                                   iset_val_m.log_prob(iset_val_sample)))
    action = Action(aset_sample.bool().numpy(),
                    iset_val_sample.bool().numpy())
    # return action.item()
    return action


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-torch.cat(log_prob) * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()