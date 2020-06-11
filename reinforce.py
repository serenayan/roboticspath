#taken from https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
import argparse
from gym_opt_env import GymOptEnv, Action, OptimState
from graph_gen import generate_graph, point_in_which_regions
from mip_planner import MIPPlanner, DEFAULT_DYN
from obstacle_environment import DEFAULT_ENV
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.autograd

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

torch.autograd.set_detect_anomaly(True)

T = 200 #number of timesteps
NSection = 10
obs_env = DEFAULT_ENV
dyn = DEFAULT_DYN
NCVX = obs_env.get_num_cvx()
planner = MIPPlanner(obs_env, dyn, T, h_k=1e-3, presolve=1)

env = GymOptEnv(planner, T, obs_env.get_num_cvx(), args.seed)


env.seed(args.seed)
torch.manual_seed(args.seed)


class RandomPolicy(nn.Module):
    def __init__(self, ncvx):
        super(RandomPolicy, self).__init__()
        self.delta_t = T // NSection
        self.delta_t_i = 1
        self.ncvx = ncvx
        #AB to H
        self.Asl1 = nn.Linear(self.ncvx*4, 128)
        self.Bsl1 = nn.Linear(self.ncvx*2, 64)
        self.l1l2 = nn.Linear(128 + 64, 256)
        self.l2H = nn.Linear(256, 256)
        #start end to H
        self.sel1 = nn.Linear(8, 64)
        self.l1H = nn.Linear(64, 64)

        #H to region decision
        self.Hd1 = nn.Linear(256 + 64, 64)
        self.d1d2 = nn.Linear(64 + self.ncvx, 32)
        self.d2F = nn.Linear(32, self.ncvx)

        self.saved_log_probs = []
        self.rewards = []

    def setEnvParams(self, As, bs, start, end):
        self.As = As
        self.bs = bs
        self.start = start
        self.end = end
        self.startR = point_in_which_regions(start.numpy(),
                                             obs_env.get_cvx_ineqs()[0],
                                             obs_env.get_cvx_ineqs()[1])[0]
        self.endR = point_in_which_regions(end.numpy(),
                                           obs_env.get_cvx_ineqs()[0],
                                           obs_env.get_cvx_ineqs()[1])[0]
        self.G = generate_graph(
                       obs_env.get_cvx_ineqs()[0],
                       obs_env.get_cvx_ineqs()[1])


    def _make_H(self):
        l1A = nn.ReLU()(self.Asl1(self.As.flatten()))
        l1b = nn.ReLU()(self.Bsl1(self.bs.flatten()))
        l1 = torch.cat((l1A, l1b))
        l2 = nn.ReLU()(self.l1l2(l1))
        Hcvx = nn.ReLU()(self.l2H(l2))

        l1pts = nn.ReLU()(self.sel1(torch.cat((self.start, self.end))))
        Hpts = nn.ReLU()(self.l1H(l1pts))
        return torch.cat([Hcvx, Hpts])

    def _adjacency_filter(self, node:int):
        nodes = [n for n in self.G.neighbors(node)]
        filter = torch.zeros((self.ncvx))
        filter.scatter_(0, torch.tensor(nodes), 1.0)
        return filter

    def _prev_x_to_next_x(self, x_prev):
        d1 = nn.ReLU()(self.Hd1(self._make_H()))
        d2 = nn.ReLU()(self.d1d2(torch.cat([d1, x_prev])))
        f = nn.ReLU()(self.d2F(d2))
        f = F.softmax(f, dim=0)
        return f
    def forward(self, x, u, aset, iset_val):
        """

        :param x:
        :param u:
        :param aset:
        :param iset_val:
        :return:
        """

        # x = self.affine1(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        assert self.delta_t_i < NSection
        if self.delta_t_i == 1:
            iset_val = iset_val - 1
            iset_val[:self.delta_t, self.startR] = 1
            iset_val[-self.delta_t:,self.endR] = 1
        iset_val[self.delta_t*self.delta_t_i:-self.delta_t] = 0
        aset.fill_(0)
        if self.delta_t_i < NSection -1:
            aset[self.delta_t_i*self.delta_t: \
                 self.delta_t_i*self.delta_t + self.delta_t] = 1



        for i in range(self.delta_t_i,NSection - 1):
            f = self._prev_x_to_next_x(iset_val[self.delta_t * i -1])
            if i == self.delta_t_i:
                prev_filter = iset_val[self.delta_t*i-1]
                n = torch.argmax(prev_filter).tolist()
                filtered = f * self._adjacency_filter(n)
                # assert torch.norm(filtered, 1) > 0
                # filtered /= torch.norm(filtered, 1)
                f = filtered
            elif i == NSection -1:
                last_filter = iset_val[-1]
                n = torch.argmax(last_filter).tolist()
                filtered = f * self._adjacency_filter(n)
                # assert torch.norm(filtered, 1) > 0
                # filtered /= torch.norm(filtered)
                f = filtered
            iset_val[self.delta_t*i:self.delta_t*(i+1)] = f

        return aset, iset_val

Abs = obs_env.get_cvx_ineqs()
policy = RandomPolicy(NCVX)
policy.setEnvParams(torch.stack([torch.from_numpy(A).float() for A in Abs[0]]),
                    torch.cat([torch.from_numpy(b).float() for b in Abs[1]]),
                    torch.FloatTensor(obs_env.start),
                    torch.FloatTensor(obs_env.end))
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    x = torch.from_numpy(state.x)
    u = torch.from_numpy(state.u)
    # print(state)
    # print(state.aset)
    # aset = torch.from_numpy(state.aset.float())
    iset_val = torch.from_numpy(state.iset_val).float()

    aset, iset_val_dist = policy(x, u, aset, iset_val)
    deltat = policy.delta_t

    iset_val_m = Categorical(iset_val_dist)
    iset_val_sample = iset_val_m.sample()
    iset_val_one_hot = torch.zeros_like(iset_val)
    for i in range(NSection):
        iset_val_one_hot[i*deltat:i*deltat + deltat, iset_val_sample[i*deltat]] = 1
        iset_val_sample[i*deltat:i*deltat + deltat] = iset_val_sample[i*deltat]


    policy.saved_log_probs.append(iset_val_m.log_prob(iset_val_sample))
    action = Action(aset.bool().numpy(),
                    iset_val_one_hot.bool().numpy())
    policy.delta_t_i += 1
    return action


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    else:
        returns = (returns - returns.mean()) / (eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), 0.0001)
    optimizer.step()
    policy.delta_t_i = 1
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, 100):  # Don't infinite loop while learning
            action = select_action(state)
            print("action\n", action)
            state, reward, done = env.step(action)
            print("state\n", state, "reward\n", reward)
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
        if running_reward > env.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
