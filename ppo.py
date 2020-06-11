# taken from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from gym_opt_env import GymOptEnv, Action, OptimState
from graph_gen import generate_graph, point_in_which_regions
from mip_planner import MIPPlanner, DEFAULT_DYN
from obstacle_environment import DEFAULT_ENV

T = 1000 # number of timesteps
seed = 1
obs_env = DEFAULT_ENV
dyn = DEFAULT_DYN
NCVX = obs_env.get_num_cvx()
planner = MIPPlanner(obs_env, dyn, T, h_k=1e-3, presolve=1)
env = GymOptEnv(planner, T, obs_env.get_num_cvx(), seed)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                    obs_batch, recurrent_hidden_states_batch, masks_batch,
                    actions_batch)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class RandomPolicy(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(RandomPolicy, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(recurrent_input_size, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()


    def forward(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flattened to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return self.critic_linear(x), x, hxs

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

Abs = obs_env.get_cvx_ineqs()
policy = RandomPolicy(NCVX, 2, 10)
policy.setEnvParams(torch.stack([torch.from_numpy(A).float() for A in Abs[0]]),
                    torch.cat([torch.from_numpy(b).float() for b in Abs[1]]),
                    torch.FloatTensor(obs_env.start),
                    torch.FloatTensor(obs_env.end))
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
