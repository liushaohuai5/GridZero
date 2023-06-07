import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_utils
from torch_utils import *
from utilize.settings import settings
import copy
from game.gridsim.utils import *
# from game.gridsim.utils import action_mapping, calc_running_cost_rew
from torch.cuda.amp import autocast

def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict


class ResidualForwardModel(nn.Module):
    def __init__(self, s_shape, a_shape, dyn_model, activation=nn.ReLU, use_bn=False):
        super().__init__()
        self.mlp = mlp(s_shape + a_shape, dyn_model, s_shape, activation=activation, use_bn=use_bn)
        # self.mlp = nn.Sequential(
        #     nn.Linear(s_shape+a_shape, dyn_model[0]),
        #     nn.BatchNorm1d(dyn_model[0]),
        #     nn.ReLU(),
        # )

    def forward(self, s, a):
        delta_s = self.mlp(torch.cat((s, a), dim=-1))
        s = s + delta_s
        return s

class MultipleRewardModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_bn = config.use_bn
        self.activation = config.activation
        self.rew_dyn_act_dim = config.rew_dyn_act_dim
        self.hidden_shape = config.mlp_hidden_shape
        self.rew_net_shape = config.mlp_rew_shape

        self.reward_support_size = config.reward_support_size
        self.reward_support_step = config.reward_support_step
        self.full_reward_support_size = 2 * config.reward_support_size + 1

        self.line_of_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                   self.full_reward_support_size,
                                   activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)
        self.renewable_consump_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                             self.full_reward_support_size,
                                             activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)
        self.run_cost_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                             self.full_reward_support_size,
                                             activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)
        self.bal_gen_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                             self.full_reward_support_size,
                                             activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)
        self.reac_pow_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                             self.full_reward_support_size,
                                             activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)

    def forward(self, hidden_action):
        line_of_rew = self.line_of_rew_net(hidden_action)
        renewable_comsump_rew = self.renewable_consump_rew_net(hidden_action)
        run_cost_rew = self.run_cost_rew_net(hidden_action)
        bal_gen_rew = self.bal_gen_rew_net(hidden_action)
        reac_pow_rew = self.bal_gen_rew_net(hidden_action)
        return [line_of_rew, renewable_comsump_rew, run_cost_rew, bal_gen_rew, reac_pow_rew]


class MLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_bn = config.use_bn
        self.activation = config.activation
        self.gamma = config.discount
        self.support_size = config.support_size
        self.obs_shape = config.mlp_obs_shape
        self.action_shape = config.mlp_action_shape
        self.rew_dyn_act_dim = config.rew_dyn_act_dim
        self.policy_act_dim = config.policy_act_dim
        self.hidden_shape = config.mlp_hidden_shape
        self.proj_shape = config.mlp_proj_shape
        self.n_stacked_obs = config.stacked_observations

        self.rep_net_shape = config.mlp_rep_shape
        self.rew_net_shape = config.mlp_rew_shape
        self.val_net_shape = config.mlp_val_shape
        self.pi_net_shape = config.mlp_pi_shape
        self.expert_pi_net_shape = config.mlp_expert_pi_shape
        self.dyn_shape = config.mlp_dyn_shape
        self.proj_net_shape = config.mlp_proj_net_shape
        self.proj_pred_shape = config.mlp_proj_pred_net_shape

        self.reward_support_size = config.reward_support_size
        self.reward_support_step = config.reward_support_step
        self.full_reward_support_size = 2 * config.reward_support_size + 1
        self.attacker_action_shape = config.attacker_action_dim

        self.generator_num = self.config.generator_num

        '''
            Models
        '''
        self.rep_net = mlp(self.obs_shape * self.n_stacked_obs, self.rep_net_shape, self.hidden_shape,
                           activation=self.activation, use_bn=self.use_bn)
        self.dyn_net = ResidualForwardModel(self.hidden_shape, self.rew_dyn_act_dim, self.dyn_shape,
                                            activation=self.activation, use_bn=self.use_bn)
        if self.config.multi_reward:
            self.rew_net = MultipleRewardModel(config)
        else:
            self.rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape, self.full_reward_support_size,
                activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)

        self.pi_net = mlp(self.hidden_shape, self.pi_net_shape,
                          self.policy_act_dim,
                          activation=self.activation,
                          use_bn=self.use_bn,
                          init_zero=config.init_zero
                          )

        # if self.config.add_attacker:
        self.pi_net_attacker = mlp(self.hidden_shape, self.pi_net_shape, self.attacker_action_shape,
                                   activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)

        self.rew_net_attacker = mlp(self.hidden_shape + self.attacker_action_shape, self.rew_net_shape,
                                    self.full_reward_support_size,
                                    activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)
        self.dyn_net_attacker = ResidualForwardModel(self.hidden_shape, self.attacker_action_shape, self.dyn_shape)
        # self.rew_net_attacker = mlp(self.hidden_shape + 1, self.rew_net_shape,
        #                             self.full_reward_support_size,
        #                             activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)
        # self.dyn_net_attacker = ResidualForwardModel(self.hidden_shape, 1, self.dyn_shape)

        self.val_net = mlp(self.hidden_shape, self.val_net_shape, config.support_size * 2 + 1,
                           activation=self.activation, use_bn=self.use_bn, init_zero=config.init_zero)

        if self.config.efficient_imitation:
            self.expert_pi = mlp(self.hidden_shape, self.expert_pi_net_shape,
                          self.policy_act_dim,
                          activation=self.activation,
                          use_bn=self.use_bn,
                          init_zero=config.init_zero
                          )

        self.proj_net = mlp(self.hidden_shape, self.proj_net_shape, self.proj_shape,
                            activation=self.activation, use_bn=self.use_bn)
        self.proj_pred_net = mlp(self.proj_shape, self.proj_pred_shape, self.proj_shape,
                                 activation=self.activation, use_bn=self.use_bn)
        self.time_step = 0

        self.init_std = 0.0
        self.min_std = 0.1


    def discrim(self, hidden, action):
        return self.dis_net(hidden, action)

    def value(self, hidden):
        return self.val_net(hidden)

    # def value_attacker(self, hidden):
    #     return self.val_net_attacker(hidden)

    def value_obs(self, obs):
        hidden = self.rep_net(obs)
        return self.value(hidden)

    # @profile
    def policy_no_split(self, hidden):
        pi = self.pi_net(hidden)

        generator_num = self.generator_num
        pi[:, :generator_num] = 5 * torch.tanh(pi[:, :generator_num] / 5)  # soft clamp mu
        pi[:, generator_num:2 * generator_num] = torch.nn.functional.softplus(pi[:, generator_num:2 * generator_num] + self.init_std) + self.min_std

        return pi

    def policy_attacker(self, hidden):
        pi = self.pi_net_attacker(hidden)
        return pi

    def policy(self, hidden):
        pi = self.pi_net(hidden)
        mu, log_std = pi[:, :self.action_shape], pi[:, self.action_shape:]
        return mu, log_std

    def policy_obs(self, obs):
        hidden = self.rep_net(obs)
        return self.policy(hidden)

    def reward(self, hidden, action):
        return self.rew_net(torch.cat((hidden, action), dim=1))

    def reward_attacker(self, hidden, action, prev_r):
        '''
        reward = mask * last_reward + (1 - mask) * predicted_reward
        :param hidden:
        :param action:
        :param prev_r:
        :return:
        '''
        mask = action[:, -1].reshape(hidden.shape[0], 1).repeat(1, prev_r.shape[1])     # mask for no-attack
        # action_modified = torch.where(action > 0)[-1].reshape(action.shape[0], 1) / self.config.attacker_action_dim
        # predicted_r = self.rew_net_attacker(torch.cat((hidden, action_modified), dim=1))
        predicted_r = self.rew_net_attacker(torch.cat((hidden, action), dim=-1))
        r = mask * prev_r + (1 - mask) * predicted_r
        return r

    def reward_obs(self, obs, action):
        hidden = self.rep_net(obs)
        return self.reward(hidden, action)

    def sample_action(self, policy):
        action_dim = policy.shape[-1] // 2
        mu, log_std = policy[:, :action_dim], policy[:, action_dim:]
        sigma = torch.exp(log_std)
        distr = torch.distributions.Normal(mu, sigma)
        action = distr.sample()
        # Tanh inverse?
        action = torch.nn.functional.tanh(action)
        # action, _ = reparameterize_clip(mu, log_std)
        return action

    def sample_actions(self, policy, n):
        action_dim = policy.shape[-1] // 2
        mu, log_std = policy[:, :action_dim], policy[:, action_dim:]
        sigma = torch.exp(log_std)

        distr = torch.distributions.Normal(mu, sigma)
        action = distr.sample(torch.Size([n]))
        action = torch.nn.functional.tanh(action)
        action = action.permute(1, 0, 2)

        return action

    # @profile
    def sample_mixed_actions(self, policy, config, is_root,
                             state=None, action_high=None, action_low=None, ready_mask=None, closable_mask=None, is_test=False, expert_policy=None, train_steps=0):
        n_batchsize = policy.shape[0]
        n_policy_action = config.mcts_num_policy_samples
        n_random_action = config.mcts_num_random_samples
        n_expert_action = config.mcts_num_expert_samples

        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim

        temperature = self.config.visit_softmax_temperature_fn(train_steps)
        if self.config.efficient_imitation:
            policy, expert_policy = torch.chunk(policy, 2, dim=-1)
        else:
            expert_policy = None

        if is_root:
            # RETURNING, [BATCHSIZE, N, ACTION_DIM]

            if config.explore_type == 'add':
                action_dim = policy.shape[-1] // 2
                policy = policy.reshape(-1, policy.shape[-1])

                mu, log_std = policy[:, :action_dim], policy[:, action_dim:]

                sigma = torch.exp(log_std)
                distr = torch.distributions.Normal(mu, sigma)

                policy_action = distr.sample(torch.Size([n_policy_action]))  # [n_pol, batchsize, a_dim]
                policy_action = torch.nn.functional.tanh(policy_action)
                policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

                if n_random_action > 0:
                    random_action = distr.sample(torch.Size([n_random_action]))  # [n_pol, batchsize, a_dim]
                    random_action += torch.randn_like(random_action) * config.explore_scale
                    random_action = torch.nn.functional.tanh(random_action)
                    random_action = torch_utils.tensor_to_numpy(random_action.permute(1, 0, 2))
                else:
                    random_action = None

                if n_random_action > 0:
                    return np.concatenate([policy_action, random_action], axis=1)
                else:
                    return policy_action

            elif config.explore_type == 'normal':

                if n_random_action > 0:
                    random_action = config.sample_random_actions_fast(n_random_action * n_batchsize)
                    random_action = random_action.reshape(n_batchsize, -1, config.action_space_size)
                    random_action = np.tanh(random_action).astype(np.float32)
                else:
                    random_action = None

                # random_actions = np.clip(random_actions, -1, 1)

                action_dim = policy.shape[-1] // 2
                policy = policy.reshape(-1, policy.shape[-1])

                mu, log_std = policy[:, :action_dim], policy[:, action_dim:]
                # log_std = torch.atanh(log_std)

                sigma = torch.exp(log_std)
                distr = torch.distributions.Normal(mu, sigma)

                policy_action = distr.sample(torch.Size([n_policy_action])) #[n_pol, batchsize, a_dim]
                policy_action = torch.nn.functional.tanh(policy_action)
                policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

                if n_random_action > 0:
                    return np.concatenate([policy_action, random_action], axis=1)
                else:
                    return policy_action

            elif config.explore_type == 'manual':
                if self.config.parameters['only_power']:
                    policy = policy.reshape(-1, policy.shape[-1])
                    mu, log_std, open_logits, close_logits = policy[:, :generator_num], \
                                                             policy[:, generator_num:2*generator_num], \
                                                             policy[:, 2*generator_num:2*generator_num+one_hot_dim], \
                                                             policy[:, 2*generator_num+one_hot_dim:]
                else:
                    policy = policy.reshape(-1, policy.shape[-1])
                    mu, log_std, open_logits, close_logits = policy[:, :2*generator_num], \
                                                             policy[:, 2*generator_num:4*generator_num], \
                                                             policy[:, 4*generator_num:4*generator_num+one_hot_dim], \
                                                             policy[4*generator_num+one_hot_dim:]
                # sigma = torch.exp(log_std) * temperature
                # sigma = torch.exp(log_std)
                sigma = log_std
                # distr = SquashedNormal(mu, sigma + 5e-4)
                distr = SquashedNormal(mu, sigma)

                if expert_policy is not None:
                    if self.config.parameters['only_power']:
                        expert_mu, expert_log_std, expert_open_logits, expert_close_logits = expert_policy[:, :generator_num], \
                                                                                             expert_policy[:, generator_num:2*generator_num], \
                                                                                             policy[:, 2 * generator_num:2 * generator_num + one_hot_dim], \
                                                                                             policy[:, 2 * generator_num + one_hot_dim:]
                    else:
                        expert_mu, expert_log_std, expert_open_logits, expert_close_logits = policy[:, :2 * generator_num], \
                                                                                             policy[:, 2 * generator_num:4 * generator_num], \
                                                                                             policy[:, 4 * generator_num:4 * generator_num + one_hot_dim], \
                                                                                             policy[4 * generator_num + one_hot_dim:]
                    # expert_sigma = torch.exp(expert_log_std)
                    expert_sigma = expert_log_std
                    # expert_distr = SquashedNormal(expert_mu, expert_sigma + 5e-4)
                    expert_distr = SquashedNormal(expert_mu, expert_sigma)

                if not is_test:
                    policy_action = distr.sample(torch.Size([n_policy_action]))  # [n_pol, batchsize, a_dim]
                    policy_action = policy_action.clip(-0.999, 0.999)
                    policy_action = policy_action.permute(1, 0, 2)

                    if n_random_action > 0:
                        random_action = distr.sample(torch.Size([n_random_action//2]))  # [n_pol, batchsize, a_dim]
                        random_action += torch.randn_like(random_action) * config.explore_scale
                        # random_action = torch.nn.functional.tanh(random_action)
                        new_distr = SquashedNormal(mu, 3*sigma+5e-4)
                        random_action_new = new_distr.sample(torch.Size([n_random_action//2]))
                        random_action = torch.cat((random_action, random_action_new), dim=0)
                        random_action = random_action.clip(-0.999, 0.999)
                        random_action = random_action.permute(1, 0, 2)
                    else:
                        random_action = None

                    if n_random_action > 0:
                        policy_action = torch.cat((policy_action, random_action), dim=1)

                    if expert_policy is not None:
                        if n_expert_action > 0:
                            expert_action = expert_distr.sample(torch.Size([n_expert_action]))
                            expert_action = expert_action.clip(-0.999, 0.999)
                            expert_action = expert_action.permute(1, 0, 2)
                            policy_action = torch.cat((policy_action, expert_action), dim=1)
                else:
                    policy_action = distr.sample(torch.Size([n_policy_action + n_random_action]))
                    policy_action = policy_action.clip(-0.999, 0.999)
                    policy_action = policy_action.permute(1, 0, 2)

                    if expert_policy is not None:
                        if n_expert_action > 0:
                            expert_action = expert_distr.sample(torch.Size([n_expert_action]))
                            expert_action = expert_action.clip(-0.999, 0.999)
                            expert_action = expert_action.permute(1, 0, 2)
                            policy_action = torch.cat((policy_action, expert_action), dim=1)

                state = torch.from_numpy(state).float().to('cuda')
                action_high = torch.from_numpy(action_high).float().to('cuda')
                action_low = torch.from_numpy(action_low).float().to('cuda')
                ready_mask = torch.from_numpy(ready_mask).float().to('cuda')
                closable_mask = torch.from_numpy(closable_mask).float().to('cuda')
                policy_action = policy_action.float()

                modified_action = self.check_balance(
                    state,
                    policy_action,
                    action_high,
                    action_low,
                    ready_mask,
                    closable_mask)

                if not is_test:
                    policy_open_one_hots_real, policy_close_one_hots_real = \
                        self.determine_open_close_one_hot(policy, n_policy_action, modified_action[:, :n_policy_action, :],
                                                          ready_mask, closable_mask,
                                                          add_explore_noise=False, is_root=is_root, ori_states=state)
                    policy_open_one_hots_random, policy_close_one_hots_random = \
                        self.determine_open_close_one_hot(policy, n_random_action, modified_action[:, n_policy_action:n_policy_action+n_random_action, :],
                                                          ready_mask, closable_mask,
                                                          add_explore_noise=True, is_root=is_root, ori_states=state)
                    if expert_policy is not None:
                        policy_open_one_hots_expert, policy_close_one_hots_expert = self.determine_open_close_one_hot(
                            expert_policy, n_expert_action, modified_action[:, n_policy_action+n_random_action:, :],
                            ready_mask, closable_mask,
                            add_explore_noise=False, is_root=is_root, ori_states=state)
                        policy_open_one_hots = torch.cat((policy_open_one_hots_real, policy_open_one_hots_random, policy_open_one_hots_expert), dim=1)
                        policy_close_one_hots = torch.cat((policy_close_one_hots_real, policy_close_one_hots_random, policy_close_one_hots_expert), dim=1)
                    else:
                        policy_open_one_hots = torch.cat((policy_open_one_hots_real, policy_open_one_hots_random), dim=1)
                        policy_close_one_hots = torch.cat((policy_close_one_hots_real, policy_close_one_hots_random), dim=1)

                    modified_action = torch.cat((modified_action, policy_open_one_hots, policy_close_one_hots),
                                                     dim=2)
                else:
                    policy_open_one_hots, policy_close_one_hots = \
                        self.determine_open_close_one_hot(policy, n_policy_action+n_random_action,
                                                          modified_action[:, :n_policy_action+n_random_action, :],
                                                          ready_mask, closable_mask,
                                                          add_explore_noise=False, is_root=is_root, ori_states=state)
                    if expert_policy is not None:
                        policy_open_one_hots_expert, policy_close_one_hots_expert = self.determine_open_close_one_hot(
                            expert_policy, n_expert_action, modified_action[:, n_policy_action + n_random_action:, :],
                            ready_mask, closable_mask,
                            add_explore_noise=False, is_root=is_root, ori_states=state)
                        policy_open_one_hots = torch.cat((policy_open_one_hots, policy_open_one_hots_expert), dim=1)
                        policy_close_one_hots = torch.cat((policy_close_one_hots, policy_close_one_hots_expert), dim=1)

                    modified_action = torch.cat((modified_action, policy_open_one_hots, policy_close_one_hots), dim=2)

                modified_action = self.check_balance_round2(state, modified_action, action_high, action_low,
                                                            ready_mask, closable_mask)
                return modified_action.detach().cpu().numpy()

            elif config.explore_type == 'reject':
                assert False, 'Not implemented'
            else:
                assert False, 'exploration type wrong! Get: {}'.format(config.explore_type)

        else:
            if self.config.parameters['only_power']:
                mu, log_std = policy[:, :generator_num], policy[:, generator_num:2*generator_num]
            else:
                mu, log_std = policy[:, :2*generator_num], policy[:, 2*generator_num:4*generator_num]
            # sigma = torch.exp(log_std)
            sigma = log_std
            # distr = SquashedNormal(mu, sigma + 5e-4)
            distr = SquashedNormal(mu, sigma)
            policy_action = distr.sample(torch.Size([n_policy_action + n_random_action + n_expert_action]))  # [n_pol, batchsize, a_dim]
            policy_action = policy_action.clip(-0.999, 0.999)
            policy_action = policy_action.permute(1, 0, 2)

            policy_open_one_hots, policy_close_one_hots = self.determine_open_close_one_hot(
                policy, n_policy_action+n_random_action+n_expert_action, add_explore_noise=False, is_root=is_root)
            policy_action = torch.cat((policy_action, policy_open_one_hots, policy_close_one_hots), dim=2)

            return policy_action.detach().cpu().numpy()


    def eval_q(self, obs, actions, attacker_actions, prev_r, to_plays=None):
        # with autocast():
        batch_shape = obs.size(0)
        action_num = actions.size(1)
        obs_expand = obs.reshape(obs.size(0), 1, obs.size(1)).repeat(1, actions.size(1), 1)
        obs_expand = obs_expand.reshape(obs_expand.size(0) * obs_expand.size(1), -1)

        attacker_actions = attacker_actions.reshape(batch_shape * action_num, -1)
        actions = actions.reshape(batch_shape * action_num, -1)

        h = self.encode(obs_expand)
        to_plays = np.array([to_plays for _ in range(action_num)])
        to_plays = to_plays.reshape(to_plays.shape[0] * to_plays.shape[1], -1).tolist()
        prev_r = prev_r.reshape(prev_r.size(0), 1, prev_r.size(1)).repeat(1, action_num, 1)
        prev_r = prev_r.reshape(prev_r.size(0) * prev_r.size(1), -1)

        next_v, r, _, _, _ = self.recurrent_inference(h, actions, attacker_actions, to_plays, prev_r)

        next_v = support_to_scalar(next_v, self.support_size, self.config.value_support_step)
        r = support_to_scalar(r, self.reward_support_size, self.reward_support_step)
        r = r.reshape(batch_shape, action_num, 1)
        next_v = next_v.reshape(batch_shape, action_num, 1)

        values = r + self.gamma * next_v

        return values.squeeze()

    # def eval_q(self, obs, actions, real_gen_ps=None, action_highs=None, action_lows=None, ready_masks=None, closable_masks=None):
    #     if len(actions.shape) == 2:
    #         actions = actions.reshape(1, *actions.shape)
    #
    #     # Obs shape = [BATCHSIZE, O_DIM]
    #     # Obs shape = [BATCHSIZE, N, A_DIM]
    #
    #     batch_shape = obs.size(0)
    #     num_actions = actions.size(1)
    #
    #     obs_expand = obs.reshape(obs.size(0), 1, obs.size(1)).repeat(1, actions.size(1), 1)
    #     # print('INTERIOR', obs_expand.shape, actions.shape)
    #
    #     obs_expand = obs_expand.reshape(obs_expand.size(0) * obs_expand.size(1), -1)
    #     actions = actions.reshape(actions.size(0) * actions.size(1), -1)
    #
    #     # print('INTERIOR_II', obs_expand.shape, actions.shape)
    #     h = self.encode(obs_expand)
    #     # print('H', h.shape)
    #     r = self.reward(h, actions)
    #     # print('R', r.shape)
    #     next_h = self.dynamics(h, actions)
    #     # print('NH', next_h.shape)
    #     next_v = self.value(next_h)
    #     next_v = support_to_scalar(next_v, self.support_size, self.config.value_support_step)
    #     # print('NV', next_v.shape)
    #     if self.config.multi_reward:
    #         reward = 0
    #         cnt = 0
    #         for coeff, item in zip(self.config.reward_coeffs, r):
    #             if cnt == 2 and self.config.ground_truth_running_cost_reward:
    #                 for i in range(actions.shape[1]):
    #                     real_root_actions = action_mapping(actions[:, i, :].cpu().numpy(), self.config, action_highs,
    #                                                        action_lows, ready_masks, closable_masks)
    #                     reward[:, i] += coeff * torch.from_numpy(calc_running_cost_rew(real_gen_ps+real_root_actions)).unsqueeze(1).float().to('cuda')
    #             else:
    #                 reward += coeff * support_to_scalar(item, self.reward_support_size, self.reward_support_step)
    #         r = reward
    #     else:
    #         r = support_to_scalar(r, self.reward_support_size, self.reward_support_step)
    #     r = r.reshape(batch_shape, num_actions, 1)
    #     next_v = next_v.reshape(batch_shape, num_actions, 1)
    #     # print('NV2', next_v.shape)
    #     assert len(next_v.shape) == 3, 'Next v error'.format(next_v.shape)
    #     assert len(r.shape) == 3, 'R shape error:{}'.format(r.shape)
    #     values = r + self.gamma * next_v
    #     # print('VAL', values)
    #     return values.squeeze()

    def encode(self, obs):
        # with autocast():
        h = self.rep_net(obs)
        return h

    def dynamics(self, hidden, action):
        return self.dyn_net(hidden, action)

    def dynamics_attacker(self, hidden, action):
        '''
        next_hidden = mask * hidden + (1 - mask) * predicted_hidden, mask = Identity(action is nothing)
        :param hidden:
        :param action:
        :return:
        '''
        mask = action[:, -1].reshape(hidden.shape[0], 1).repeat(1, hidden.shape[1])     # mask for no attack
        # action_modified = torch.where(action > 0)[-1].reshape(action.shape[0], 1) / self.config.attacker_action_dim
        # predicted_hidden = self.dyn_net_attacker(hidden, action_modified)
        predicted_hidden = self.dyn_net_attacker(hidden, action)
        next_hidden = mask * hidden + (1 - mask) * predicted_hidden
        return next_hidden

    def mask(self, hidden, action):
        return self.mask_net(torch.cat((hidden, action), dim=1))

    def expert(self, hidden):
        return self.expert_pi(hidden)

    def project(self, h, with_grad=True):
        h_proj = self.proj_net(h)

        if with_grad:
            return self.proj_pred_net(h_proj)

        else:
            return h_proj.detach()

    # def project_attacker(self, h, with_grad=True):
    #     h_proj = self.proj_net_attacker(h)
    #
    #     if with_grad:
    #         return self.proj_pred_net_attacker(h_proj)
    #
    #     else:
    #         return h_proj.detach()

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def initial_inference(self, observation):
        with autocast():
            h = self.encode(observation)
            pi = self.policy_no_split(h)
            attacker_pi = self.policy_attacker(h)
            if self.config.efficient_imitation:
                pi = torch.cat((pi, self.expert(h)), dim=-1)
            value = self.value(h)

            # reward equal to 0 for consistency
            reward = (torch.zeros(1, self.full_reward_support_size)
                      .scatter(1, torch.tensor([[self.full_reward_support_size // 2]]).long(), 1.0)
                      .repeat(len(observation), 1)
                      .to(observation.device))

        return (value, reward, pi, attacker_pi, h)

    def atanh(self, x):
        return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

    # @profile
    def determine_open_close_one_hot(self, policy, num, modified_action=None, ready_masks=None, closable_masks=None,
                                     add_explore_noise=False, is_root=False, ori_states=None):
        batch_size = policy.shape[0]
        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim
        if ready_masks is not None:
            ready_masks = torch.cat((ready_masks[:, settings.thermal_ids], ready_masks[:, -1:]), dim=1)
            closable_masks = torch.cat((closable_masks[:, settings.thermal_ids], closable_masks[:, -1:]), dim=1)
            if not self.config.parameters['only_power']:
                ready_masks = torch.cat((ready_masks[:, :generator_num], ready_masks[:, -1:]), dim=1)
                closable_masks = torch.cat((closable_masks[:, :generator_num], closable_masks[:, -1:]), dim=1)
            ready_masks = ready_masks.unsqueeze(1).repeat(1, num, 1)
            closable_masks = closable_masks.unsqueeze(1).repeat(1, num, 1)
        if ori_states is not None:
            delta_load_p = ori_states[:, 0]
            balance_up_redundency = ori_states[:, -1]
            balance_down_redundency = ori_states[:, -2]
            addition = 20
            redundency_adjust = -(1 - torch.sign(balance_up_redundency)) / 2 * (balance_up_redundency - addition) + \
                                (1 - torch.sign(balance_down_redundency)) / 2 * (balance_down_redundency - addition)

            modified_action = torch.cat((modified_action[:, :, :settings.balanced_id],
                                              torch.zeros_like(modified_action[:, :, :settings.balanced_id])[:, :, 0:1],
                                              modified_action[:, :, settings.balanced_id:]), dim=2)

            if self.config.parameters['only_power']:
                delta = delta_load_p.unsqueeze(1).repeat(1, num) - \
                        modified_action.sum(2) + \
                        redundency_adjust.unsqueeze(1).repeat(1, num)
            else:
                delta = delta_load_p.unsqueeze(1).repeat(1, num, 1) - \
                        modified_action[:, :, :generator_num].sum(2) + \
                        redundency_adjust.unsqueeze(1).repeat(1, num, 1)

            # print(f'determine delta={delta}, max={torch.max(torch.abs(delta)):.3f}, min={torch.min(torch.abs(delta)):.3f}')
            # import ipdb
            # ipdb.set_trace()

            delta = delta.unsqueeze(2).repeat(1, 1, one_hot_dim)

            gen_q = ori_states[:, -289:-235]
            # min_gen_q = settings.min_gen_q
            # max_gen_q = settings.max_gen_q
            # q_overload = (gen_q / max_gen_q > 1.0).float()
            # q_underload = (gen_q / min_gen_q < 1.0).float()
            nextstep_renewable_gen_p_max = ori_states[:, -20:-2].unsqueeze(1).repeat(1, num, 1)
            gen_p = ori_states[:, 1:55]
            load_p = ori_states[:, 55:55+91]
            next_load_p = ori_states[:, 55+91:55+91+91].unsqueeze(1).repeat(1, num, 1)
            renewable_gen_p = gen_p[:, settings.renewable_ids].unsqueeze(1).repeat(1, num, 1) + modified_action[:, :, settings.renewable_ids]
            renewable_consump_rate = renewable_gen_p.sum(-1) / nextstep_renewable_gen_p_max.sum(-1)
            sum_renewable_up_redundency = nextstep_renewable_gen_p_max.sum(-1) - renewable_gen_p.sum(-1)

            expand_gen_p = gen_p.unsqueeze(1).repeat(1, num, 1)
            # running_cost = calc_running_cost_rew(expand_gen_p, is_real=True)
            open_action_low = torch.zeros_like(modified_action)
            open_action_low[:, :, settings.renewable_ids] = nextstep_renewable_gen_p_max
            action_low = ori_states[:, -91:-38]
            # tmp = np.expand_dims(np.expand_dims(np.array((settings.max_gen_p[settings.balanced_id]+settings.min_gen_p[settings.balanced_id])/2-80), axis=0), axis=0).repeat(ori_states.shape[0], axis=0)
            tmp = np.array((settings.max_gen_p[settings.balanced_id]+settings.min_gen_p[settings.balanced_id])/2-80)
            tmp = torch.from_numpy(tmp).float().unsqueeze(0).repeat(batch_size, 1).to('cuda')
            action_low = torch.cat((action_low[:, :settings.balanced_id], tmp, action_low[:, settings.balanced_id:]), dim=1)
            action_low = action_low.unsqueeze(1).repeat(1, num, 1)
            open_action_low[:, :, settings.thermal_ids] = gen_p[:, settings.thermal_ids].unsqueeze(1).repeat(1, num, 1) + action_low[:, :, settings.thermal_ids]
            open_action_low[:, :, settings.balanced_id:settings.balanced_id+1] = \
                gen_p[:, settings.balanced_id:settings.balanced_id+1].unsqueeze(1).repeat(1, num, 1) + \
                action_low[:, :, settings.balanced_id:settings.balanced_id+1]

            close_mask = ((delta < -40).float()
                          + (renewable_consump_rate < 0.6 + 0.3 * random.random()).unsqueeze(2).repeat(1, 1, one_hot_dim).float()
                          # + (next_load_p.sum(-1) - open_action_low.sum(-1) < 80).unsqueeze(2).repeat(1, 1, one_hot_dim).float()
                          + (sum_renewable_up_redundency > 100).unsqueeze(2).repeat(1, 1, one_hot_dim).float()
                          # + (balance_down_redundency.unsqueeze(1).repeat(1, num, 1) < -20).float()
                          # + (q_underload.sum(-1) > 0).float()
                          ) #* (closable_masks.sum(-1, keepdims=True).repeat(1, 1, one_hot_dim) > 0).float() #* (ready_masks.sum(-1) > 0).float()
            close_mask = (close_mask > 0).float()
            open_mask = ((delta > 40).float()
                         # + (balance_up_redundency.unsqueeze(1).repeat(1, num, 1) < -20).float()
                         # + (delta_load_p.unsqueeze(1).repeat(1, num, 1) > 150).float()
                         # + (running_cost < -0.5).float()
                         # + (next_load_p.sum(-1) - open_action_low.sum(-1) < 30) * (closable_masks.sum(-1) == 0)
                         # + (q_overload.sum(-1) > 0).float()
                         ) #* (ready_masks.sum(-1, keepdims=True).repeat(1, 1, one_hot_dim) > 0).float()

        if ready_masks is not None:
            restricted_ready_masks = copy.deepcopy(ready_masks)
            restricted_ready_masks[:, :, -1] = 0
            restricted_closable_masks = copy.deepcopy(closable_masks)
            restricted_closable_masks[:, :, -1] = 0

        if self.config.parameters['only_power']:
            ori_mu, std, open_logits, close_logits = policy[:, :generator_num], \
                                                     policy[:, generator_num:2 * generator_num], \
                                                     policy[:, 2 * generator_num:2 * generator_num+one_hot_dim], \
                                                     policy[:, 2 * generator_num+one_hot_dim:]
        else:
            ori_mu, std, open_logits, close_logits = policy[:, :2*generator_num], policy[:, 2*generator_num:4 * generator_num], \
                                                     policy[:, 4 * generator_num:4*generator_num+one_hot_dim], \
                                                     policy[:, 4*generator_num+one_hot_dim:]

        min_gen_p = torch.from_numpy(np.append(np.asarray(settings.min_gen_p)[settings.thermal_ids], 0)).float().to('cuda')
        min_gen_p = min_gen_p.unsqueeze(0).repeat(batch_size, 1)
        min_gen_p = min_gen_p.unsqueeze(1).repeat(1, num, 1)

        open_logits = open_logits.unsqueeze(1).repeat(1, num, 1)
        close_logits = close_logits.unsqueeze(1).repeat(1, num, 1)

        if ori_states is not None:
            open_factor = F.softmax(-torch.abs(min_gen_p - delta) / 80, dim=2) * 50
            close_factor = F.softmax(-torch.abs(min_gen_p + delta) / 80, dim=2) * 50
            open_close_factor = torch.cat((open_factor, close_factor), dim=0)
        else:
            open_close_factor = torch.ones((min_gen_p.shape[0], min_gen_p.shape[1], min_gen_p.shape[2])).float().to('cuda')
            open_close_factor = open_close_factor.repeat(2, 1, 1)

        if add_explore_noise:
            if self.config.parameters['only_power']:
                noises = torch.from_numpy(np.random.dirichlet([self.config.root_dirichlet_alpha] * one_hot_dim, (2*open_logits.shape[0], num))).float().to('cuda')
            else:
                noises = torch.from_numpy(np.random.dirichlet([self.config.root_dirichlet_alpha] * one_hot_dim, (2*open_logits.shape[0], num))).float().to('cuda')
            frac = self.config.root_exploration_fraction

            priors = F.softmax(torch.cat((open_logits, close_logits), dim=0), dim=2)
            priors = priors * (1 - frac) + noises * frac
            if ori_states is not None:
                priors *= torch.cat((ready_masks, closable_masks), dim=0)
            priors *= open_close_factor
            one_hots = F.gumbel_softmax(priors, hard=True, dim=2)
            open_one_hots, close_one_hots = one_hots[:one_hots.shape[0] // 2], one_hots[one_hots.shape[0] // 2:]

            if ori_states is not None:
                restricted_priors = priors * torch.cat((restricted_ready_masks, restricted_closable_masks), dim=0)
                restricted_one_hots = F.gumbel_softmax(restricted_priors, hard=True, dim=2, tau=1e-4)
                restricted_open_one_hots, restricted_close_one_hots = restricted_one_hots[:one_hots.shape[0]//2], restricted_one_hots[one_hots.shape[0]//2:]

                open_one_hots = open_mask * restricted_open_one_hots + (1 - open_mask) * open_one_hots
                close_one_hots = close_mask * restricted_close_one_hots + (1 - close_mask) * close_one_hots

        else:
            priors = F.softmax(torch.cat((open_logits, close_logits), dim=0), dim=2)
            if ori_states is not None:
                priors *= torch.cat((ready_masks, closable_masks), dim=0)
            priors *= open_close_factor
            _, ids = torch.max(priors, dim=2)
            one_hots = F.one_hot(ids, num_classes=one_hot_dim).float()
            open_one_hots, close_one_hots = one_hots[:one_hots.shape[0]//2], one_hots[one_hots.shape[0]//2:]

            if ori_states is not None:
                restricted_priors = priors * torch.cat((restricted_ready_masks, restricted_closable_masks), dim=0)
                _, restricted_ids = torch.max(restricted_priors, dim=2)
                restricted_one_hots = F.one_hot(restricted_ids, num_classes=one_hot_dim).float()
                restricted_open_one_hots, restricted_close_one_hots = restricted_one_hots[:one_hots.shape[0] // 2], restricted_one_hots[one_hots.shape[0] // 2:]

                open_one_hots = open_mask * restricted_open_one_hots + (1 - open_mask) * open_one_hots
                close_one_hots = close_mask * restricted_close_one_hots + (1 - close_mask) * close_one_hots

        if is_root:
            open_one_hots *= ready_masks
            close_one_hots *= closable_masks
        #     # open_real_mask = np.expand_dims((delta > 10).astype(np.float32), axis=2).repeat(open_one_hots.shape[-1], axis=2)
        #     # close_real_mask = np.expand_dims((delta < -10).astype(np.float32), axis=2).repeat(open_one_hots.shape[-1], axis=2)
        #     # close_one_hots = close_one_hots * torch.from_numpy((1 - open_real_mask)).float().to('cuda')
        #     # open_one_hots = open_one_hots * torch.from_numpy((1 - close_real_mask)).float().to('cuda')
            sum_open_one_hots = open_one_hots[:, :, :-1].sum(2)
            open_one_hots[:, :, -1] = 1 - sum_open_one_hots
            sum_close_one_hots = close_one_hots[:, :, :-1].sum(2)
            close_one_hots[:, :, -1] = 1 - sum_close_one_hots

        return open_one_hots, close_one_hots


    def modify_policy(self, action, ready_masks, closable_masks, action_high, action_low, is_test=False):
        ori_mu = torch.cat((action[:, :settings.balanced_id], torch.zeros_like(action)[:, 0:1], action[:, settings.balanced_id:]), dim=1)
        mu = (ori_mu + torch.ones_like(ori_mu)) / (2 * torch.ones_like(ori_mu)) * (
                action_high - action_low) + action_low
        modified_mu = mu * (torch.ones_like(ready_masks) - ready_masks)[:, :-1] * (torch.ones_like(closable_masks) - closable_masks)[:, :-1]

        modified_mu += torch.clamp(mu * closable_masks[:, :-1], 0, 10000)
        modified_mu = torch.cat((modified_mu[:, :settings.balanced_id], modified_mu[:, settings.balanced_id+1:]), dim=1)
        return modified_mu

    def check_balance(self, state, sampled_actions, action_high, action_low, ready_mask, closable_mask, norm_action=True):

        delta_load_p = state[:, 0]
        balance_up_redundency = state[:, -1]
        balance_down_redundency = state[:, -2]

        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim
        batch_size = state.shape[0]
        num = sampled_actions.shape[1]

        addition = 20
        tmp = 20
        redundency_adjust = -(1 - torch.sign(balance_up_redundency)) / 2 * (balance_up_redundency
                                                                                    - tmp
                                                                                    # - tmp * random.random()
                                                                                    ) + \
                            (1 - torch.sign(balance_down_redundency)) / 2 * (balance_down_redundency
                                                                                    - tmp
                                                                                     # - tmp * random.random()
                                                                                     )

        mask = ((action_high != 0).float() + (action_low != 0).float()) * \
               (1 - closable_mask[:, :-1]) * \
               (1 - ready_mask[:, :-1])     # represent adjustable generators
        # thermal_mask = torch.zeros_like(mask)
        # thermal_mask[:, settings.thermal_ids] = 1
        # mask *= thermal_mask  # only readjust thermal units
        mask = (mask > 0).float()
        if not self.config.parameters['only_power']:
            power_mask = torch.zeros_like(action_high)
            power_mask[:, :generator_num] = 1   # represent active power control dimensions
            mask *= power_mask
        # inv_mask = torch.ones_like(mask) - mask  # represent non-adjustable generators, voltage control, closed or balance or open_gen_logit

        if norm_action:
            real_actions = torch.zeros_like(sampled_actions)
            for i in range(sampled_actions.shape[1]):
                real_actions[:, i, :] = self.modify_policy(sampled_actions[:, i, :],
                                                           ready_mask,
                                                           closable_mask,
                                                           action_high,
                                                           action_low)
        else:
            real_actions = sampled_actions

        if self.config.parameters['only_power']:
            delta = delta_load_p.unsqueeze(1).repeat(1, num) - \
                    real_actions.sum(2) + \
                    redundency_adjust.unsqueeze(1).repeat(1, num)
        else:
            delta = delta_load_p.unsqueeze(1).repeat(1, num, 1) - \
                    real_actions[:, :, :generator_num].sum(2) + \
                    redundency_adjust.unsqueeze(1).repeat(1, num, 1)

        # print(f'check bal delta={delta}, max={torch.max(torch.abs(delta)):.3f}, min={torch.min(torch.abs(delta)):.3f}')
        # import ipdb
        # ipdb.set_trace()
        change_mask = (torch.abs(delta) > 30)
        real_actions = torch.cat((
            real_actions[:, :, :settings.balanced_id],
            torch.zeros_like(real_actions[:, :, :settings.balanced_id])[:, :, 0:1],
            real_actions[:, :, settings.balanced_id:]
        ), dim=2)
        change_mask = change_mask.unsqueeze(2).repeat(1, 1, real_actions.shape[2])
        delta = delta.unsqueeze(2).repeat(1, 1, real_actions.shape[2])
        action_high = action_high.unsqueeze(1).repeat(1, num, 1)
        action_low = action_low.unsqueeze(1).repeat(1, num, 1)
        mask = mask.unsqueeze(1).repeat(1, num, 1)

        upgrade_redundency = (action_high - real_actions) * mask
        downgrade_redundency = (real_actions - action_low) * mask

        upgrade_redundency_sum = upgrade_redundency.sum(2, keepdims=True).repeat(1, 1, upgrade_redundency.shape[-1])
        downgrade_redundency_sum = downgrade_redundency.sum(2, keepdims=True).repeat(1, 1, downgrade_redundency.shape[-1])

        modification = (1 + torch.sign(delta)) / 2 * delta * upgrade_redundency / (upgrade_redundency_sum + 1e-3) + \
                       (1 - torch.sign(delta)) / 2 * delta * downgrade_redundency / (downgrade_redundency_sum + 1e-3)

        real_actions += modification * change_mask
        real_actions = torch.cat((real_actions[:, :, :settings.balanced_id], real_actions[:, :, settings.balanced_id+1:]), dim=2)
        return real_actions


    def check_balance_round2(self, state, sampled_actions, action_high, action_low, ready_mask, closable_mask, norm_action=True):
        delta_load_p = state[:, 0]
        balance_up_redundency = state[:, -1]
        balance_down_redundency = state[:, -2]

        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim
        batch_size = state.shape[0]
        num = sampled_actions.shape[1]

        open_one_hot = sampled_actions[:, :, generator_num:generator_num+one_hot_dim]
        close_one_hot = sampled_actions[:, :, generator_num+one_hot_dim:]

        addition = 20
        tmp = 20
        redundency_adjust = -(1 - torch.sign(balance_up_redundency)) / 2 * (balance_up_redundency - tmp) + \
                            (1 - torch.sign(balance_down_redundency)) / 2 * (balance_down_redundency - tmp)
        mask = ((action_high != 0).float() + (action_low != 0).float()) * \
               (1 - closable_mask[:, :-1]) * \
               (1 - ready_mask[:, :-1])     # represent adjustable generators
        # thermal_mask = torch.zeros_like(mask)
        # thermal_mask[:, settings.thermal_ids] = 1
        # mask *= thermal_mask    # only readjust thermal units
        mask = (mask > 0).float()
        if not self.config.parameters['only_power']:
            power_mask = torch.zeros_like(action_high)
            power_mask[:, :generator_num] = 1   # represent active power control dimensions
            mask *= power_mask


        real_actions = torch.cat((sampled_actions[:, :, :settings.balanced_id],
                                  torch.zeros_like(sampled_actions[:, :, :settings.balanced_id])[:, :, 0:1],
                                  sampled_actions[:, :, settings.balanced_id:generator_num]), dim=2)

        min_gen_p = torch.from_numpy(np.append(np.array(settings.min_gen_p)[settings.thermal_ids], 0)).float()
        min_gen_p = min_gen_p.unsqueeze(0).repeat(batch_size, 1).unsqueeze(1).repeat(1, num, 1).to('cuda')
        if self.config.parameters['only_power']:
            delta = delta_load_p.unsqueeze(1).repeat(1, num) - \
                    real_actions.sum(2) + \
                    redundency_adjust.unsqueeze(1).repeat(1, num) + \
                    (-min_gen_p * open_one_hot + min_gen_p * close_one_hot).sum(2)
        else:
            delta = delta_load_p.unsqueeze(1).repeat(1, num, 1) - \
                    real_actions[:, :, :generator_num].sum(2) + \
                    redundency_adjust.unsqueeze(1).repeat(1, num, 1) + \
                    (-min_gen_p * open_one_hot + min_gen_p * close_one_hot).sum(2)

        # print(f'check bal round2 delta={delta}, max={torch.max(torch.abs(delta)):.3f}, min={torch.min(torch.abs(delta)):.3f}')
        change_mask = (torch.abs(delta) > 30).float()
        change_mask = change_mask.unsqueeze(2).repeat(1, 1, real_actions.shape[2])
        delta = delta.unsqueeze(2).repeat(1, 1, real_actions.shape[-1])
        action_high = action_high.unsqueeze(1).repeat(1, num, 1)
        action_low = action_low.unsqueeze(1).repeat(1, num, 1)
        mask = mask.unsqueeze(1).repeat(1, num, 1)


        upgrade_redundency = (action_high - real_actions) * mask
        downgrade_redundency = (real_actions - action_low) * mask

        upgrade_redundency_sum = upgrade_redundency.sum(2, keepdims=True).repeat(1, 1, upgrade_redundency.shape[-1])
        downgrade_redundency_sum = downgrade_redundency.sum(2, keepdims=True).repeat(1, 1, downgrade_redundency.shape[-1])
        modification = (1 + torch.sign(delta)) / 2 * delta * upgrade_redundency / (upgrade_redundency_sum + 1e-3) + \
                       (1 - torch.sign(delta)) / 2 * delta * downgrade_redundency / (downgrade_redundency_sum + 1e-3)

        modified_actions = real_actions + modification * change_mask

        modified_sampled_actions = (modified_actions - action_low) / (action_high - action_low + 1e-3) * 2 - 1
        modified_sampled_actions = torch.cat((modified_sampled_actions[:, :, :settings.balanced_id],
                                                   modified_sampled_actions[:, :, settings.balanced_id + 1:]), dim=2)

        mask = torch.cat((mask[:, :, :settings.balanced_id], mask[:, :, settings.balanced_id + 1:]), dim=2)
        inv_mask = torch.ones_like(mask) - mask  # represent non-adjustable generators, voltage control, closed or balance or open_gen_logit
        modified_sampled_actions = modified_sampled_actions * mask + sampled_actions[:, :, :generator_num] * inv_mask

        modified_sampled_actions = modified_sampled_actions.clip(-0.999, 0.999)
        return torch.cat((modified_sampled_actions, open_one_hot, close_one_hot), dim=2)

    # @profile
    def recurrent_inference(self, h, action, attacker_action, to_plays, prev_r):
        """
        :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
        :param action: shape: [Batchsize, Action_Dim]
        :return:
        """
        # attacker_action_modified = torch.where(attacker_action > 0)[-1].reshape(attacker_action.shape[0], 1) / self.config.attacker_action_dim
        # open_one_hots = action[:, self.config.generator_num:self.config.generator_num+self.config.one_hot_dim]
        # close_one_hots = action[:, self.config.generator_num+self.config.one_hot_dim:]
        # open_one_hots = torch.where(open_one_hots > 0)[-1].reshape(open_one_hots.shape[0], 1) / self.config.one_hot_dim
        # close_one_hots = torch.where(close_one_hots > 0)[-1].reshape(close_one_hots.shape[0], 1) / self.config.one_hot_dim
        # action = torch.cat((action[:, :self.config.generator_num], open_one_hots, close_one_hots), dim=-1)
        with autocast():
            r = self.reward(h, action)
            h = self.dynamics(h, action)
            to_plays_r = torch.from_numpy(np.asarray(to_plays)).float().to('cuda')
            to_plays_r = to_plays_r.reshape(to_plays_r.shape[0], 1).repeat(1, r.shape[1])
            attacker_r = self.reward_attacker(h, attacker_action, prev_r)
            r = (1 - to_plays_r) * r + to_plays_r * attacker_r
            to_plays_h = torch.from_numpy(np.asarray(to_plays)).float().to('cuda')
            to_plays_h = to_plays_h.reshape(to_plays_h.shape[0], 1).repeat(1, h.shape[1])
            attacker_h = self.dynamics_attacker(h, attacker_action)
            h = (1 - to_plays_h) * h + to_plays_h * attacker_h
            pi = self.policy_no_split(h)
            attacker_pi = self.policy_attacker(h)
            if self.config.efficient_imitation:
                pi = torch.cat((pi, self.expert(h)), dim=-1)
            value = self.value(h)
        return value, r, pi, attacker_pi, h