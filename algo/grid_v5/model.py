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
    def __init__(self, s_shape, a_shape, dyn_model):
        super(ResidualForwardModel, self).__init__()
        self.mlp = mlp(s_shape + a_shape, dyn_model, s_shape, activation=nn.LeakyReLU, use_bn=False)
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
        super(MultipleRewardModel, self).__init__()
        self.config = config
        self.use_bn = config.use_bn
        self.rew_dyn_act_dim = config.rew_dyn_act_dim
        self.hidden_shape = config.mlp_hidden_shape
        self.rew_net_shape = config.mlp_rew_shape

        self.reward_support_size = config.reward_support_size
        self.reward_support_step = config.reward_support_step
        self.full_reward_support_size = 2 * config.reward_support_size + 1

        self.line_of_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                   self.full_reward_support_size,
                                   activation=nn.LeakyReLU, use_bn=self.use_bn, init_zero=config.init_zero)
        self.renewable_consump_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                             self.full_reward_support_size,
                                             activation=nn.LeakyReLU, use_bn=self.use_bn, init_zero=config.init_zero)
        self.run_cost_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                             self.full_reward_support_size,
                                             activation=nn.LeakyReLU, use_bn=self.use_bn, init_zero=config.init_zero)
        self.bal_gen_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                             self.full_reward_support_size,
                                             activation=nn.LeakyReLU, use_bn=self.use_bn, init_zero=config.init_zero)
        self.reac_pow_rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape,
                                             self.full_reward_support_size,
                                             activation=nn.LeakyReLU, use_bn=self.use_bn, init_zero=config.init_zero)

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

        '''
            Models
        '''
        self.rep_net = mlp(self.obs_shape * self.n_stacked_obs, self.rep_net_shape, self.hidden_shape, activation=nn.LeakyReLU,
                           # output_activation=nn.LeakyReLU,
                           use_bn=self.use_bn)
        # if self.config.parameters['only_power']:
        self.dyn_net = ResidualForwardModel(self.hidden_shape, self.rew_dyn_act_dim, self.dyn_shape)
        if self.config.multi_reward:
            self.rew_net = MultipleRewardModel(config)
        else:
            self.rew_net = mlp(self.hidden_shape + self.rew_dyn_act_dim, self.rew_net_shape, self.full_reward_support_size,
                activation=nn.LeakyReLU, use_bn=self.use_bn, init_zero=config.init_zero)

        self.pi_net = mlp(self.hidden_shape, self.pi_net_shape,
                          self.policy_act_dim,
                          activation=nn.LeakyReLU,
                          # output_activation=nn.Tanh,    # TODO: move Tanh in output layer to modify_policy function
                          use_bn=self.use_bn,
                          init_zero=config.init_zero
                          )


        self.val_net = mlp(self.hidden_shape, self.val_net_shape, config.support_size * 2 + 1, activation=nn.LeakyReLU,
                           use_bn=self.use_bn, init_zero=config.init_zero)

        self.softmax = nn.Softmax(dim=1)

        if self.config.efficient_imitation:
            self.expert_pi = mlp(self.hidden_shape, self.expert_pi_net_shape,
                          self.policy_act_dim,
                          activation=nn.LeakyReLU,
                          # output_activation=nn.Tanh,    # TODO: move Tanh in output layer to modify_policy function
                          use_bn=self.use_bn,
                          init_zero=config.init_zero
                          )

        # self.attacker_pi_net = mlp(self.hidden_shape, self.attacker_pi_net_shape, self.attacker_action_shape, use_bn=self.use_bn, init_zero=config.init_zero)
        #
        # self.attacker_map_net = mlp(self.attacker_action_shape, self.attacker_map_net_shape, 2 * self.action_shape, use_bn=self.use_bn, init_zero=config.init_zero)
        # self.pi_net = nn.Sequential(
        #     nn.Linear(self.hidden_shape, self.hidden_shape),
        #     nn.BatchNorm1d(self.hidden_shape),
        #     nn.LeakyReLU(),
        #     nn.Linear(self.hidden_shape, 2*self.action_shape)
        # )

        self.proj_net = mlp(self.hidden_shape, self.proj_net_shape, self.proj_shape, use_bn=self.use_bn)
        self.proj_pred_net = mlp(self.proj_shape, self.proj_pred_shape, self.proj_shape, use_bn=self.use_bn)
        self.time_step = 0
        # print(self.use_bn)


    def discrim(self, hidden, action):
        return self.dis_net(hidden, action)

    def value(self, hidden):
        return self.val_net(hidden)

    def value_obs(self, obs):
        hidden = self.rep_net(obs)
        return self.value(hidden)

    # @profile
    def policy_no_split(self, hidden):
        pi = self.pi_net(hidden)
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
                sigma = torch.exp(log_std)
                distr = SquashedNormal(mu, sigma + 5e-4)

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
                    expert_sigma = torch.exp(expert_log_std)
                    expert_distr = SquashedNormal(expert_mu, expert_sigma + 5e-4)

                if not is_test:
                    policy_action = distr.sample(torch.Size([n_policy_action]))  # [n_pol, batchsize, a_dim]
                    policy_action = policy_action.clip(-0.999, 0.999)
                    policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

                    if n_random_action > 0:
                        random_action = distr.sample(torch.Size([n_random_action//2]))  # [n_pol, batchsize, a_dim]
                        random_action += torch.randn_like(random_action) * config.explore_scale
                        # random_action = torch.nn.functional.tanh(random_action)
                        new_distr = SquashedNormal(mu, 3*sigma+5e-4)
                        random_action_new = new_distr.sample(torch.Size([n_random_action//2]))
                        random_action = torch.cat((random_action, random_action_new), dim=0)
                        random_action = random_action.clip(-0.999, 0.999)
                        random_action = torch_utils.tensor_to_numpy(random_action.permute(1, 0, 2))
                    else:
                        random_action = None

                    if n_random_action > 0:
                        policy_action = np.concatenate([policy_action, random_action], axis=1)

                    if expert_policy is not None:
                        if n_expert_action > 0:
                            expert_action = expert_distr.sample(torch.Size([n_expert_action]))
                            expert_action = expert_action.clip(-0.999, 0.999)
                            expert_action = torch_utils.tensor_to_numpy(expert_action.permute(1, 0, 2))
                            policy_action = np.concatenate([policy_action, expert_action], axis=1)
                else:
                    policy_action = distr.sample(torch.Size([n_policy_action + n_random_action]))
                    policy_action = policy_action.clip(-0.999, 0.999)
                    policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

                    if expert_policy is not None:
                        if n_expert_action > 0:
                            expert_action = expert_distr.sample(torch.Size([n_expert_action]))
                            expert_action = expert_action.clip(-0.999, 0.999)
                            expert_action = torch_utils.tensor_to_numpy(expert_action.permute(1, 0, 2))
                            policy_action = np.concatenate([policy_action, expert_action], axis=1)

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
                                                          ready_mask, closable_mask, action_high, action_low,
                                                          add_explore_noise=False, is_root=is_root, ori_states=state)
                    policy_open_one_hots_random, policy_close_one_hots_random = \
                        self.determine_open_close_one_hot(policy, n_random_action, modified_action[:, n_policy_action:n_policy_action+n_random_action, :],
                                                          ready_mask, closable_mask, action_high, action_low,
                                                          add_explore_noise=True, is_root=is_root, ori_states=state)
                    if expert_policy is not None:
                        policy_open_one_hots_expert, policy_close_one_hots_expert = self.determine_open_close_one_hot(
                            expert_policy, n_expert_action, modified_action[:, n_policy_action+n_random_action:, :],
                            ready_mask, closable_mask, action_high, action_low,
                            add_explore_noise=False, is_root=is_root, ori_states=state)
                        policy_open_one_hots = np.concatenate((policy_open_one_hots_real, policy_open_one_hots_random, policy_open_one_hots_expert),
                                                              axis=1)
                        policy_close_one_hots = np.concatenate(
                            (policy_close_one_hots_real, policy_close_one_hots_random, policy_close_one_hots_expert), axis=1)
                    else:
                        policy_open_one_hots = np.concatenate((policy_open_one_hots_real, policy_open_one_hots_random), axis=1)
                        policy_close_one_hots = np.concatenate((policy_close_one_hots_real, policy_close_one_hots_random), axis=1)

                    modified_action = np.concatenate((modified_action, policy_open_one_hots, policy_close_one_hots),
                                                     axis=2)
                else:
                    policy_open_one_hots, policy_close_one_hots = \
                        self.determine_open_close_one_hot(policy, n_policy_action+n_random_action,
                                                          modified_action[:, :n_policy_action+n_random_action, :],
                                                          ready_mask, closable_mask, action_high, action_low,
                                                          add_explore_noise=False, is_root=is_root, ori_states=state)
                    if expert_policy is not None:
                        policy_open_one_hots_expert, policy_close_one_hots_expert = self.determine_open_close_one_hot(
                            expert_policy, n_expert_action, modified_action[:, n_policy_action + n_random_action:, :],
                            ready_mask, closable_mask, action_high, action_low,
                            add_explore_noise=False, is_root=is_root, ori_states=state)
                        policy_open_one_hots = np.concatenate((policy_open_one_hots, policy_open_one_hots_expert), axis=1)
                        policy_close_one_hots = np.concatenate((policy_close_one_hots, policy_close_one_hots_expert), axis=1)

                    modified_action = np.concatenate((modified_action, policy_open_one_hots, policy_close_one_hots), axis=2)

                modified_action = self.check_balance_round2(state, modified_action, action_high, action_low,
                                                            ready_mask, closable_mask)

                return modified_action

            elif config.explore_type == 'reject':
                assert False, 'Not implemented'

            else:
                assert False, 'exploration type wrong! Get: {}'.format(config.explore_type)

        else:
            if self.config.parameters['only_power']:
                mu, log_std = policy[:, :generator_num], policy[:, generator_num:2*generator_num]
            else:
                mu, log_std = policy[:, :2*generator_num], policy[:, 2*generator_num:4*generator_num]
            sigma = torch.exp(log_std)
            distr = SquashedNormal(mu, sigma + 5e-4)
            policy_action = distr.sample(torch.Size([n_policy_action + n_random_action + n_expert_action]))  # [n_pol, batchsize, a_dim]
            policy_action = policy_action.clip(-0.999, 0.999)
            policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

            policy_open_one_hots, policy_close_one_hots = self.determine_open_close_one_hot(
                policy, n_policy_action+n_random_action+n_expert_action, add_explore_noise=False, is_root=is_root)
            policy_action = np.concatenate((policy_action, policy_open_one_hots, policy_close_one_hots), axis=2)

            return policy_action

    def sample_mixed_actions_v2(self, policy, config, is_root,
                             state=None, action_high=None, action_low=None, ready_mask=None, closable_mask=None, is_test=False,
                                # expert_policy=None,
                                train_steps=0):
        n_batchsize = policy.shape[0]
        n_policy_action = config.mcts_num_policy_samples
        n_random_action = config.mcts_num_random_samples
        n_expert_action = config.mcts_num_expert_samples

        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim

        if self.config.efficient_imitation:
            policy, expert_policy = torch.chunk(policy, 2, dim=-1)
        else:
            expert_policy = None

        if is_root:
            # RETURNING, [BATCHSIZE, N, ACTION_DIM]

            if config.explore_type == 'manual':
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
                sigma = torch.exp(log_std)
                distr = SquashedNormal(mu, sigma + 5e-4)

                if expert_policy is not None:
                    if self.config.parameters['only_power']:
                        expert_mu, expert_log_std, expert_open_logits, expert_close_logits = expert_policy[:, :generator_num], \
                                                                                             expert_policy[:, generator_num:2*generator_num], \
                                                                                             expert_policy[:, 2 * generator_num:2 * generator_num + one_hot_dim], \
                                                                                             expert_policy[:, 2 * generator_num + one_hot_dim:]
                    else:
                        expert_mu, expert_log_std, expert_open_logits, expert_close_logits = expert_policy[:, :2 * generator_num], \
                                                                                             expert_policy[:, 2 * generator_num:4 * generator_num], \
                                                                                             expert_policy[:, 4 * generator_num:4 * generator_num + one_hot_dim], \
                                                                                             expert_policy[4 * generator_num + one_hot_dim:]
                    expert_sigma = torch.exp(expert_log_std)
                    expert_distr = SquashedNormal(expert_mu, expert_sigma + 5e-4)

                if not is_test:
                    policy_action = distr.sample(torch.Size([n_policy_action]))  # [n_pol, batchsize, a_dim]
                    policy_action = policy_action.clip(-0.999, 0.999)
                    policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

                    if n_random_action > 0:
                        random_action = distr.sample(torch.Size([n_random_action//2]))  # [n_pol, batchsize, a_dim]
                        random_action += torch.randn_like(random_action) * config.explore_scale
                        # random_action = torch.nn.functional.tanh(random_action)
                        new_distr = SquashedNormal(mu, 3*sigma+5e-4)
                        random_action_new = new_distr.sample(torch.Size([n_random_action//2]))
                        random_action = torch.cat((random_action, random_action_new), dim=0)
                        random_action = random_action.clip(-0.999, 0.999)
                        random_action = torch_utils.tensor_to_numpy(random_action.permute(1, 0, 2))
                    else:
                        random_action = None

                    if n_random_action > 0:
                        # policy_action = torch.cat((policy_action, random_action), dim=1)
                        policy_action = np.concatenate([policy_action, random_action], axis=1)

                    if expert_policy is not None:
                        if n_expert_action > 0:
                            expert_action = expert_distr.sample(torch.Size([n_expert_action]))
                            expert_action = expert_action.clip(-0.999, 0.999)
                            expert_action = torch_utils.tensor_to_numpy(expert_action.permute(1, 0, 2))
                            policy_action = np.concatenate([policy_action, expert_action], axis=1)
                else:
                    policy_action = distr.sample(torch.Size([n_policy_action + n_random_action]))
                    policy_action = policy_action.clip(-0.999, 0.999)
                    policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

                    if expert_policy is not None:
                        if n_expert_action > 0:
                            expert_action = expert_distr.sample(torch.Size([n_expert_action]))
                            expert_action = expert_action.clip(-0.999, 0.999)
                            expert_action = torch_utils.tensor_to_numpy(expert_action.permute(1, 0, 2))
                            policy_action = np.concatenate([policy_action, expert_action], axis=1)

                if not is_test:
                    # policy_open_one_hots, policy_close_one_hots = self.determine_open_close_one_hot(policy, n_policy_action+n_random_action, modified_action, ready_mask, closable_mask, add_explore_noise=True, is_root=is_root, ori_states=state)
                    policy_open_one_hots_real, policy_close_one_hots_real = self.determine_open_close_one_hot_v2(policy, n_policy_action, policy_action[:, :n_policy_action, :], ready_mask, closable_mask, add_explore_noise=False, is_root=is_root, ori_states=state)
                    policy_open_one_hots_random, policy_close_one_hots_random = self.determine_open_close_one_hot_v2(policy, n_random_action, policy_action[:, n_policy_action:n_policy_action+n_random_action, :], ready_mask, closable_mask, add_explore_noise=True, is_root=is_root, ori_states=state)
                    if expert_policy is not None:
                        policy_open_one_hots_expert, policy_close_one_hots_expert = self.determine_open_close_one_hot_v2(expert_policy, n_expert_action, policy_action[:, n_policy_action+n_random_action:, :], ready_mask, closable_mask, add_explore_noise=False, is_root=is_root, ori_states=state)
                        policy_open_one_hots = np.concatenate((policy_open_one_hots_real, policy_open_one_hots_random, policy_open_one_hots_expert), axis=1)
                        policy_close_one_hots = np.concatenate((policy_close_one_hots_real, policy_close_one_hots_random, policy_close_one_hots_expert), axis=1)
                    else:
                        policy_open_one_hots = np.concatenate((policy_open_one_hots_real, policy_open_one_hots_random), axis=1)
                        policy_close_one_hots = np.concatenate((policy_close_one_hots_real, policy_close_one_hots_random), axis=1)

                    modified_action = np.concatenate((policy_action, policy_open_one_hots, policy_close_one_hots),
                                                     axis=2)
                else:
                    policy_open_one_hots, policy_close_one_hots = self.determine_open_close_one_hot_v2(policy, n_policy_action+n_random_action, policy_action,
                                                                                                      ready_mask,
                                                                                                      closable_mask,
                                                                                                      add_explore_noise=False,
                                                                                                      is_root=is_root,
                                                                                                      ori_states=state
                                                                                                    )
                    if expert_policy is not None:
                        policy_open_one_hots_expert, policy_close_one_hots_expert = self.determine_open_close_one_hot_v2(expert_policy,
                                                                                                           n_expert_action,
                                                                                                           policy_action,
                                                                                                           ready_mask,
                                                                                                           closable_mask,
                                                                                                           add_explore_noise=False,
                                                                                                           is_root=is_root,
                                                                                                           ori_states=state
                                                                                                           )
                        policy_open_one_hots = np.concatenate((policy_open_one_hots, policy_open_one_hots_expert),
                                                              axis=1)
                        policy_close_one_hots = np.concatenate((policy_close_one_hots, policy_close_one_hots_expert),
                                                               axis=1)
                    modified_action = np.concatenate((policy_action, policy_open_one_hots, policy_close_one_hots), axis=2)

                modified_action = self.check_balance_v2(
                    state,
                    modified_action,
                    action_high,
                    action_low,
                    ready_mask,
                    closable_mask)

                modified_action = np.concatenate((modified_action, policy_open_one_hots, policy_close_one_hots), axis=2)

                return modified_action

            elif config.explore_type == 'reject':
                assert False, 'Not implemented'

            else:
                assert False, 'exploration type wrong! Get: {}'.format(config.explore_type)

        else:
            if self.config.parameters['only_power']:
                mu, log_std = policy[:, :generator_num], policy[:, generator_num:2*generator_num]
            else:
                mu, log_std = policy[:, :2*generator_num], policy[:, 2*generator_num:4*generator_num]

            sigma = torch.exp(log_std)
            distr = SquashedNormal(mu, sigma + 5e-4)
            policy_action = distr.sample(torch.Size([n_policy_action + n_random_action]))  # [n_pol, batchsize, a_dim]
            policy_action = policy_action.clip(-0.999, 0.999)
            policy_action = torch_utils.tensor_to_numpy(policy_action.permute(1, 0, 2))

            if expert_policy is not None:
                if self.config.parameters['only_power']:
                    expert_mu, expert_log_std = expert_policy[:, :generator_num], expert_policy[:, generator_num:2 * generator_num],
                else:
                    expert_mu, expert_log_std = expert_policy[:, :2 * generator_num], expert_policy[:, 2 * generator_num:4 * generator_num]
                expert_sigma = torch.exp(expert_log_std)
                expert_distr = SquashedNormal(expert_mu, expert_sigma + 5e-4)
                if n_expert_action > 0:
                    expert_action = expert_distr.sample(torch.Size([n_expert_action]))
                    expert_action = expert_action.clip(-0.999, 0.999)
                    expert_action = torch_utils.tensor_to_numpy(expert_action.permute(1, 0, 2))
                    policy_action = np.concatenate([policy_action, expert_action], axis=1)

            policy_open_one_hots, policy_close_one_hots = self.determine_open_close_one_hot_v2(policy, n_policy_action+n_random_action, add_explore_noise=False, is_root=is_root)
            if expert_policy is not None:
                policy_open_one_hots_expert, policy_close_one_hots_expert = self.determine_open_close_one_hot_v2(
                    expert_policy,
                    n_expert_action,
                    add_explore_noise=False,
                    is_root=is_root
                    )
                policy_open_one_hots = np.concatenate((policy_open_one_hots, policy_open_one_hots_expert),
                                                      axis=1)
                policy_close_one_hots = np.concatenate((policy_close_one_hots, policy_close_one_hots_expert),
                                                       axis=1)
            policy_action = np.concatenate((policy_action, policy_open_one_hots, policy_close_one_hots), axis=2)
            # print('non_root_shape', policy_action.shape)
            return policy_action

    def eval_q(self, obs, actions, real_gen_ps=None, action_highs=None, action_lows=None, ready_masks=None, closable_masks=None):
        if len(obs.shape) == 2:
            if len(actions.shape) == 2:
                actions = actions.reshape(1, *actions.shape)

            # Obs shape = [BATCHSIZE, O_DIM]
            # Obs shape = [BATCHSIZE, N, A_DIM]

            batch_shape = obs.size(0)
            num_actions = actions.size(1)

            obs_expand = obs.reshape(obs.size(0), 1, obs.size(1)).repeat(1, actions.size(1), 1)
            # print('INTERIOR', obs_expand.shape, actions.shape)

            obs_expand = obs_expand.reshape(obs_expand.size(0) * obs_expand.size(1), -1)
            actions = actions.reshape(actions.size(0) * actions.size(1), -1)

            # print('INTERIOR_II', obs_expand.shape, actions.shape)
            h = self.encode(obs_expand)
            # print('H', h.shape)
            r = self.reward(h, actions)
            # print('R', r.shape)
            next_h = self.dynamics(h, actions)
            # print('NH', next_h.shape)
            next_v = self.value(next_h)
            next_v = support_to_scalar(next_v, self.support_size, self.config.value_support_step)
            # print('NV', next_v.shape)
            if self.config.multi_reward:
                reward = 0
                cnt = 0
                for coeff, item in zip(self.config.reward_coeffs, r):
                    if cnt == 2 and self.config.ground_truth_running_cost_reward:
                        for i in range(actions.shape[1]):
                            real_root_actions = action_mapping(actions[:, i, :].cpu().numpy(), self.config, action_highs,
                                                               action_lows, ready_masks, closable_masks)
                            reward[:, i] += coeff * torch.from_numpy(calc_running_cost_rew(real_gen_ps+real_root_actions)).unsqueeze(1).float().to('cuda')
                    else:
                        reward += coeff * support_to_scalar(item, self.reward_support_size, self.reward_support_step)
                r = reward
            else:
                r = support_to_scalar(r, self.reward_support_size, self.reward_support_step)
            r = r.reshape(batch_shape, num_actions, 1)
            next_v = next_v.reshape(batch_shape, num_actions, 1)
            # print('NV2', next_v.shape)
            assert len(next_v.shape) == 3, 'Next v error'.format(next_v.shape)
            assert len(r.shape) == 3, 'R shape error:{}'.format(r.shape)
            values = r + self.gamma * next_v
            # print('VAL', values)
            return values.squeeze()

        elif len(obs.shape) == 1:
            # Obs shape = [O_DIM]
            # Obs shape = [N, A_DIM]
            obs_expand = obs.reshape(1, -1).repeat(actions.size(0), 1)
            h = self.encode(obs_expand)
            r = self.reward(h, actions)
            if self.config.multi_reward:
                reward = 0
                cnt = 0
                for coeff, item in zip(self.config.reward_coeffs, r):
                    if cnt == 2 and self.config.ground_truth_running_cost_reward:
                        for i in range(actions.shape[1]):
                            real_root_actions = action_mapping(actions[:, i, :].cpu().numpy(), self.config, action_highs,
                                                               action_lows, ready_masks, closable_masks)
                            reward[:, i] += coeff * torch.from_numpy(calc_running_cost_rew(real_gen_ps+real_root_actions)).unsqueeze(1).float().to('cuda')
                    else:
                        reward += coeff * support_to_scalar(item, self.reward_support_size, self.reward_support_step)
                r = reward
            else:
                r = support_to_scalar(r, self.reward_support_size, self.reward_support_step)

            next_h = self.dynamics(h, actions)
            next_v = self.value(next_h)
            next_v = support_to_scalar(next_v, self.support_size, self.config.value_support_step)

            assert len(next_v.shape) == 2, 'Next v error'.format(next_v.shape)
            assert len(r.shape) == 2, 'R shape error:{}'.format(r.shape)
            values = r + self.gamma * next_v

            return values.reshape(-1)

        else:
            assert False, 'Q Evaluation Assertion Error. Obs shape:{}, Action shape: {}'.format(obs.shape,
                                                                                                actions.shape)

    def encode(self, obs):
        return self.rep_net(obs)

    def dynamics(self, hidden, action):
        return self.dyn_net(hidden, action)

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

    def get_weights(self):
        return dict_to_cpu(self.state_dict())

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def initial_inference(self, observation):
        h = self.encode(observation)
        pi = self.policy_no_split(h)
        value = self.value(h)

        # reward equal to 0 for consistency
        reward = (torch.zeros(1, self.full_reward_support_size)
                  .scatter(1, torch.tensor([[self.full_reward_support_size // 2]]).long(), 1.0)
                  .repeat(len(observation), 1)
                  .to(observation.device))

        if self.config.efficient_imitation:
            pibc = self.expert(h)
            return (
                value,
                reward,
                torch.cat((pi, pibc), dim=-1),
                h
            )
        else:
            return (
                value,
                reward,
                pi,
                h,
            )

    def atanh(self, x):
        return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

    # @profile
    def determine_open_close_one_hot(self, policy, num, modified_action=None, ready_masks=None, closable_masks=None,
                                     action_highs=None, action_lows=None,
                                     add_explore_noise=False, is_root=False, ori_states=None):
        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim
        if ready_masks is not None:
            if not self.config.parameters['only_power']:
                ready_masks = np.concatenate((ready_masks[:, :generator_num], ready_masks[:, -1:]), axis=1)
                closable_masks = np.concatenate((closable_masks[:, :generator_num], closable_masks[:, -1:]), axis=1)
            ready_masks = np.expand_dims(ready_masks, axis=1).repeat(num, axis=1)
            closable_masks = np.expand_dims(closable_masks, axis=1).repeat(num, axis=1)
        if ori_states is not None:
            delta_load_p = ori_states[:, 0]
            balance_up_redundency = ori_states[:, -1]
            balance_down_redundency = ori_states[:, -2]
            addition = 20
            redundency_adjust = -(1 - np.sign(balance_up_redundency)) / 2 * (
                        balance_up_redundency - addition) + \
                                (1 - np.sign(balance_down_redundency)) / 2 * (
                                            balance_down_redundency - addition)

            real_actions = np.zeros_like(modified_action)
            for i in range(modified_action.shape[1]):
                real_actions[:, i, :] = self.modify_policy(torch.from_numpy(modified_action[:, i, :]),
                                                           torch.from_numpy(ready_masks[:, i, :]).squeeze(1),
                                                           torch.from_numpy(closable_masks[:, i, :]).squeeze(1),
                                                           torch.from_numpy(action_highs),
                                                           torch.from_numpy(action_lows)
                                                           ).detach().cpu().numpy()
            modified_action = copy.deepcopy(real_actions)

            if self.config.parameters['only_power']:
                delta = np.expand_dims(delta_load_p, axis=1).repeat(num, axis=1) - \
                        modified_action.sum(2) + \
                        np.expand_dims(redundency_adjust, axis=1).repeat(num, axis=1)
            else:
                delta = np.expand_dims(delta_load_p, axis=1).repeat(num, axis=1) - \
                        modified_action[:, :, :generator_num].sum(2) + \
                        np.expand_dims(redundency_adjust, axis=1).repeat(num, axis=1)

            gen_q = ori_states[:, -289:-235]
            min_gen_q = settings.min_gen_q
            max_gen_q = settings.max_gen_q
            q_overload = (gen_q / max_gen_q > 1.0).astype(np.float32)
            q_underload = (gen_q / min_gen_q < 1.0).astype(np.float32)
            nextstep_renewable_gen_p_max = np.expand_dims(ori_states[:, -20:-2], axis=1).repeat(num, axis=1)
            gen_p = ori_states[:, 1:55]
            load_p = ori_states[:, 55:55+91]
            next_load_p = np.expand_dims(ori_states[:, 55+91:55+91+91], axis=1).repeat(num, axis=1)
            renewable_gen_p = np.expand_dims(gen_p[:, settings.renewable_ids], axis=1).repeat(num, axis=1) + modified_action[:, :, settings.renewable_ids]
            renewable_consump_rate = renewable_gen_p.sum(-1) / nextstep_renewable_gen_p_max.sum(-1)
            sum_renewable_up_redundency = nextstep_renewable_gen_p_max.sum(-1) - renewable_gen_p.sum(-1)

            expand_gen_p = np.expand_dims(gen_p, axis=1).repeat(num, axis=1)
            running_cost = calc_running_cost_rew(expand_gen_p, is_real=True)
            open_action_low = np.zeros_like(modified_action)
            open_action_low[:, :, settings.renewable_ids] = nextstep_renewable_gen_p_max
            action_low = ori_states[:, -91:-38]
            tmp = np.expand_dims(np.expand_dims(np.array((settings.max_gen_p[settings.balanced_id]+settings.min_gen_p[settings.balanced_id])/2-80), axis=0), axis=0).repeat(ori_states.shape[0], axis=0)
            action_low = np.concatenate((action_low[:, :settings.balanced_id], tmp, action_low[:, settings.balanced_id:]), axis=1)
            action_low = np.expand_dims(action_low, axis=1).repeat(num, axis=1)
            open_action_low[:, :, settings.thermal_ids] = np.expand_dims(gen_p[:, settings.thermal_ids], axis=1).repeat(num, axis=1) + action_low[:, :, settings.thermal_ids]
            open_action_low[:, :, settings.balanced_id] = np.expand_dims(gen_p[:, settings.balanced_id], axis=1).repeat(num, axis=1) + action_low[:, :, settings.balanced_id]

            close_mask = ((delta < -40).astype(np.float32)
                          + (renewable_consump_rate < 0.7 + 0.2 * random.random()).astype(np.float32)
                          + (next_load_p.sum(-1) - open_action_low.sum(-1) < 80)
                          + (sum_renewable_up_redundency > 100).astype(np.float32)
                          # + (q_underload.sum(-1) > 0).astype(np.float32)
                          ) * (closable_masks.sum(-1) > 0).astype(np.float32) #* (ready_masks.sum(-1) > 0).astype(np.float32)
            close_mask = (close_mask > 0).astype(np.float32)
            open_mask = ((delta > 40).astype(np.float32)
                         # + (running_cost < -0.5).astype(np.float32)
                         # + (next_load_p.sum(-1) - open_action_low.sum(-1) < 30) * (closable_masks.sum(-1) == 0)
                         # + (q_overload.sum(-1) > 0).astype(np.float32)
                         ) * (ready_masks.sum(-1) > 0).astype(np.float32)
            open_mask, close_mask = torch.from_numpy(open_mask).float().to('cuda'), torch.from_numpy(close_mask).float().to('cuda')
            open_mask = open_mask.unsqueeze(2)
            close_mask = close_mask.unsqueeze(2)
            open_mask = open_mask.repeat(1, 1, one_hot_dim)
            close_mask = close_mask.repeat(1, 1, one_hot_dim)

        # policy = policy.detach().cpu()
        if ready_masks is not None:
            ready_masks = torch.from_numpy(ready_masks).float().to('cuda')
            closable_masks = torch.from_numpy(closable_masks).float().to('cuda')
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
                                                     policy[:, 4 * generator_num:4*generator_num+one_hot_dim], policy[:, 4*generator_num+one_hot_dim:]

        # open_close_fee = -np.append(np.asarray(settings.startup_cost), 0)
        # open_close_factor = F.softmax(torch.from_numpy(open_close_fee).float().to('cuda')/150)*50
        min_gen_p = np.append(np.asarray(settings.min_gen_p), 0)
        min_gen_p = np.expand_dims(min_gen_p, axis=0).repeat(policy.shape[0], axis=0)
        min_gen_p = np.expand_dims(min_gen_p, axis=1).repeat(num, axis=1)
        if ori_states is not None:
            open_factor = F.softmax(torch.from_numpy(-np.abs(min_gen_p + np.expand_dims(delta, axis=2).repeat(min_gen_p.shape[2], axis=2))).float().to('cuda')/80, dim=2)*50
            close_factor = F.softmax(torch.from_numpy(-np.abs(min_gen_p - np.expand_dims(delta, axis=2).repeat(min_gen_p.shape[2], axis=2))).float().to('cuda')/80, dim=2)*50
            # print(f'root_shape={open_close_factor.shape}')
            open_close_factor = torch.cat((open_factor, close_factor), dim=0)
            open_close_factor[:, :, settings.renewable_ids] = 1
            open_close_factor[:, :, settings.balanced_id] = 1
        else:
            open_close_factor = torch.ones((min_gen_p.shape[0], min_gen_p.shape[1], min_gen_p.shape[2])).float().to('cuda')
            # print(f'non-root_shape={open_close_factor.shape}')
            open_close_factor = open_close_factor.repeat(2, 1, 1)
        open_logits = open_logits.unsqueeze(1).repeat(1, num, 1)
        close_logits = close_logits.unsqueeze(1).repeat(1, num, 1)

        if add_explore_noise:
            if self.config.parameters['only_power']:
                noises = torch.from_numpy(np.random.dirichlet([self.config.root_dirichlet_alpha] * one_hot_dim, (2*open_logits.shape[0], num))).float().to('cuda')
            else:
                noises = torch.from_numpy(np.random.dirichlet([self.config.root_dirichlet_alpha] * one_hot_dim, (2*open_logits.shape[0], num))).float().to('cuda')
            frac = self.config.root_exploration_fraction
            # logits = torch.cat((open_logits, close_logits), dim=0) * (1 - frac) + noises * frac
            # one_hots = F.gumbel_softmax(logits, hard=True, dim=2)
            # open_one_hots, close_one_hots = one_hots[:one_hots.shape[0]//2], one_hots[one_hots.shape[0]//2:]
            # restricted_logits *= torch.cat((ready_masks, closable_masks), dim=0)

            priors = F.softmax(torch.cat((open_logits, close_logits), dim=0), dim=2)
            priors = priors * (1 - frac) + noises * frac
            priors *= open_close_factor
            one_hots = F.gumbel_softmax(priors, hard=True, dim=2)
            # _, ids = torch.max(priors, dim=2)
            # one_hots = F.one_hot(ids, num_classes=one_hot_dim).float()
            open_one_hots, close_one_hots = one_hots[:one_hots.shape[0] // 2], one_hots[one_hots.shape[0] // 2:]

            if ori_states is not None:
                restricted_priors = priors * torch.cat((restricted_ready_masks, restricted_closable_masks), dim=0)
                restricted_one_hots = F.gumbel_softmax(restricted_priors, hard=True, dim=2, tau=1e-4)
                # _, restricted_ids = torch.max(restricted_priors, dim=2)
                # restricted_one_hots = F.one_hot(restricted_ids, num_classes=one_hot_dim).float()
                restricted_open_one_hots, restricted_close_one_hots = restricted_one_hots[:one_hots.shape[0]//2], restricted_one_hots[one_hots.shape[0]//2:]

                open_one_hots = open_mask * restricted_open_one_hots + (1 - open_mask) * open_one_hots
                close_one_hots = close_mask * restricted_close_one_hots + (1 - close_mask) * close_one_hots

        else:
            priors = F.softmax(torch.cat((open_logits, close_logits), dim=0), dim=2)
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
            sum_open_one_hots = open_one_hots[:, :, :-1].sum(2)
            open_one_hots[:, :, -1] = 1 - sum_open_one_hots
            sum_close_one_hots = close_one_hots[:, :, :-1].sum(2)
            close_one_hots[:, :, -1] = 1 - sum_close_one_hots

        return open_one_hots.cpu().numpy(), close_one_hots.cpu().numpy()

    def determine_open_close_one_hot_v2(self, policy, num, modified_action=None, ready_masks=None, closable_masks=None, add_explore_noise=False, is_root=False, ori_states=None):
        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim
        if ready_masks is not None:
            if not self.config.parameters['only_power']:
                ready_masks = np.concatenate((ready_masks[:, :generator_num], ready_masks[:, -1:]), axis=1)
                closable_masks = np.concatenate((closable_masks[:, :generator_num], closable_masks[:, -1:]), axis=1)
            ready_masks = np.expand_dims(ready_masks, axis=1).repeat(num, axis=1)
            closable_masks = np.expand_dims(closable_masks, axis=1).repeat(num, axis=1)
        if ori_states is not None:
            gen_q = ori_states[:, -181:-127]
            min_gen_q = settings.min_gen_q
            max_gen_q = settings.max_gen_q
            q_overload = (gen_q / max_gen_q > 1.0).astype(np.float32)
            q_underload = (gen_q / min_gen_q < 1.0).astype(np.float32)
            # nextstep_renewable_gen_p_max = np.expand_dims(ori_states[:, -19:-2], axis=1).repeat(num, axis=1)
            curstep_renewable_gen_p_max = np.expand_dims(ori_states[:, -36:-19], axis=1).repeat(num, axis=1)
            gen_p = ori_states[:, 1:55]
            # renewable_gen_p = np.expand_dims(gen_p[:, settings.renewable_ids], axis=1).repeat(num, axis=1) + modified_action[:, :, settings.renewable_ids]
            renewable_gen_p = np.expand_dims(gen_p[:, settings.renewable_ids], axis=1).repeat(num, axis=1)
            # renewable_consump_rate = renewable_gen_p.sum(-1) / nextstep_renewable_gen_p_max.sum(-1)
            renewable_consump_rate = renewable_gen_p.sum(-1) / curstep_renewable_gen_p_max.sum(-1)

            # sum_renewable_up_redundency = nextstep_renewable_gen_p_max.sum(-1) - renewable_gen_p.sum(-1)
            sum_renewable_up_redundency = curstep_renewable_gen_p_max.sum(-1) - renewable_gen_p.sum(-1)

            expand_gen_p = np.expand_dims(gen_p, axis=1).repeat(num, axis=1)
            running_cost = calc_running_cost_rew(expand_gen_p, is_real=True)

            close_mask = (
                                 # (delta < -40).astype(np.float32) +
                          (renewable_consump_rate < 0.6 + 0.3 * random.random()).astype(np.float32)
                          + (sum_renewable_up_redundency > 50).astype(np.float32)
                          # + (q_underload.sum(-1) > 0).astype(np.float32)
                          ) * (closable_masks.sum(-1) > 0).astype(np.float32) #* (ready_masks.sum(-1) > 0).astype(np.float32)
            close_mask = (close_mask > 0).astype(np.float32)
            open_mask = (
                                # (delta > 40).astype(np.float32) +
                         (running_cost < -0.5).astype(np.float32)
                         # + (q_overload.sum(-1) > 0).astype(np.float32)
                         ) * (ready_masks.sum(-1) > 0).astype(np.float32)
            open_mask = (open_mask > 0).astype(np.float32)
            open_mask, close_mask = torch.from_numpy(open_mask).float().to('cuda'), torch.from_numpy(close_mask).float().to('cuda')
            open_mask = open_mask.unsqueeze(2)
            close_mask = close_mask.unsqueeze(2)
            open_mask = open_mask.repeat(1, 1, one_hot_dim)
            close_mask = close_mask.repeat(1, 1, one_hot_dim)

        # policy = policy.detach().cpu()
        if ready_masks is not None:
            ready_masks = torch.from_numpy(ready_masks).float().to('cuda')
            closable_masks = torch.from_numpy(closable_masks).float().to('cuda')
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
                                                     policy[:, 4 * generator_num:4*generator_num+one_hot_dim], policy[:, 4*generator_num+one_hot_dim:]

        open_close_fee = -np.append(np.asarray(settings.startup_cost), 0)
        open_close_factor = F.softmax(torch.from_numpy(open_close_fee).float().to('cuda')/150)*50
        open_logits = open_logits.unsqueeze(1).repeat(1, num, 1)
        close_logits = close_logits.unsqueeze(1).repeat(1, num, 1)

        if add_explore_noise:
            if self.config.parameters['only_power']:
                noises = torch.from_numpy(np.random.dirichlet([self.config.root_dirichlet_alpha] * one_hot_dim, (2*open_logits.shape[0], num))).float().to('cuda')
            else:
                noises = torch.from_numpy(np.random.dirichlet([self.config.root_dirichlet_alpha] * one_hot_dim, (2*open_logits.shape[0], num))).float().to('cuda')
            frac = self.config.root_exploration_fraction
            # logits = torch.cat((open_logits, close_logits), dim=0) * (1 - frac) + noises * frac
            # one_hots = F.gumbel_softmax(logits, hard=True, dim=2)
            # open_one_hots, close_one_hots = one_hots[:one_hots.shape[0]//2], one_hots[one_hots.shape[0]//2:]
            # restricted_logits *= torch.cat((ready_masks, closable_masks), dim=0)

            priors = F.softmax(torch.cat((open_logits, close_logits), dim=0), dim=2)
            priors = priors * (1 - frac) + noises * frac
            priors *= open_close_factor
            one_hots = F.gumbel_softmax(priors, hard=True, dim=2)
            # _, ids = torch.max(priors, dim=2)
            # one_hots = F.one_hot(ids, num_classes=one_hot_dim).float()
            open_one_hots, close_one_hots = one_hots[:one_hots.shape[0] // 2], one_hots[one_hots.shape[0] // 2:]

            if ori_states is not None:
                restricted_priors = priors * torch.cat((restricted_ready_masks, restricted_closable_masks), dim=0)
                restricted_one_hots = F.gumbel_softmax(restricted_priors, hard=True, dim=2, tau=1e-4)
                # _, restricted_ids = torch.max(restricted_priors, dim=2)
                # restricted_one_hots = F.one_hot(restricted_ids, num_classes=one_hot_dim).float()
                restricted_open_one_hots, restricted_close_one_hots = restricted_one_hots[:one_hots.shape[0]//2], restricted_one_hots[one_hots.shape[0]//2:]

                open_one_hots = open_mask * restricted_open_one_hots + (1 - open_mask) * open_one_hots
                close_one_hots = close_mask * restricted_close_one_hots + (1 - close_mask) * close_one_hots

        else:
            priors = F.softmax(torch.cat((open_logits, close_logits), dim=0), dim=2)
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
            sum_open_one_hots = open_one_hots.sum(2)
            open_one_hots[:, :, -1] = 1 - sum_open_one_hots
            sum_close_one_hots = close_one_hots.sum(2)
            close_one_hots[:, :, -1] = 1 - sum_close_one_hots

        return open_one_hots.cpu().numpy(), close_one_hots.cpu().numpy()

    def modify_policy(self, action, ready_masks, closable_masks, action_high, action_low, is_test=False):
        ori_mu = action
        mu = (ori_mu + torch.ones_like(ori_mu)) / (2 * torch.ones_like(ori_mu)) * (
                action_high - action_low) + action_low
        modified_mu = mu * (torch.ones_like(ready_masks) - ready_masks)[:, :-1] * (torch.ones_like(closable_masks) - closable_masks)[:, :-1]

        modified_mu += torch.clamp(mu * closable_masks[:, :-1], 0, 10000)
        return modified_mu

    def modify_policy_v2(self, action_one_hot, ready_masks, closable_masks, action_high, action_low, is_test=False):
        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim
        if self.config.parameters['only_power']:
            action, open_one_hot, close_one_hot = action_one_hot[:, :generator_num], \
                                                  action_one_hot[:, generator_num:generator_num + one_hot_dim], \
                                                  action_one_hot[:, generator_num + one_hot_dim:]
        else:
            action, open_one_hot, close_one_hot = action_one_hot[:, :2 * generator_num], \
                                                  action_one_hot[:, 2 * generator_num:2 * generator_num + one_hot_dim], \
                                                  action_one_hot[:, 2 * generator_num + one_hot_dim:]
        ready_thermal_mask = ready_masks[:, :-1]
        closable_thermal_mask = closable_masks[:, :-1]
        mu = (action + torch.ones_like(action)) / (2 * torch.ones_like(action)) * (
                    action_high - action_low) + action_low
        modified_mu = mu * (torch.ones_like(ready_thermal_mask) - ready_thermal_mask)
        modified_mu *= (torch.ones_like(closable_thermal_mask) - closable_thermal_mask)

        modified_mu += torch.clamp(mu * closable_thermal_mask, 0, 10000)

        # if sum(open_one_hot) > 0:
        # open_id = open_one_hot.argmax(dim=-1)
        # if open_id < generator_num:
        #     modified_mu[open_id] = action_high[open_id]
        #     ready_thermal_mask[open_id] = 0

        open_one_hot = open_one_hot[:, :-1]
        close_one_hot = close_one_hot[:, :-1]
        modified_mu = modified_mu * (torch.ones_like(open_one_hot) - open_one_hot) + action_high * open_one_hot

        # if sum(close_one_hot) > 0:
        # close_id = close_one_hot.argmax(dim=-1)
        # if close_id < generator_num:
        #     modified_mu[close_id] = action_low[close_id]
        #     closable_thermal_mask[close_id] = 0
        modified_mu = modified_mu * (torch.ones_like(close_one_hot) - close_one_hot) - action_low * close_one_hot

        return modified_mu.detach().cpu().numpy()

    def check_balance(self, state, sampled_actions, action_high, action_low, ready_mask, closable_mask, norm_action=True):

        delta_load_p = state[:, 0]
        balance_up_redundency = state[:, -1]
        balance_down_redundency = state[:, -2]

        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim

        addition = 20
        tmp = 20
        redundency_adjust = -(1 - np.sign(balance_up_redundency)) / 2 * (balance_up_redundency
                                                                                    - tmp
                                                                                    # - tmp * random.random()
                                                                                    ) + \
                            (1 - np.sign(balance_down_redundency)) / 2 * (balance_down_redundency
                                                                                    - tmp
                                                                                     # - tmp * random.random()
                                                                                     )

        mask = ((action_high != 0).astype(np.float32) + (action_low != 0).astype(np.float32)) * \
               (1 - closable_mask[:, :-1]) * \
               (1 - ready_mask[:, :-1])     # represent adjustable generators
        thermal_mask = np.zeros_like(mask)
        thermal_mask[:, settings.thermal_ids] = 1
        mask *= thermal_mask  # only readjust thermal units
        mask = (mask > 0).astype(np.float32)
        if not self.config.parameters['only_power']:
            power_mask = np.zeros_like(action_high)
            power_mask[:, :generator_num] = 1   # represent active power control dimensions
            mask *= power_mask
        inv_mask = np.ones_like(mask) - mask  # represent non-adjustable generators, voltage control, closed or balance or open_gen_logit

        if norm_action:
            real_actions = np.zeros_like(sampled_actions)
            for i in range(sampled_actions.shape[1]):
                real_actions[:, i, :] = self.modify_policy(torch.from_numpy(sampled_actions[:, i, :]),
                                                              torch.from_numpy(ready_mask),
                                                              torch.from_numpy(closable_mask),
                                                              torch.from_numpy(action_high),
                                                              torch.from_numpy(action_low)
                                                              ).detach().cpu().numpy()
        else:
            real_actions = sampled_actions

        if self.config.parameters['only_power']:
            delta = np.expand_dims(delta_load_p, axis=1).repeat(real_actions.shape[1], axis=1) - \
                    real_actions.sum(2) + \
                    np.expand_dims(redundency_adjust, axis=1).repeat(real_actions.shape[1], axis=1)
        else:
            delta = np.expand_dims(delta_load_p, axis=1).repeat(real_actions.shape[1], axis=1) - \
                    real_actions[:, :, :generator_num].sum(2) + \
                    np.expand_dims(redundency_adjust, axis=1).repeat(real_actions.shape[1], axis=1)

        change_mask = (np.abs(delta) > 30).astype(np.float32)
        change_mask = np.expand_dims(change_mask, axis=2).repeat(real_actions.shape[2], axis=2)
        # change_mask = np.ones_like(change_mask)

        upgrade_redundency = (np.expand_dims(action_high, axis=1).repeat(real_actions.shape[1], axis=1) - real_actions) * \
                             np.expand_dims(mask, axis=1).repeat(real_actions.shape[1], axis=1)
        downgrade_redundency = (real_actions - np.expand_dims(action_low, axis=1).repeat(real_actions.shape[1], axis=1)) * \
                               np.expand_dims(mask, axis=1).repeat(real_actions.shape[1], axis=1)

        modification = (1 + np.sign(np.expand_dims(delta, axis=2).repeat(upgrade_redundency.shape[-1], axis=2))) / 2 * \
                       np.expand_dims(delta, axis=2).repeat(upgrade_redundency.shape[-1], axis=2) * upgrade_redundency / (
                                   np.expand_dims(upgrade_redundency.sum(2), axis=2).repeat(upgrade_redundency.shape[-1], axis=2) + 1e-3) + \
                       (1 - np.sign(np.expand_dims(delta, axis=2).repeat(downgrade_redundency.shape[-1], axis=2))) / 2 * \
                       np.expand_dims(delta, axis=2).repeat(downgrade_redundency.shape[-1], axis=2) * downgrade_redundency / (
                                   np.expand_dims(downgrade_redundency.sum(2), axis=2).repeat(downgrade_redundency.shape[-1], axis=2) + 1e-3)

        real_actions += modification * change_mask
        # real_actions += modification * change_mask * power_mask

        if not norm_action:
            return real_actions.squeeze()

        modified_sampled_actions = (real_actions - np.expand_dims(action_low, axis=1).repeat(real_actions.shape[1], axis=1)) / \
                                   (np.expand_dims(action_high, axis=1).repeat(real_actions.shape[1], axis=1) -
                                    np.expand_dims(action_low, axis=1).repeat(real_actions.shape[1], axis=1) + 1e-3) * 2 - 1

        modified_sampled_actions = modified_sampled_actions * np.expand_dims(mask, axis=1).repeat(real_actions.shape[1], axis=1) + \
                                   sampled_actions * np.expand_dims(inv_mask, axis=1).repeat(real_actions.shape[1], axis=1)

        modified_sampled_actions = modified_sampled_actions.clip(-0.999, 0.999)
        return modified_sampled_actions

    def check_balance_round2(self, state, sampled_actions, action_high, action_low, ready_mask, closable_mask, norm_action=True):
        delta_load_p = state[:, 0]
        balance_up_redundency = state[:, -1]
        balance_down_redundency = state[:, -2]

        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim
        open_one_hot = sampled_actions[:, :, generator_num:generator_num+one_hot_dim]
        close_one_hot = sampled_actions[:, :, generator_num+one_hot_dim:]

        addition = 20
        tmp = 20
        redundency_adjust = -(1 - np.sign(balance_up_redundency)) / 2 * (balance_up_redundency - tmp) + \
                            (1 - np.sign(balance_down_redundency)) / 2 * (balance_down_redundency - tmp)
        mask = ((action_high != 0).astype(np.float32) + (action_low != 0).astype(np.float32)) * \
               (1 - closable_mask[:, :-1]) * \
               (1 - ready_mask[:, :-1])     # represent adjustable generators
        thermal_mask = np.zeros_like(mask)
        thermal_mask[:, settings.thermal_ids] = 1
        mask *= thermal_mask    # only readjust thermal units
        mask = (mask > 0).astype(np.float32)
        if not self.config.parameters['only_power']:
            power_mask = np.zeros_like(action_high)
            power_mask[:, :generator_num] = 1   # represent active power control dimensions
            mask *= power_mask
        inv_mask = np.ones_like(mask) - mask  # represent non-adjustable generators, voltage control, closed or balance or open_gen_logit

        if norm_action:
            # real_actions = np.zeros_like(sampled_actions)
            real_actions = np.zeros((sampled_actions.shape[0], sampled_actions.shape[1], generator_num))
            for i in range(sampled_actions.shape[1]):
                real_actions[:, i, :] = self.modify_policy(torch.from_numpy(sampled_actions[:, i, :generator_num]),
                                                                                      torch.from_numpy(ready_mask),
                                                                                      torch.from_numpy(closable_mask),
                                                                                      torch.from_numpy(action_high),
                                                                                      torch.from_numpy(action_low)
                                                                                      )
        else:
            real_actions = sampled_actions

        min_gen_p = np.expand_dims(np.append(np.array(settings.min_gen_p), 0), axis=0).repeat(sampled_actions.shape[0], axis=0)
        min_gen_p = np.expand_dims(min_gen_p, axis=1).repeat(sampled_actions.shape[1], axis=1)
        if self.config.parameters['only_power']:
            delta = np.expand_dims(delta_load_p, axis=1).repeat(real_actions.shape[1], axis=1) - \
                    real_actions.sum(2) + \
                    np.expand_dims(redundency_adjust, axis=1).repeat(real_actions.shape[1], axis=1) + \
                    (-min_gen_p * open_one_hot + min_gen_p * close_one_hot).sum(2)
        else:
            delta = np.expand_dims(delta_load_p, axis=1).repeat(real_actions.shape[1], axis=1) - \
                    real_actions[:, :, :generator_num].sum(2) + \
                    np.expand_dims(redundency_adjust, axis=1).repeat(real_actions.shape[1], axis=1) + \
                    (-min_gen_p * open_one_hot + min_gen_p * close_one_hot).sum(2)

        change_mask = (np.abs(delta) > 30).astype(np.float32)
        change_mask = np.expand_dims(change_mask, axis=2).repeat(real_actions.shape[2], axis=2)
        # change_mask = np.ones_like(change_mask)

        upgrade_redundency = (np.expand_dims(action_high, axis=1).repeat(real_actions.shape[1], axis=1) - real_actions) * \
                             np.expand_dims(mask, axis=1).repeat(real_actions.shape[1], axis=1)
        downgrade_redundency = (real_actions - np.expand_dims(action_low, axis=1).repeat(real_actions.shape[1], axis=1)) * \
                               np.expand_dims(mask, axis=1).repeat(real_actions.shape[1], axis=1)

        modification = (1 + np.sign(np.expand_dims(delta, axis=2).repeat(upgrade_redundency.shape[-1], axis=2))) / 2 * \
                       np.expand_dims(delta, axis=2).repeat(upgrade_redundency.shape[-1], axis=2) * upgrade_redundency / (
                                   np.expand_dims(upgrade_redundency.sum(2), axis=2).repeat(upgrade_redundency.shape[-1], axis=2) + 1e-3) + \
                       (1 - np.sign(np.expand_dims(delta, axis=2).repeat(downgrade_redundency.shape[-1], axis=2))) / 2 * \
                       np.expand_dims(delta, axis=2).repeat(downgrade_redundency.shape[-1], axis=2) * downgrade_redundency / (
                                   np.expand_dims(downgrade_redundency.sum(2), axis=2).repeat(downgrade_redundency.shape[-1], axis=2) + 1e-3)

        modified_actions = real_actions + modification * change_mask
        # print(np.expand_dims(delta_load_p, axis=1).repeat(real_actions.shape[1], axis=1) - \
        #             modified_actions[:, :, :generator_num].sum(2) + \
        #             np.expand_dims(redundency_adjust, axis=1).repeat(real_actions.shape[1], axis=1))
        # real_actions += modification * change_mask * power_mask

        if not norm_action:
            return real_actions.squeeze()

        modified_sampled_actions = (modified_actions - np.expand_dims(action_low, axis=1).repeat(modification.shape[1], axis=1)) / \
                                   (np.expand_dims(action_high, axis=1).repeat(modified_actions.shape[1], axis=1) -
                                    np.expand_dims(action_low, axis=1).repeat(modified_actions.shape[1], axis=1) + 1e-3) * 2 - 1

        modified_sampled_actions = modified_sampled_actions * np.expand_dims(mask, axis=1).repeat(modified_actions.shape[1], axis=1) + \
                                   sampled_actions[:, :, :generator_num] * np.expand_dims(inv_mask, axis=1).repeat(real_actions.shape[1], axis=1)

        modified_sampled_actions = modified_sampled_actions.clip(-0.999, 0.999)
        return np.concatenate((modified_sampled_actions, open_one_hot, close_one_hot), axis=2)


    def check_balance_v2(self, state, sampled_actions, action_high, action_low, ready_mask, closable_mask, norm_action=True):
        delta_load_p = state[:, 0]
        balance_up_redundency = state[:, -1]
        balance_down_redundency = state[:, -2]

        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim

        addition = 20
        tmp = 80
        redundency_adjust = -(1 - np.sign(balance_up_redundency)) / 2 * (balance_up_redundency
                                                                                    - tmp
                                                                                    # - tmp * random.random()
                                                                                    ) + \
                            (1 - np.sign(balance_down_redundency)) / 2 * (balance_down_redundency
                                                                                    - tmp
                                                                                     # - tmp * random.random()
                                                                                     )

        mask = ((action_high != 0).astype(np.float32) + (action_low != 0).astype(np.float32)) * \
               (1 - closable_mask[:, :-1]) * \
               (1 - ready_mask[:, :-1])     # represent adjustable generators
        mask = (mask > 0).astype(np.float32)
        if not self.config.parameters['only_power']:
            power_mask = np.zeros_like(action_high)
            power_mask[:, :generator_num] = 1   # represent active power control dimensions
            mask *= power_mask
        inv_mask = np.ones_like(mask) - mask  # represent non-adjustable generators, voltage control, closed or balance or open_gen_logit

        if norm_action:
            # real_actions = np.zeros_like(sampled_actions)
            real_actions = np.zeros((sampled_actions.shape[0], sampled_actions.shape[1], generator_num))
            for i in range(sampled_actions.shape[1]):
                real_actions[:, i, :] = self.modify_policy_v2(torch.from_numpy(sampled_actions[:, i, :]),
                                                                                      torch.from_numpy(ready_mask),
                                                                                      torch.from_numpy(closable_mask),
                                                                                      torch.from_numpy(action_high),
                                                                                      torch.from_numpy(action_low)
                                                                                      )
        else:
            real_actions = sampled_actions

        # import ipdb
        # ipdb.set_trace()
        if self.config.parameters['only_power']:
            delta = np.expand_dims(delta_load_p, axis=1).repeat(real_actions.shape[1], axis=1) - \
                    real_actions.sum(2) + \
                    np.expand_dims(redundency_adjust, axis=1).repeat(real_actions.shape[1], axis=1)
        else:
            delta = np.expand_dims(delta_load_p, axis=1).repeat(real_actions.shape[1], axis=1) - \
                    real_actions[:, :, :generator_num].sum(2) + \
                    np.expand_dims(redundency_adjust, axis=1).repeat(real_actions.shape[1], axis=1)

        change_mask = (np.abs(delta) > 30).astype(np.float32)
        change_mask = np.expand_dims(change_mask, axis=2).repeat(real_actions.shape[2], axis=2)
        # change_mask = np.ones_like(change_mask)

        upgrade_redundency = (np.expand_dims(action_high, axis=1).repeat(real_actions.shape[1], axis=1) - real_actions) * \
                             np.expand_dims(mask, axis=1).repeat(real_actions.shape[1], axis=1)
        downgrade_redundency = (real_actions - np.expand_dims(action_low, axis=1).repeat(real_actions.shape[1], axis=1)) * \
                               np.expand_dims(mask, axis=1).repeat(real_actions.shape[1], axis=1)

        modification = (1 + np.sign(np.expand_dims(delta, axis=2).repeat(upgrade_redundency.shape[-1], axis=2))) / 2 * \
                       np.expand_dims(delta, axis=2).repeat(upgrade_redundency.shape[-1], axis=2) * upgrade_redundency / (
                                   np.expand_dims(upgrade_redundency.sum(2), axis=2).repeat(upgrade_redundency.shape[-1], axis=2) + 1e-3) + \
                       (1 - np.sign(np.expand_dims(delta, axis=2).repeat(downgrade_redundency.shape[-1], axis=2))) / 2 * \
                       np.expand_dims(delta, axis=2).repeat(downgrade_redundency.shape[-1], axis=2) * downgrade_redundency / (
                                   np.expand_dims(downgrade_redundency.sum(2), axis=2).repeat(downgrade_redundency.shape[-1], axis=2) + 1e-3)

        modified_actions = real_actions + modification * change_mask
        # print(np.expand_dims(delta_load_p, axis=1).repeat(real_actions.shape[1], axis=1) - \
        #             modified_actions[:, :, :generator_num].sum(2) + \
        #             np.expand_dims(redundency_adjust, axis=1).repeat(real_actions.shape[1], axis=1))
        # real_actions += modification * change_mask * power_mask

        if not norm_action:
            return real_actions.squeeze()

        modified_sampled_actions = (modified_actions - np.expand_dims(action_low, axis=1).repeat(modification.shape[1], axis=1)) / \
                                   (np.expand_dims(action_high, axis=1).repeat(modified_actions.shape[1], axis=1) -
                                    np.expand_dims(action_low, axis=1).repeat(modified_actions.shape[1], axis=1) + 1e-3) * 2 - 1

        modified_sampled_actions = modified_sampled_actions * np.expand_dims(mask, axis=1).repeat(modified_actions.shape[1], axis=1) + \
                                   sampled_actions[:, :, :generator_num] * np.expand_dims(inv_mask, axis=1).repeat(real_actions.shape[1], axis=1)

        modified_sampled_actions = modified_sampled_actions.clip(-0.999, 0.999)
        return modified_sampled_actions

    # @profile
    def recurrent_inference(self, h, action):
        """

        :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
        :param action: shape: [Batchsize, Action_Dim]
        :return:
        """
        r = self.reward(h, action)
        h = self.dynamics(h, action)
        pi = self.policy_no_split(h)
        value = self.value(h)
        if self.config.efficient_imitation:
            pibc = self.expert(h)
            return value, r, torch.cat((pi, pibc), dim=-1), h
        else:
            return value, r, pi, h