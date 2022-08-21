import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_utils import *


class MLPModel(nn.Module):
    def __init__(self, config):
        self.obs_shape = config.network_obs_shape
        self.action_shape = config.network_action_shape
        self.hidden_shape = config.network_hidden_shape
        self.proj_shape = config.network_proj_shape
        '''
            Models
        '''
        self.rep_net = mlp(self.obs_shape, [64,], self.hidden_shape, use_bn=False)
        self.dyn_net = mlp(self.hidden_shape + self.action_shape, [64,], self.hidden_shape, use_bn=False)
        self.rew_net = mlp(self.hidden_shape + self.action_shape, [64, 64], 1, use_bn=False)
        self.val_net = mlp(self.hidden_shape, [64, 64], 1, use_bn=False)
        self.pi_net = mlp(self.hidden_shape, [64, 64], self.action_shape * 2, use_bn=False)

        self.proj_net = mlp(self.hidden_shape, [], self.proj_shape, use_bn=False)
        self.proj_pred_net = mlp(self.proj_shape, [], self.proj_shape, use_bn=False)
        '''
            Optimizers
        '''
        self.model_optim = optim.Adam(list(self.rep_net.parameters()) +
                                      list(self.dyn.parameters()) +
                                      list(self.rew_net.parameters()) +
                                      list(self.proj_net.parameters()) +
                                      list(self.proj_pred_net.parameters()), lr=config.model_lr)
        self.value_optim = optim.Adam(list(self.val_net.parameters()), lr=config.value_lr)
        self.pi_optim = optim.Adam(list(self.pi_net.parameters()), lr=config.pi_lr)

    def discrim(self, hidden, action):
        return self.dis_net(hidden, action)

    def value(self, hidden):
        return self.val_net(hidden)

    def value_obs(self, obs):
        hidden = self.rep_net(obs)
        return self.value(hidden)

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

    def sample_action_obs(self, obs):
        mu, log_std = self.policy_obs(obs)
        action, _ = reparameterize(mu, log_std)
        return action

    def sample_n_action_obs(self, obs, n):
        mu, log_std = self.policy_obs(obs)
        action = reparameterize_n(mu, log_std, n)
        return action

    def sample_action(self, hidden):
        mu, log_std = self.policy(hidden)
        action, _ = reparameterize(mu, log_std)
        return action

    def sample_n_action(self, hidden, n):
        mu, log_std = self.policy(hidden)
        action, _ = reparameterize_n(mu, log_std, n)
        return action

    def encode(self, obs):
        return self.rep_net(obs)

    def dynamics(self, hidden, action):
        return self.dyn_net(torch.cat((hidden, action), dim=1))

    def project(self, h, with_grad=True):
        h_proj = self.proj_net(h)

        if with_grad:
            return self.proj_pred_net(h_proj)

        else:
            return h_proj.detach()

    def initial_inference(self, observation):
        h = self.encode(observation)
        pi = self.policy(h)
        value = self.value(h)
        reward = torch.zeros(observation.size(0), 1).to(observation.device)

        # policy_logits, value = self.prediction(encoded_state)
        #
        # # reward equal to 0 for consistency
        # reward = (torch.zeros(1, self.full_support_size)
        #           .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
        #           .repeat(len(observation), 1)
        #           .to(observation.device))
        return (
            value,
            reward,
            pi,
            h,
        )

    def recurrent_inference(self, h, action):
        """

        :param encoded_state: [Batchsize, Encoded_channel_dim, Encoded_w, Encoded_h]
        :param action: shape: [Batchsize, Action_Dim]
        :return:
        """
        r = self.reward(h, action)
        h = self.dynamics(h, action)
        pi = self.policy(h)
        value = self.value(h)
        return value, r, pi, h