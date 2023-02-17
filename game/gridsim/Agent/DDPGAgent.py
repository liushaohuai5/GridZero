from Agent.BaseAgent import BaseAgent
from DDPG import ActorNet, CriticNet, ActorNet_GCN, CriticNet_GCN
from utils import get_action_space, voltage_action, adjust_renewable_generator, form_p_action, correct_thermal_gen, adjust_generator_p
from utilize.form_action import *
import torch.nn as nn
from torch.nn import L1Loss
import numpy as np

import torch
import copy

class DDPG_Agent(BaseAgent):
    def __init__(
            self,
            settings,
            replay_buffer,
            device,
            feature_dim,
            action_dim,
            state_dim,
            parameters,
            gen2busM
        ):

        BaseAgent.__init__(self, settings.num_gen)

        self.device = device
        self.replay_buffer = replay_buffer
        self.settings = settings
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.state_shape = (-1, state_dim)

        self.parameters = parameters
        self.gamma = parameters['discount']
        self.tau = parameters['tau']
        self.initial_eps = parameters['initial_eps']
        self.end_eps = parameters['end_eps']
        self.eps_decay = parameters['eps_decay']

        self.pool = torch.from_numpy(gen2busM).to('cuda')

        if parameters['encoder'] == 'mlp':
            self.actor = ActorNet(state_dim, action_dim, self.settings).to(self.device)
            self.critic = CriticNet(state_dim, action_dim, self.settings).to(self.device)
        elif parameters['encoder'] == 'gcn':
            self.actor = ActorNet_GCN(feature_dim, state_dim, action_dim, self.settings, self.pool).to(self.device)
            self.critic = CriticNet_GCN(feature_dim, state_dim, action_dim, self.settings).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=parameters['optimizer_parameters']['lr'], weight_decay=parameters['optimizer_parameters']['weight_decay'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=parameters['optimizer_parameters']['lr'] / 10, weight_decay=parameters['optimizer_parameters']['weight_decay'])
        self.cnt = 0

        import ipdb
        ipdb.set_trace()
        self.original_bus_branch = {}
        cc = np.load("ori_bus_branch.npy", allow_pickle=True).tolist()
        for key, value in cc.items():
            self.original_bus_branch[key] = value

        self.last_p_or = np.zeros(settings.num_line)
        self.last_p_ex = np.zeros(settings.num_line)
        self.last_rho = np.zeros(settings.num_line)

    # def act(self, state, obs, done=False):
    def act(self, item, obs, done=False):
        self.actor.eval()
        X, weighted_A, state = item
        # state = torch.from_numpy(np.append(state, np.float32(thermal_object))).unsqueeze(0).to(self.device)
        X = torch.from_numpy(X).unsqueeze(0).to(self.device)
        weighted_A = torch.from_numpy(weighted_A).unsqueeze(0).to(self.device)
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        action_high, action_low = get_action_space(obs, self.parameters, self.settings)
        action_high, action_low = torch.from_numpy(action_high).unsqueeze(0).to(self.device), torch.from_numpy(
            action_low).unsqueeze(0).to(self.device)
        # adjust_gen_p = self.actor(state, action_high, action_low).squeeze().detach().cpu().numpy()
        adjust_gen_p = self.actor(X, weighted_A, state, action_high, action_low).squeeze().detach().cpu().numpy()
        adjust_gen_v = voltage_action(obs, self.settings)

        # adjust_gen_p = adjust_generator_p(obs, self.settings, self.original_bus_branch, self.last_p_or, self.last_p_ex, self.last_rho)
        # adjust_gen_v = voltage_action(obs, self.settings)
        # adjust_gen_v = np.zeros_like(adjust_gen_p)
        bias = 0
        # self.last_p_or = obs.p_or
        # self.last_p_ex = obs.p_ex
        # self.last_rho = obs.rho
        return form_action(adjust_gen_p, adjust_gen_v), bias

    def copy_target_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def train(self, time_step):
        self.actor.train()
        self.critic.train()

        for epoch in range(self.parameters['K_epochs']):
            # Sample replay buffer
            X, weighted_A, state, action, action_high, action_low, next_X, next_weighted_A, next_state, next_action_high, next_action_low, reward, done, renewable_objects, rho, ind, weights_lst = self.replay_buffer.sample(time_step)
            weights = torch.from_numpy(weights_lst).to(self.device).float()

            # Compute the target Q value using the information of next state
            # next_state = next_state.to(self.device)
            # import ipdb
            # ipdb.set_trace()
            next_action = self.actor_target(next_X, next_weighted_A, next_state, next_action_high, next_action_low)
            Q_tmp = self.critic_target(next_X, next_weighted_A, next_state, next_action)
            Q_target = reward + self.gamma * (1 - done) * Q_tmp
            Q_current = self.critic(X, weighted_A, state, action)

            # td_errors = L1Loss(reduction='none')(Q, Q_target)
            td_errors = Q_target - Q_current
            priorities = L1Loss(reduction='none')(Q_current, Q_target).data.cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(ind, priorities)

            # Compute the current Q value and the loss
            critic_loss = torch.mean(weights * (td_errors ** 2))   # with importance sampling
            # critic_loss = nn.MSELoss()(Q_current, Q_target)     # without importance sampling

            # Optimize the critic network
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)
            self.critic_optimizer.step()

            # Make action and evaluate its action values
            action_out = self.actor(X, weighted_A, state, action_high, action_low)
            Q = self.critic(X, weighted_A, state, action_out)
            G = torch.sum(action_out[:, :17], axis=1) + torch.sum(action_out[:, 18:], axis=1)
            # delta_load = state[:, 0]
            delta_load = state[:, 0]
            # balanced_p = state[:, 18]
            balanced_p = state[:, 1]
            L = delta_load - (self.settings.max_gen_p[17] - balanced_p)
            U = delta_load + (balanced_p - self.settings.min_gen_p[17])
            # grid_loss = state[:, -1]
            grid_loss = state[:, 2]
            rho_early_warning = 0.9

            actor_loss = -torch.mean(Q) \
                         + torch.mean(torch.max(torch.cat(((L-G).unsqueeze(0), torch.zeros_like(L-G).unsqueeze(0)), 0), 0).values) \
                         + torch.mean(torch.max(torch.cat(((G-U).unsqueeze(0), torch.zeros_like(G-U).unsqueeze(0)), 0), 0).values) \
                         + torch.mean(grid_loss) \
                         + torch.sum(torch.max(torch.cat(((rho - rho_early_warning).unsqueeze(0), torch.zeros_like(rho).unsqueeze(0)), 0), 0).values) \
                         + torch.mean(action_high[self.settings.renewable_ids] - action_out[self.settings.renewable_ids])   # more renewable generation
                         # + max(torch.mean(L-X), 0) + max(torch.mean(X-U), 0)
                         # + torch.mean(torch.max(torch.cat(((state[237:422] - rho_early_warning).unsqueeze(0), torch.zeros_like(state[237:422]).unsqueeze(0)), 0), 0).values) \

            # Optimize the actor network
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
            self.actor_optimizer.step()

            if self.cnt % 2 == 0:
            #     print(f'actor gradient max={max([np.abs(p).max() for p in self.actor.get_gradients()])}')
            #     print(f'critic gradient max={max([np.abs(p).max() for p in self.critic.get_gradients()])}')
                print(f'actor loss={actor_loss:.3f}, critic loss={critic_loss:.3f}')
            self.cnt += 1

        return {
            'training/Q': Q_current.mean().detach().cpu().numpy(),
            'training/critic_loss': critic_loss.mean().detach().cpu().numpy(),
            'training/actor_loss': actor_loss.mean().detach().cpu().numpy()
        }

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_DDPG_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "DDPG_actor_optimizer")
        torch.save(self.critic.state_dict(), filename + "_DDPG_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_DDPG_critic_optimizer")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_DDPG_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_DDPG_actor_optimizer"))
        self.critic.load_state_dict(torch.load(filename + "_DDPG_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_DDPG_critic_optimizer"))