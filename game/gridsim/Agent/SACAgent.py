import numpy as np
import torch
import torch.nn.functional as F

from Agent.BaseAgent import BaseAgent
from SAC_model import DiagGaussianActor, DoubleQCritic
import copy

from utils import get_action_space, form_p_action, adjust_renewable_generator, voltage_action
from utilize.form_action import *



class SACAgent(BaseAgent):
    """SAC algorithm."""
    def __init__(self, parameters, settings, replay_buffer, device):
        BaseAgent.__init__(self, settings.num_gen)


        self.parameters = parameters
        self.settings = settings
        self.replay_buffer = replay_buffer
        self.state_dim = parameters['state_dim']
        self.action_dim = parameters['action_dim']
        self.initial_eps = parameters['initial_eps']
        self.end_eps = parameters['end_eps']
        self.eps_decay = parameters['eps_decay']
        self.device = device
        self.discount = parameters['discount']
        self.critic_tau = parameters['critic']['tau']
        self.actor_update_frequency = parameters['actor']['update_frequency']
        self.critic_target_update_frequency = parameters['critic']['target_update_frequency']
        self.batch_size = parameters['batch_size']
        self.learnable_temperature = parameters['learnable_temperature']

        self.critic = DoubleQCritic(parameters['state_dim'], parameters['action_dim'],
                                    parameters['critic']['hidden_dim'], parameters['critic']['hidden_depth']).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor = DiagGaussianActor(parameters['state_dim'], parameters['action_dim'],
                                       parameters['actor']['hidden_dim'], parameters['actor']['hidden_depth'],
                                       parameters['actor']['log_std_bound']).to(self.device)

        self.log_alpha = torch.tensor(np.log(parameters['init_temperature'])).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -parameters['action_dim']

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=parameters['actor']['lr'],
                                                # betas=parameters['actor']['betas'],
                                                weight_decay=parameters['actor']['weight_decay'])

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=parameters['critic']['lr'],
                                                 # betas=parameters['critic']['betas'],
                                                 weight_decay=parameters['critic']['weight_decay'])

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=parameters['alpha']['lr'],
                                                    # betas=parameters['alpha']['betas']
                                                    )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, state, obs, sample=True):
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)
        dist = self.actor(state)
        action = dist.sample() if sample else dist.mean
        action = torch.tanh(action)
        # print(f'action max={action.max()}, action min={action.min()}')
        action = action.squeeze().detach().cpu().numpy()
        action_high, action_low = get_action_space(obs, self.parameters, self.settings)
        # action = (action - action.min()) / (action.max() - action.min()) * (action_high - action_low) + action_low
        action = (action + np.ones_like(action)) / (2 * np.ones_like(action)) \
            * (action_high - action_low) + action_low  # for Tanh or Sigmoid
        action = (action - (action_high / 2 + action_low / 2)) * 0.9 + (
                action_high / 2 + action_low / 2)  # compressed action space
        adjust_gen = action

        if self.parameters['only_power']:
            if self.parameters['only_thermal']:
                adjust_gen_renewable = adjust_renewable_generator(obs, self.settings)
                adjust_gen_p = form_p_action(adjust_gen, adjust_gen_renewable, self.settings)
            else:
                adjust_gen_p = adjust_gen
            adjust_gen_v = voltage_action(obs, self.settings)
        else:
            adjust_gen_p, adjust_gen_v = adjust_gen[:len(adjust_gen) // 2], adjust_gen[len(adjust_gen) // 2:]
        return form_action(adjust_gen_p, adjust_gen_v)

    def update_critic(self, obs, action, reward, next_obs, done, indices, weights):
        dist = self.actor(next_obs)
        next_action = dist.rsample()  # rsample is differentiable
        # next_action = torch.tanh(next_action)
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1,
                             target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + ((1-done) * self.discount * target_V)
        # target_Q = target_Q.detach()  # why detach?

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)

        # update priorities
        td_errors = target_Q - torch.min(current_Q1, current_Q2)
        priorities = torch.nn.L1Loss(reduction='none')(torch.min(current_Q1, current_Q2), target_Q).data.cpu().numpy() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities)

        critic_loss = torch.mean(weights * (td_errors ** 2))
        # critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
        #     current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=2)
        self.critic_optimizer.step()

        return {'training/Q': torch.min(current_Q1, current_Q2).mean().detach().cpu().numpy(),
                'training/critic_loss': critic_loss.mean().detach().cpu().numpy()}

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        # action = torch.tanh(action)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()
        # actor_loss = (self.alpha * log_prob - actor_Q).mean()

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=2)
        self.actor_optimizer.step()

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            # alpha_loss = (self.alpha *
            #               (-log_prob - self.target_entropy)).mean()
            alpha_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.log_alpha.parameters(), max_norm=1)
            self.log_alpha_optimizer.step()

        return {'training/actor_loss': actor_loss.mean().detach().cpu().numpy(),
                'training/alpha_loss': alpha_loss.mean().detach().cpu().numpy(),
                'training/alpha': self.alpha.mean().detach().cpu().numpy(),
                'training/log_prob': log_prob.detach().cpu().numpy()}

    def update(self, step):  # TODO: add PER
        for _ in range(self.parameters['K_epochs']):
            obs, action, action_high, action_low, next_obs, next_action_high, next_action_low, reward, done, ind, weights_lst = self.replay_buffer.sample(step)

            weights = torch.from_numpy(weights_lst).float().to(self.device)

            critic_info = self.update_critic(obs, action, reward, next_obs, done, ind, weights)

            if step % self.actor_update_frequency == 0:
                actor_info = self.update_actor_and_alpha(obs)
            else:
                actor_info = {}

            if step % self.critic_target_update_frequency == 0:
                print('update critic target')
                for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.critic_tau) + param.data * self.critic_tau)

            print(f'actor gradient max={max([np.abs(p).max() for p in self.actor.get_gradients()])}')
            print(f'critic gradient max={max([np.abs(p).max() for p in self.critic.get_gradients()])}')

        info = critic_info.copy()
        info.update(actor_info)
        return info

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_SAC_actor.pth")
        torch.save(self.actor_optimizer.state_dict(), filename + "SAC_actor_optimizer.pth")
        torch.save(self.critic.state_dict(), filename + "_SAC_critic.pth")
        torch.save(self.critic_optimizer.state_dict(), filename + "_SAC_critic_optimizer.pth")
