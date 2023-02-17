import torch
import torch.nn as nn
from utils import get_action_space, voltage_action, get_state_from_obs
from utilize.form_action import *
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from collections import deque



################################## set device ##################################
#
# print("============================================================================================")
#
#
# # set device to cpu or cuda
# device = torch.device('cpu')
#
# if(torch.cuda.is_available()):
#     device = torch.device('cuda:0')
#     torch.cuda.empty_cache()
#     print("Device set to : " + str(torch.cuda.get_device_name(device)))
# else:
#     print("Device set to : cpu")
#
# print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.max_size = 1 * 1000 * 1000
        self.actions = deque(maxlen=self.max_size)
        self.actions_high = deque(maxlen=self.max_size)
        self.actions_low = deque(maxlen=self.max_size)
        self.ready_thermal_masks = deque(maxlen=self.max_size)
        self.closable_thermal_masks = deque(maxlen=self.max_size)
        self.action_forms = deque(maxlen=self.max_size)
        self.expert_actions = deque(maxlen=self.max_size)
        self.states = deque(maxlen=self.max_size)
        self.logprobs = deque(maxlen=self.max_size)
        self.rewards = deque(maxlen=self.max_size)
        self.is_terminals = deque(maxlen=self.max_size)

    # def get_mean_std(self):


    # def clear(self):
    #     del self.actions[:]
    #     del self.actions_high[:]
    #     del self.actions_low[:]
    #     del self.ready_thermal_masks[:]
    #     del self.closable_thermal_masks[:]
    #     del self.action_forms[:]
    #     del self.expert_actions[:]
    #     del self.states[:]
        # del self.logprobs[:]
        # del self.rewards[:]
        # del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, parameters, device, settings):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space
        self.parameters = parameters
        self.device = device
        self.settings = settings

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(self.device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 1024),
                nn.Tanh(),
                nn.Linear(1024, 512),
                nn.Tanh(),
                nn.Linear(512, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 1024),
                nn.Tanh(),
                nn.Linear(1024, 512),
                nn.Tanh(),
                nn.Linear(512, action_dim),
                nn.Softmax(dim=-1)
            )


        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        self.softmax = nn.Softmax()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError


    def legalize_action(self, action, action_high, action_low, ready_thermal_mask, closable_thermal_mask):
        adjust_action = (action + torch.ones_like(action).to(self.device)) / (
                    2 * torch.ones_like(action).to(self.device)) \
                        * (action_high - action_low) + action_low

        modified_m = adjust_action * (torch.ones_like(ready_thermal_mask) - ready_thermal_mask)
        modified_m *= (torch.ones_like(closable_thermal_mask) - closable_thermal_mask)
        sum_ready_thermal_mask = ready_thermal_mask.sum(axis=1)
        open_m = torch.zeros_like(adjust_action)
        for i in range(ready_thermal_mask.shape[0]):
            if sum_ready_thermal_mask[i] > 0:
                open_id = ready_thermal_mask[i].nonzero()[
                    self.softmax(action[i].gather(0, ready_thermal_mask[i].nonzero().squeeze()).unsqueeze(0)).argmax()]
                if open_id == 108:
                    open_m[i, open_id] = 1
                else:
                    open_m[i, open_id] = adjust_action[i, open_id]
                    open_m[i, 108] = 1
        modified_m += open_m
        closable_m = torch.clamp(adjust_action * closable_thermal_mask, 0, 10000)  # prohibit thermal generator closing
        modified_m += closable_m

        adjust_action = modified_m
        return adjust_action

    def act(self, state, obs):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        # action = torch.clamp(action, min=-1, max=1)
        _, ready_thermal_mask, closable_thermal_mask = get_state_from_obs(obs, self.settings, self.parameters)
        action_high, action_low = get_action_space(obs, self.parameters, self.settings)
        ready_thermal_mask = torch.from_numpy(ready_thermal_mask).unsqueeze(0).to(self.device)
        closable_thermal_mask = torch.from_numpy(closable_thermal_mask).unsqueeze(0).to(self.device)
        action_high, action_low = torch.from_numpy(action_high).unsqueeze(0).to(self.device), torch.from_numpy(action_low).unsqueeze(0).to(self.device)
        # adjust_action = self.legalize_action(action, action_high, action_low, ready_thermal_mask, closable_thermal_mask)
        adjust_action = self.legalize_action(action_mean, action_high, action_low, ready_thermal_mask, closable_thermal_mask)

        action_logprob = dist.log_prob(action)
        adjust_action = adjust_action.squeeze().detach().cpu().numpy()

        if self.parameters['only_power']:
            adjust_gen_p = adjust_action
            adjust_gen_v = voltage_action(obs, self.settings)
        else:
            adjust_gen_p, adjust_gen_v = adjust_action[:54], adjust_action[54:108]

        action_form = form_action(adjust_gen_p, adjust_gen_v)
        recover_thermal_flag = adjust_action[-1]
        return action.detach(), action_logprob.detach(), action_form, recover_thermal_flag

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, parameters, device, lr_actor, lr_critic, gamma, K_epochs, eps_clip, settings, has_continuous_action_space, action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.parameters = parameters
        self.device = device
        self.settings = settings
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.train_freq = parameters['train_freq']

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, self.parameters, self.device, self.settings).to(self.device)
        self.imitation_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=lr_actor)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, self.parameters, self.device, self.settings).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    # def select_action(self, state, obs):
    def act(self, state, obs):
        bias = 0
        if self.has_continuous_action_space:
            # with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, action_logprob, action_form, recover_thermal_flag = self.policy_old.act(state, obs)

            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.action_forms.append(np.concatenate([action_form['adjust_gen_p'], action_form['adjust_gen_v'], np.reshape(np.asarray(recover_thermal_flag), (-1,))]).flatten())
            # self.buffer.logprobs.append(action_logprob)
            # self.buffer.obs.append(obs)

            return action_form, bias

        else:
            # with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, action_form = self.policy_old.act(state, obs)

            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)

            return action_form, bias


    def imitation(self, state_mean, state_std):
        expert_actions = torch.from_numpy(np.stack(self.buffer.expert_actions)[-self.train_freq:]).float().to(self.device)
        # TODO: do normalization
        states = np.stack(self.buffer.states)[-self.train_freq:]
        states = (states - state_mean) / (state_std + 1e-6)
        states = torch.from_numpy(states).float().to(self.device)
        actions_high = torch.from_numpy(np.stack(self.buffer.actions_high)[-self.train_freq:]).float().to(self.device)
        actions_low = torch.from_numpy(np.stack(self.buffer.actions_low)[-self.train_freq:]).float().to(self.device)
        ready_thermal_mask = torch.from_numpy(np.stack(self.buffer.ready_thermal_masks)[-self.train_freq:]).float().to(self.device)
        closable_thermal_mask = torch.from_numpy(np.stack(self.buffer.closable_thermal_masks)[-self.train_freq:]).float().to(self.device)
        for _ in range(self.K_epochs):
            actions_net = self.policy.actor(states)
            actions_net = self.policy.legalize_action(actions_net, actions_high, actions_low, ready_thermal_mask, closable_thermal_mask)
            # supervised_loss = self.MseLoss(expert_actions, actions_net)
            supervised_loss = torch.mean((expert_actions - actions_net)**2)

            print(f'imitation loss={supervised_loss}')

            self.imitation_optimizer.zero_grad()
            supervised_loss.mean().backward()
            self.imitation_optimizer.step()

        # self.buffer.clear()
        self.policy_old.actor.load_state_dict(self.policy.actor.state_dict())


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))