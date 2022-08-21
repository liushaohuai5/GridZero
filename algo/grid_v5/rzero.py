import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from algo.grid_v5.model import MLPModel
import torch_utils
from torch_utils import *
import os
import ray
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from pickle_utils import *
from utilize.settings import settings
from torch.cuda.amp import autocast, GradScaler

torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, checkpoint, config):
        self.model = MLPModel(config).to(torch.device('cuda'))
        self.model.set_weights(copy.deepcopy(checkpoint["weights"]))
        self.model.train()

        self.target_model = MLPModel(config).to(torch.device('cuda'))
        self.target_model.set_weights(self.model.get_weights())
        self.target_model.eval()

        self.target_weight = copy.deepcopy(self.model.get_weights())

        # self.target_model = MLPModel(config).to(torch.device('cuda'))

        self.rep_lr_init = config.lr_init
        # self.rep_lr_init = 0.0
        self.pi_lr_init = config.lr_init
        # self.pi_lr_init = 0.0
        self.dyn_rew_lr_init = config.lr_init

        self.scaler = GradScaler()

        if config.optimizer == 'SGD':
            # if self.config.imitation:
            self.optimizer = optim.SGD(
                                       # self.model.parameters(),
                                       # lr=config.lr_init,
                                       [
                                           {'params': self.model.rep_net.parameters(), 'lr': self.rep_lr_init},
                                           {'params': self.model.pi_net.parameters(), 'lr': self.pi_lr_init},
                                           {'params': self.model.rew_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                           {'params': self.model.dyn_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                           {'params': self.model.val_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                           {'params': self.model.proj_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                           {'params': self.model.proj_pred_net.parameters(), 'lr': self.dyn_rew_lr_init}
                                       ],
                                       weight_decay=config.weight_decay,
                                       momentum=config.momentum
            )
        else:
            self.optimizer = optim.Adam(
                                        # self.model.parameters(),
                                        # lr=config.lr_init,
                                        [
                                            {'params': self.model.rep_net.parameters(), 'lr': self.rep_lr_init},
                                            {'params': self.model.pi_net.parameters(), 'lr': self.pi_lr_init},
                                            {'params': self.model.rew_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                            {'params': self.model.dyn_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                            {'params': self.model.val_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                            {'params': self.model.proj_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                            {'params': self.model.proj_pred_net.parameters(), 'lr': self.dyn_rew_lr_init}
                                        ],
                                        weight_decay=config.weight_decay)

        if config.lr_decay_type == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                   T_max=config.training_steps - config.warmup_steps)
        else:
            self.scheduler = None

        self.config = config
        self.training_step = 0

        self.train_debug_batch = []

        self.imitation_log_std = self.config.imitation_log_std
        self.tanh = nn.Tanh()

        self.imitation_to_finetune = False


    def loss_consistency_atomic_fn(self, h, h_t):
        # print('FORWARD', h.shape, h_t.shape)
        h = self.model.project(h, with_grad=True)
        h_t = self.model.project(h_t, with_grad=False)
        h = F.normalize(h, p=2., dim=-1, eps=1e-5)
        h_t = F.normalize(h_t, p=2., dim=-1, eps=1e-5)
        loss = -(h * h_t).sum(dim=1, keepdim=True)
        return loss

    def loss_consistency_fn(self, obs, action):
        '''
        :param obs: [b, unroll_step+1, o_dim]
        :param action: [b, unroll_step, a_dim]
        :return:
        '''
        unroll_steps = action.size(1)
        h = self.model.encode(obs[:, 0, :])
        a = action[:, 0, :]

        loss = 0
        for t in range(unroll_steps):
            h_next = self.model.dynamics(h, a)
            h_next_t = self.model.encode(obs[:, t+1, :])

            h_next_proj = self.model.project(h_next, with_grad=True)
            h_next_t_proj = self.model.project(h_next_t, with_grad=False)

            h_next_proj = F.normalize(h_next_proj, p=2., dim=-1, eps=1e-5)
            h_next_t_proj = F.normalize(h_next_t_proj, p=2., dim=-1, eps=1e-5)
            loss += -(h_next_proj, h_next_t_proj).sum(dim=1, keepdim=True)

            h = h_next

        return loss

    def loss_reward_fn(self, r, target_r):
        # r = self.model.reward(hidden, action)
        target_r = scalar_to_support(target_r, self.config.reward_support_size,
                                     self.config.reward_support_step).squeeze()
        loss = (-target_r * torch.nn.LogSoftmax(dim=1)(r)).sum(1)
        return loss

    def loss_value_fn(self, v, target_v):
        '''
        :param hidden:      [b, h_dim]
        :param target_v:    [b, 1]
        :return:
        '''

        '''
            We use mse loss here. If this does not work, we can use 
        '''
        # v = self.model.value(hidden)
        target_v = scalar_to_support(target_v, self.config.support_size, self.config.value_support_step).squeeze()
        # print('TV', target_v.shape, v.shape)
        loss = (-target_v * torch.nn.LogSoftmax(dim=1)(v)).sum(1)
        # print('LOSS V', loss.shape)
        return loss


    def loss_pi_kl_fn(self, policy, target_action, target_policy):
        generator_num = self.config.generator_num
        one_hot_dim = self.config.one_hot_dim
        # policy: [batchsize, action_dim * 2]
        # target_action = [batch_size, n, action]
        # target_policies = [batch_size, n]
        if self.config.parameters['only_power']:
            action_dim = generator_num
        else:
            action_dim = 2*generator_num
        n_branches = target_policy.size(1)

        # def atanh(x):
        #     return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

        # if self.config.parameters['only_power']:
        target_action[:, :, :action_dim] = target_action[:, :, :action_dim].clip(-0.999, 0.999)
        distr = SquashedNormal(policy[:, :action_dim], policy[:, action_dim:2 * action_dim].exp() + 5e-4)
        # else:
        #     target_action[:, :, :2*generator_num] = target_action[:, :, :2*generator_num].clip(-0.999, 0.999)
        #     distr = SquashedNormal(policy[:, :2 * generator_num], policy[:, 2 * generator_num:4 * generator_num].exp() + 5e-4)

        # target_action_inverse = torch.atanh(target_action)

        # distr = torch.distributions.Normal(policy[:, :action_dim], policy[:, action_dim:].exp()+1e-5)


        log_probs = []

        # max_pol_pos = torch.argmax(target_policy, dim=1)
        # one_hot_mask = torch.nn.functional.one_hot(max_pol_pos, num_classes=n_branches).float()
        # print('A', 'P', target_action.shape, target_policy.shape)
        for i in range(n_branches):
            '''
                Modification Dec 31.
            '''
            # if self.config.parameters['only_power']:
            log_prob = distr.log_prob(target_action[:, i, :action_dim]).sum(-1, keepdim=True)
            # else:
            #     log_prob = distr.log_prob(target_action[:, i, :2*generator_num]).sum(-1, keepdim=True)

            '''
                Original work version
            log_prob = distr.log_prob(target_action[:, i]).sum(-1, keepdim=True)
            '''

            log_probs.append(log_prob)

        policy_log_prob = torch.cat(log_probs, dim=1)
        loss = (-target_policy * policy_log_prob).sum(1)

        # if self.config.parameters['only_power']:
        target_open_policy = (target_policy.unsqueeze(2).repeat(1, 1, one_hot_dim) * target_action[:, :, action_dim:action_dim+one_hot_dim]).sum(1).squeeze()
        target_close_policy = (target_policy.unsqueeze(2).repeat(1, 1, one_hot_dim) * target_action[:, :, action_dim+one_hot_dim:]).sum(1).squeeze()
        loss += -(torch.log_softmax(policy[:, 2*action_dim:2 * action_dim+one_hot_dim], dim=1) * target_open_policy).sum(1)
        loss += -(torch.log_softmax(policy[:, 2 * action_dim+one_hot_dim:], dim=1) * target_close_policy).sum(1)
        # else:
        #     target_open_policy = (target_policy.unsqueeze(2).repeat(1, 1, one_hot_dim) * target_action[:, :, 2*generator_num:2*generator_num+one_hot_dim]).sum(1).squeeze()
        #     target_close_policy = (target_policy.unsqueeze(2).repeat(1, 1, one_hot_dim) * target_action[:, :, 2*generator_num+one_hot_dim:]).sum(1).squeeze()
        #     loss += -(torch.log_softmax(policy[:, 4*generator_num:4*generator_num+one_hot_dim], dim=1) * target_open_policy).sum(1)
        #     loss += -(torch.log_softmax(policy[:, 4*generator_num+one_hot_dim:], dim=1) * target_close_policy).sum(1)

        ent_action = distr.rsample()
        ent_action = ent_action.clip(-0.999, 0.999)
        if self.config.parameters['only_power']:
            ent_log_prob = distr.log_prob(ent_action).sum(-1, keepdim=True)
        else:
            ent_log_prob = distr.log_prob(ent_action)[:, :generator_num].sum(-1, keepdim=True)

        entropy = - ent_log_prob.mean()
        # entropy = distr.entropy().sum(-1)
        # entropy = torch.from_numpy(np.zeros(1)).float()
        if not self.config.parameters['only_power']:
            voltage_mu, voltage_log_std = policy[:, generator_num:2*generator_num], policy[:, 3*generator_num:4*generator_num]
            mean_voltage_mu = voltage_mu.mean(-1).unsqueeze(1).repeat(1, generator_num)
            loss += torch.sum((voltage_mu - mean_voltage_mu)**2)

        return loss, entropy

    # def loss_pi_imitation(self, true_policy, pred_policy):


    def loss_pi_fn(self, hidden, target_mu, target_std):
        '''
            :param hidden:      [b, h_dim]]
        :return:
        '''

        # h = hidden.reshape(hidden.size(0), 1, hidden.size(-1)).repeat(1, target_a.size(0), 1) # [b, n, h_dim]
        # log_p = evaluate_log_pi(mu, log_std, target_a)
        # loss = (- log_p * target_p).mean()
        mu, log_std = self.model.policy(hidden)
        mu_loss = 0.5 * torch.sum((mu - target_mu) * (mu - target_mu), dim=1)
        std_loss = 0.5 * torch.sum((log_std.exp() - target_std) * (log_std.exp() - target_std), dim=1)
        loss = mu_loss + std_loss

        return loss

    def online_sampling(self):
        pass

    def offline_mcts(self, initial_obs):
        pass

    # @profile
    def continuous_update_weights(self, batch_buffer, replay_buffer, target_workers, shared_storage):
        writer = SummaryWriter(self.config.results_path)

        while ray.get(shared_storage.get_info.remote("num_played_games")) < 10:
            time.sleep(1)


        print("Here!!!")

        # if self.config.norm_type == 'min_max':
        #     self.action_min = ray.get(shared_storage.get_info.remote("action_min"))
        #     self.action_max = ray.get(shared_storage.get_info.remote("action_max"))
        # elif self.config.norm_type == 'mean_std':
        #     # self.action_mean, self.action_std = ray.get(replay_buffer.calc_action_mean_std.remote())
        #     self.action_mean = ray.get(shared_storage.get_info.remote("action_mean"))
        #     self.action_std = ray.get(shared_storage.get_info.remote("action_std"))

        # Training loop
        while self.training_step < self.config.training_steps:
            x = time.time()
            batch = batch_buffer.pop()
            if batch is None or batch == -1:
                # print(f'trainer waiting....batchQ={batch_buffer.get_len()}')
                time.sleep(0.3)
                continue
            # print(f'get_time={time.time()-x:.3f}')

            if (self.training_step+1) % 100 == 0:
                # replay_buffer.calc_mean_std.remote()
                # replay_buffer.calc_state_min_max.remote()
                # self.action_mean, self.action_std = ray.get(replay_buffer.calc_action_mean_std.remote())
                # self.action_min, self.action_max = ray.get(replay_buffer.calc_action_min_max.remote())
                # replay_buffer.calc_action_min_max.remote()
                if self.config.norm_type == 'min_max':
                    # self.action_min = ray.get(shared_storage.get_info.remote("action_min"))
                    # self.action_max = ray.get(shared_storage.get_info.remote("action_max"))
                    pass
                elif self.config.norm_type == 'mean_std':
                    # self.action_mean, self.action_std = ray.get(replay_buffer.calc_action_mean_std.remote())
                    state_mean, state_std = ray.get(replay_buffer.calc_state_mean_std.remote())
                    np.save('./state_mean.npy', state_mean)
                    np.save('./state_std.npy', state_std)


            index_batch, batch_step, batch = batch
            self.update_lr()

            x = time.time()
            if self.training_step < self.config.imitation_steps:
                if self.config.only_imitate_policy:
                    (
                        priorities,
                        total_loss,
                        value_loss,
                        reward_loss,
                        policy_loss,
                        consistency_loss,
                        entropy_loss,
                        imitation_loss,
                        # mask_loss
                    ) = self.update_weights_imitation_only_policy(batch)
                else:
                    (
                        priorities,
                        total_loss,
                        value_loss,
                        reward_loss,
                        policy_loss,
                        consistency_loss,
                        entropy_loss,
                        imitation_loss,
                        # mask_loss
                    ) = self.update_weights_imitation(batch)
                print('imitation!!')
            else:
                if not self.imitation_to_finetune:
                    print('change_flag !!!')
                    self.imitation_to_finetune = True
                    # self.config.policy_loss_coeff = 0.1
                    self.scaler = GradScaler()
                    if self.config.optimizer == 'SGD':
                        self.optimizer = optim.SGD(
                            [
                                {'params': self.model.rep_net.parameters(), 'lr': self.rep_lr_init},
                                {'params': self.model.pi_net.parameters(), 'lr': self.pi_lr_init},
                                {'params': self.model.rew_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                {'params': self.model.dyn_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                {'params': self.model.val_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                {'params': self.model.proj_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                {'params': self.model.proj_pred_net.parameters(), 'lr': self.dyn_rew_lr_init}
                            ],
                            weight_decay=self.config.weight_decay,
                            momentum=self.config.momentum
                        )
                    else:
                        self.optimizer = optim.Adam(
                            [
                                {'params': self.model.rep_net.parameters(), 'lr': self.rep_lr_init},
                                {'params': self.model.pi_net.parameters(), 'lr': self.pi_lr_init},
                                {'params': self.model.rew_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                {'params': self.model.dyn_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                {'params': self.model.val_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                {'params': self.model.proj_net.parameters(), 'lr': self.dyn_rew_lr_init},
                                {'params': self.model.proj_pred_net.parameters(), 'lr': self.dyn_rew_lr_init}
                            ],
                            weight_decay=self.config.weight_decay)
                # import ipdb
                # ipdb.set_trace()
                (
                    priorities,
                    total_loss,
                    value_loss,
                    reward_loss,
                    policy_loss,
                    consistency_loss,
                    entropy_loss,
                    imitation_loss,
                    log_info
                ) = self.update_weights(batch)

            if self.training_step % 20 == 0:
                print("Training Step:{}; Batch Step:{}; P:{:.3f}, V:{:.3f}, R:{:.3f}, C:{:.3f}, E:{:.3f}, I:{:.3f}, Time:{}, AMP={}, BatchQ={}".format(
                    self.training_step, batch_step, policy_loss, value_loss, reward_loss,
                    consistency_loss, entropy_loss, imitation_loss, time.time() - x, self.config.use_amp, batch_buffer.get_len()
                ))
                # print(f'action_batch_max={np.max(np.abs(batch[2]))}')

            if self.training_step % self.config.target_update_interval == 0:
                self.target_weight = copy.deepcopy(self.model.get_weights())
                self.target_model.set_weights(self.model.get_weights())
                shared_storage.set_info.remote({
                    "weights": copy.deepcopy(self.model.get_weights())
                })
                shared_storage.set_info.remote({
                        "target_weights": copy.deepcopy(self.target_weight),
                })
                print("Updating!")

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                # print("PRIORITIES", priorities)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step
                }
            )

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights())
                    }
                )

                shared_storage.set_info.remote(
                    {
                        "training_step": self.training_step,
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "total_loss": total_loss,
                        "value_loss": value_loss,
                        "reward_loss": reward_loss,
                        "policy_loss": policy_loss,
                        "entropy_loss": entropy_loss,
                        "consistency_loss": consistency_loss,
                        "imitation_loss": imitation_loss,
                        # "mask_loss": mask_loss
                    }
                )

            # Logs.
            if self.training_step % self.config.log_interval == 0:
                # Log the information.
                writer.add_scalar("Training/Policy_lr", self.optimizer.param_groups[0]["lr"], self.training_step)
                writer.add_scalar("Training/Dynamics_lr", self.optimizer.param_groups[2]["lr"], self.training_step)
                writer.add_scalar("Training/Total_loss", total_loss, self.training_step)
                writer.add_scalar("Training/Value_loss", value_loss, self.training_step)
                writer.add_scalar("Training/Reward_loss", reward_loss, self.training_step)
                writer.add_scalar("Training/Policy_loss", policy_loss, self.training_step)
                writer.add_scalar("Training/Entropy_loss", entropy_loss, self.training_step)
                writer.add_scalar("Training/Contrastive_loss", consistency_loss, self.training_step)
                writer.add_scalar("Training/Imitation_loss", imitation_loss, self.training_step)
                # writer.add_scalar("Training/Mask_loss", mask_loss, self.training_step)

                # num_reanalyze_games = ray.get(shared_storage.get_info.remote("num_reanalysed_games"))
                # writer.add_scalar("Worker/Reanalyzed_games", num_reanalyze_games, self.training_step)
                writer.add_scalar("Worker/Batch_buffer_size", batch_buffer.get_len(), self.training_step)

                test_episode_length = ray.get(shared_storage.get_info.remote("test_episode_length"))
                test_total_reward = ray.get(shared_storage.get_info.remote("test_total_reward"))
                test_mean_value = ray.get(shared_storage.get_info.remote("test_mean_value"))
                test_vol_vio_rate = ray.get(shared_storage.get_info.remote("test_vol_vio_rate"))
                test_reac_vio_rate = ray.get(shared_storage.get_info.remote("test_reac_vio_rate"))
                test_bal_p_vio_rate = ray.get(shared_storage.get_info.remote("test_bal_p_vio_rate"))
                test_soft_overflows = ray.get(shared_storage.get_info.remote("test_soft_overflows"))
                test_hard_overflows = ray.get(shared_storage.get_info.remote("test_hard_overflows"))
                test_running_cost = ray.get(shared_storage.get_info.remote("test_running_cost"))
                test_renewable_consumption = ray.get(shared_storage.get_info.remote("test_renewable_consumption"))
                if test_episode_length is not None:
                    writer.add_scalar("Test/test_episode_length", test_episode_length, self.training_step)
                    writer.add_scalar("Test/test_total_reward", test_total_reward, self.training_step)
                    writer.add_scalar("Test/test_mean_value", test_mean_value, self.training_step)
                    writer.add_scalar("Test/test_vol_vio_rate", test_vol_vio_rate, self.training_step)
                    writer.add_scalar("Test/test_reac_vio_rate", test_reac_vio_rate, self.training_step)
                    writer.add_scalar("Test/test_bal_p_vio_rate", test_bal_p_vio_rate, self.training_step)
                    writer.add_scalar("Test/test_soft_overflows", test_soft_overflows, self.training_step)
                    writer.add_scalar("Test/test_hard_overflows", test_hard_overflows, self.training_step)
                    writer.add_scalar("Test/test_running_cost", test_running_cost, self.training_step)
                    writer.add_scalar("Test/test_renewable_consumption", test_renewable_consumption, self.training_step)


                if self.training_step > 5e3:
                    if not self.config.imitation:
                        writer.add_histogram("Training/reward_target", batch[4][:, 1:].reshape(-1), self.training_step)
                        writer.add_histogram("Training/reward_prediction", torch.cat(log_info[0], dim=0).flatten(), self.training_step)
                        writer.add_histogram("Training/value_target", batch[3][:, 1:].reshape(-1), self.training_step)
                        writer.add_histogram("Training/value_prediction", torch.cat(log_info[1], dim=0).flatten(), self.training_step)

            # print(f'target_shape={batch[4][:, 1:].reshape(-1).shape}, pred_shape={torch.cat(log_info[0], dim=0).flatten().shape}')
            if self.training_step % self.config.log_interval == 0:
                # Log reward.
                mean_reward = ray.get(shared_storage.get_info.remote("total_reward"))
                mean_true_reward = ray.get(shared_storage.get_info.remote("total_true_reward"))
                if mean_reward is not None:
                    writer.add_scalar("Training/Mean Training Reward", mean_reward, self.training_step)

                if mean_true_reward is not None:
                    writer.add_scalar("Training/Mean Training True Reward", mean_true_reward, self.training_step)

                # Log game stats.
                num_played_steps = ray.get(shared_storage.get_info.remote("num_played_steps"))
                num_played_games = ray.get(shared_storage.get_info.remote("num_played_games"))
                writer.add_scalar("Worker/num_played_steps", num_played_steps, self.training_step)
                writer.add_scalar("Worker/num_played_games", num_played_games, self.training_step)

                state_mean = ray.get(shared_storage.get_info.remote("state_mean"))
                state_std = ray.get(shared_storage.get_info.remote("state_std"))
                writer.add_scalar("Worker/state_batch_mean", np.mean(state_mean), self.training_step)
                writer.add_scalar("Worker/state_std_min", np.min(state_std), self.training_step)

                # action_min = ray.get(shared_storage.get_info.remote("action_min"))
                # action_max = ray.get(shared_storage.get_info.remote("action_max"))
                writer.add_scalar("Worker/state_batch_min", np.min(batch[1]), self.training_step)
                writer.add_scalar("Worker/state_batch_max", np.max(batch[1]), self.training_step)
                # writer.add_scalar("Worker/action_min", min(action_min), self.training_step)
                # writer.add_scalar("Worker/action_max", max(action_max), self.training_step)
                writer.add_scalar("Worker/action_batch_min", np.min(batch[2]), self.training_step)
                writer.add_scalar("Worker/action_batch_max", np.max(batch[2]), self.training_step)

                line_overflow_mean = ray.get(shared_storage.get_info.remote("line_overflow_mean"))
                renewable_consumption_mean = ray.get(shared_storage.get_info.remote("renewable_consumption_mean"))
                running_cost_mean = ray.get(shared_storage.get_info.remote("running_cost_mean"))
                bal_gen_mean = ray.get(shared_storage.get_info.remote("bal_gen_mean"))
                reac_power_mean = ray.get(shared_storage.get_info.remote("reac_power_mean"))
                writer.add_scalar("Training/line_overflow_reward", line_overflow_mean, self.training_step)
                writer.add_scalar("Training/renewable_consumption_reward", renewable_consumption_mean, self.training_step)
                writer.add_scalar("Training/running_cost_reward", running_cost_mean, self.training_step)
                writer.add_scalar("Training/balanced_power_reward", bal_gen_mean, self.training_step)
                writer.add_scalar("Training/reactive_power_reward", reac_power_mean, self.training_step)

    def update_lr(self):
        """
        Update learning rate
        """
        # lr = self.config.lr_init * self.config.lr_decay_rate ** (
        #     self.training_step / self.config.lr_decay_steps
        # )
        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = lr

        # for param_group in self.optimizer.param_groups:
        #     param_group["lr"] = param_group["lr"] * self.config.lr_decay_rate ** (self.training_step / self.config.lr_decay_steps)

        if self.training_step < self.config.warmup_steps:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if i == 0:
                    param_group["lr"] = self.rep_lr_init * (
                            self.training_step / self.config.warmup_steps
                    )
                elif i == 1:
                    param_group["lr"] = self.pi_lr_init * (
                            self.training_step / self.config.warmup_steps
                    )
                else:
                    param_group["lr"] = self.dyn_rew_lr_init * (
                            self.training_step / self.config.warmup_steps
                    )

        if self.training_step > self.config.lr_decay_steps:
            for i, param_group in enumerate(self.optimizer.param_groups):
                if self.config.lr_decay_type == 'cosine':
                    self.scheduler.step()
                else:
                    if i == 0:
                        param_group["lr"] = self.rep_lr_init * self.config.lr_decay_rate ** (
                            (self.training_step - self.config.lr_decay_steps) / self.config.lr_decay_steps
                        )
                    elif i == 1:
                        param_group["lr"] = self.pi_lr_init * self.config.lr_decay_rate ** (
                            (self.training_step - self.config.lr_decay_steps) / self.config.lr_decay_steps
                        )
                    else:
                        param_group["lr"] = self.dyn_rew_lr_init * self.config.lr_decay_rate ** (
                            (self.training_step - self.config.lr_decay_steps) / self.config.lr_decay_steps
                        )

    # @profile
    def update_weights(self, batch):
        """
        Perform one training step.
        """
        (
            observation_batch,
            next_observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_line_overflow_reward,
            target_renewable_consumption_reward,
            target_running_cost_reward,
            target_balanced_gen_reward,
            target_reactive_power_reward,
            target_mu_batch,
            target_std_batch,
            weight_batch,
            gradient_scale_batch,
            mask_batch,
            raw_action_batch,
            raw_policy_batch,
            ready_mask_batch,
            closable_mask_batch,
            action_high_batch,
            action_low_batch
        ) = batch

        # self.model.train()

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = np.array(target_value, dtype="float32")
        priorities = np.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device

        if self.config.PER:
            weight_batch = torch.from_numpy(weight_batch.copy()).float().to(device)

        # st = time.time()

        observation_batch = torch.from_numpy(observation_batch).float().to(device)  # [B,O_SHAPE]
        next_observation_batch = torch.from_numpy(next_observation_batch).float().to(device)  # [B, UNROLL + 1, OSHAPE]
        mask_batch = torch.from_numpy(mask_batch).float().to(device)  # [B, UNROLL + 1, ASHAPE]
        action_batch = torch.from_numpy(action_batch).float().to(device)  # [B, UNROLL + 1, A_SHAPE]
        target_mu_batch = torch.from_numpy(target_mu_batch).float().to(device)  # [UNROLL_R + 1, B, A_SHAPE]
        target_std_batch = torch.from_numpy(target_std_batch).float().to(device)  # [UNROLL_R + 1, B, A_SHAPE]
        target_value = torch.from_numpy(target_value).float().to(device)  # [B, UNROLL_R + 1]
        target_reward = torch.from_numpy(target_reward).float().to(device)  # [B, UNROLL + 1]
        target_line_overflow_reward = torch.from_numpy(target_line_overflow_reward).float().to(device)
        target_renewable_consumption_reward = torch.from_numpy(target_renewable_consumption_reward).float().to(device)
        target_running_cost_reward = torch.from_numpy(target_running_cost_reward).float().to(device)
        target_balanced_gen_reward = torch.from_numpy(target_balanced_gen_reward).float().to(device)
        target_reactive_power_reward = torch.from_numpy(target_reactive_power_reward).float().to(device)
        gradient_scale_batch = torch.from_numpy(gradient_scale_batch).float().to(device)
        raw_action_batch = torch.from_numpy(raw_action_batch).float().to(device)
        raw_policy_batch = torch.from_numpy(raw_policy_batch).float().to(device)

        # assert observation_batch.max() < 2, "We assume that the observation is already processed here."
        multi_reward_targets = [target_line_overflow_reward, target_renewable_consumption_reward,
                                target_running_cost_reward, target_balanced_gen_reward, target_reactive_power_reward]
        """
            Calculating loss functions.
        """

        value_loss = 0
        reward_loss = 0
        policy_loss = 0
        consistency_loss = 0
        policy_entropy_loss = 0
        imitation_loss = 0

        if self.config.use_amp:
            with autocast():
                value, reward, policy_info, hidden_state = self.model.initial_inference(
                    observation_batch
                )
        else:
            value, reward, policy_info, hidden_state = self.model.initial_inference(
                observation_batch
            )
        # policy_info[:, :policy_info.shape[1]//2] = F.tanh(policy_info[:, :policy_info.shape[1]//2])
        value_loss += self.loss_value_fn(value, target_value[:, 0:1])
        # print("Initial Value Loss", value_loss, target_value[:, 10])
        # policy_loss += self.loss_pi_fn(hidden_state, target_mu_batch[0, ...], target_std_batch[0, ...])

        # import ipdb
        # ipdb.set_trace()
        policy_loss_0, entropy = self.loss_pi_kl_fn(policy_info, raw_action_batch[0], raw_policy_batch[0])

        policy_loss += policy_loss_0
        policy_entropy_loss -= entropy
        imitation_loss += torch.zeros_like(policy_loss)

        pred_value_scalar = torch_utils.tensor_to_scalar(
            torch_utils.support_to_scalar(value, self.config.support_size, self.config.value_support_step).squeeze()
        )

        priorities[:, 0] = np.abs(pred_value_scalar - target_value_scalar[:, 0]) ** self.config.PER_alpha

        pred_rewards = []
        pred_values = []
        for i in range(1, action_batch.shape[1]):
            """
                Calculate prediction loss iteratively.
            """
            # if self.config.norm_type == 'mean_std':
            #     action_mean = torch.from_numpy(self.action_mean).float().to('cuda')
            #     action_std = torch.from_numpy(self.action_std).float().to('cuda')
            # else:
            #     action_min = torch.from_numpy(self.action_min).float().to('cuda')
            #     action_max = torch.from_numpy(self.action_max).float().to('cuda')
            if self.config.use_amp:
                with autocast():
                    value, reward, policy_info, hidden_state = self.model.recurrent_inference(
                        hidden_state,
                        # (action_batch[:, i] - action_mean) / (action_std + 1e-6)
                        # (action_batch[:, i] - action_min) / (action_max - action_min + 1e-4) if self.config.norm_type == 'min_max' else (action_batch[:, i] - action_mean) / (action_std + 1e-4)
                        action_batch[:, i]
                    )

                    if self.config.ssl_target:
                        target_hidden = self.target_model.encode(next_observation_batch[:, i])
                    else:
                        target_hidden = self.model.encode(next_observation_batch[:, i])

                    consistency_loss += (self.loss_consistency_atomic_fn(hidden_state, target_hidden)
                                         * mask_batch[:, i:(i + 1)]).squeeze()
            else:
                value, reward, policy_info, hidden_state = self.model.recurrent_inference(
                    hidden_state,
                    # (action_batch[:, i] - action_mean) / (action_std + 1e-6)
                    # (action_batch[:, i] - action_min) / (action_max - action_min + 1e-4) if self.config.norm_type == 'min_max' else (action_batch[:, i] - action_mean) / (action_std + 1e-4)
                    action_batch[:, i]
                )

                if self.config.ssl_target:
                    target_hidden = self.target_model.encode(next_observation_batch[:, i])
                else:
                    target_hidden = self.model.encode(next_observation_batch[:, i])

                consistency_loss += (self.loss_consistency_atomic_fn(hidden_state, target_hidden)
                                     * mask_batch[:, i:(i + 1)]).squeeze()

            if self.config.multi_reward:
                for item, coeff, target_item in zip(reward, self.config.reward_coeffs, multi_reward_targets):
                    reward_loss += self.loss_reward_fn(item, target_item[:, i:i+1])
            else:
                reward_loss += self.loss_reward_fn(reward, target_reward[:, i:i + 1])

            hidden_state.register_hook(lambda grad: grad * 0.5)

            pred_values.append(support_to_scalar(value, self.config.support_size, self.config.value_support_step))
            if self.config.multi_reward:
                pred_reward = 0
                for coeff, item in zip(self.config.reward_coeffs, reward):
                    pred_reward += coeff * support_to_scalar(item, self.config.reward_support_size, self.config.reward_support_step)
                pred_rewards.append(pred_reward)
            else:
                pred_rewards.append(support_to_scalar(reward, self.config.reward_support_size, self.config.reward_support_step))
            # print('CONST', consistency_loss.shape)
            if i <= self.config.num_unroll_steps_reanalyze:
                # policy_loss += self.loss_pi_fn(hidden_state, target_mu_batch[i, ...], target_std_batch[i, ...])
                # policy_info[:, :policy_info.shape[1]//2] = F.tanh(policy_info[:, :policy_info.shape[1]//2])
                policy_loss_i, entropy = self.loss_pi_kl_fn(policy_info, raw_action_batch[i], raw_policy_batch[i])

                policy_loss += policy_loss_i
                policy_entropy_loss -= entropy

                value_loss += self.loss_value_fn(value, target_value[:, i:i + 1])

                # print('Second', self.loss_value_fn(hidden_state, target_value[:, i:i+1]))

                # print('RVM', reward_loss.shape, value_loss.shape, mask_batch.shape)
                # Scale gradient by the number of unroll steps (See paper appendix Training)
                # current_value_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
                # current_reward_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])
                # current_policy_loss.register_hook(lambda grad: grad / gradient_scale_batch[:, i])

                # Scale the gradient at the start of the dynamics function (See paper appendix Training)

                # print('VALUE SHAPE', value.shape)
                # Compute priorities for the prioritized replay (See paper appendix Training)

                pred_value_scalar = torch_utils.tensor_to_scalar(
                    torch_utils.support_to_scalar(value, self.config.support_size, self.config.value_support_step).squeeze()
                )
                priorities[:, i] = np.abs(pred_value_scalar - target_value_scalar[:, i]) ** self.config.PER_alpha

        target_loss = self.config.policy_loss_coeff * policy_loss + \
                      self.config.value_loss_coeff * value_loss + \
                      self.config.reward_loss_coeff * reward_loss + \
                      self.config.consistency_loss_coeff * consistency_loss + \
                      self.config.imitation_loss_coeff * imitation_loss

        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            target_loss *= weight_batch

        loss = target_loss.mean() + self.config.entropy_loss_coeff * policy_entropy_loss.mean()
        # loss = target_loss.mean()

        loss.register_hook(lambda grad: grad * (1 / self.config.num_unroll_steps))
        parameters = self.model.parameters()
        self.optimizer.zero_grad()

        # scaled_grad_params = torch.autograd.grad(outputs=self.scaler.scale(loss),
        #                                          inputs=self.model.parameters(),
        #                                          create_graph=True)
        # inv_scale = 1./self.scaler.get_scale()
        # grad_params = [p * inv_scale for p in scaled_grad_params]
        #
        # with autocast():
        #     grad_norm = 0
        #     for grad in grad_params:
        #         grad_norm += grad.pow(2).sum()
        #     grad_norm = grad_norm.sqrt()
        #     loss += grad_norm

        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, self.config.max_grad_norm)
        if self.config.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        self.training_step += 1

        if self.training_step % self.config.save_interval == 0:
            torch.save(
                self.model.state_dict(), os.path.join(
                    self.config.results_path,
                    'model_{}.pth'.format(self.training_step)
                )
            )

        # Save model to the disk.
        log_info = (pred_rewards, pred_values)

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
            consistency_loss.mean().item(),
            policy_entropy_loss.mean().item(),
            imitation_loss.mean().item(),
            log_info
        )
