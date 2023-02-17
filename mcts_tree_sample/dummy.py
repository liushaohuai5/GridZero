import torch
import numpy as np
import mcts_tree_sample.cytree as tree
import torch_utils
import ray
from torch_utils import *
from torch.cuda.amp import autocast
from game.gridsim.utils import *
import torch.nn.functional as F
import copy

def run_multi_support_adversarial(observations, model, config,
                                  ready_masks=None, closable_masks=None,
                                  action_highs=None, action_lows=None, origin_states=None,
                                  train_steps=0, is_attackers=None, is_test=False, root_rewards=None
                                  ):

    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base, config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>
        attacker_actions_pool = []

        _, root_reward_pred, policy_info, attacker_policy_info, roots_hidden_state = \
            model.initial_inference(torch_utils.numpy_to_tensor(observations), is_attackers.tolist())

        # if root_reward_pred.size(-1) != 1:
        #     root_reward = torch_utils.support_to_scalar(root_reward_pred, config.reward_support_size, config.reward_support_step)
        if root_rewards is not None:
            root_reward = torch_utils.numpy_to_tensor(root_rewards)

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))

        action_temp0 = model.sample_mixed_actions(policy_info, config, True,
                                                 origin_states, action_highs, action_lows, ready_masks,
                                                 closable_masks, train_steps=train_steps, is_test=is_test)
        cnt = 0
        while np.isnan(action_temp0).any():
            action_temp0 = model.sample_mixed_actions(policy_info, config, True,
                                                     origin_states, action_highs, action_lows, ready_masks,
                                                     closable_masks, train_steps=train_steps, is_test=is_test)
            cnt += 1
            # print(f'dummy error, {cnt}')

        if not config.attack_all:
            action_temp1 = np.eye(config.attacker_action_dim)
            action_temp1 = action_temp1.reshape(1, *action_temp1.shape)
            action_temp1 = action_temp1.repeat(policy_info.shape[0], axis=0)
        else:
            num = config.mcts_num_policy_samples + config.mcts_num_random_samples + config.mcts_num_expert_samples
            action_temp1 = F.gumbel_softmax(attacker_policy_info.unsqueeze(1).repeat(1, num, 1), hard=True, dim=-1).detach().cpu().numpy()

        actions_pool.append(action_temp0)
        attacker_actions_pool.append(action_temp1)

        root_actions = actions_pool[0]
        root_attacker_actions = attacker_actions_pool[0]

        prev_rewards = scalar_to_support(root_reward.unsqueeze(1), config.reward_support_size,
                                         config.reward_support_step).squeeze()

        q_values = model.eval_q(
            # torch_utils.numpy_to_tensor(observations),
                                roots_hidden_state,
                                torch_utils.numpy_to_tensor(root_actions),
                                torch_utils.numpy_to_tensor(root_attacker_actions),
                                prev_r=prev_rewards,
                                # origin_states[:, 1:55],
                                # action_highs, action_lows, ready_masks, closable_masks,
                                to_plays=is_attackers
                                )
        if root_num == 1:
            q_values = q_values.reshape(1, *q_values.shape)

        hidden_state_idx_1 = 0

        if config.efficient_imitation:
            n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples + config.mcts_num_expert_samples
        else:
            n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples

        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
            np.float32).tolist() for _ in range(root_num)]

        # During preparing, set Q_init.
        to_plays = is_attackers
        # print(f'first to plays={to_plays}')
        if not is_test:
            roots.prepare(config.root_exploration_fraction,
                          noises,
                          q_values.tolist(),
                          root_reward.reshape(-1).tolist(),
                          torch_utils.tensor_to_numpy(attacker_policy_info).tolist(),
                          to_plays.tolist())
        else:
            roots.prepare_no_noise(root_reward.reshape(-1).tolist(),
                                   q_values.tolist(),
                                   torch_utils.tensor_to_numpy(attacker_policy_info).tolist(),
                                   to_plays.tolist())

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            selected_attacker_actions = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions, to_plays = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                selected_attacker_actions.append(attacker_actions_pool[idx_1][idx_0][last_actions[ptr]])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            operator_actions = torch.from_numpy(np.asarray([action for action in selected_actions])).to('cuda').float()
            attacker_actions = torch.from_numpy(np.asarray([action for action in selected_attacker_actions])).to('cuda').float()

            leaves_value, leaves_reward_pred, leaves_policy, leaves_attacker_policy, leaves_hidden_state = \
                    model.recurrent_inference(hidden_states, operator_actions, attacker_actions, to_plays, prev_rewards)

            prev_rewards = leaves_reward_pred

            if leaves_reward_pred.size(-1) != 1:
                leaves_reward = torch_utils.support_to_scalar(leaves_reward_pred,
                                                              config.reward_support_size,
                                                              config.reward_support_step)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.value_support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))

            leaves_sampled_actions = model.sample_mixed_actions(leaves_policy, config, is_root=False, train_steps=train_steps, is_test=is_test)
            cnt = 0
            while np.isnan(leaves_sampled_actions).any():
                leaves_sampled_actions = model.sample_mixed_actions(leaves_policy, config, is_root=False, train_steps=train_steps, is_test=is_test)
                cnt += 1
                # print(f'dummy error, {cnt}')

            if not config.attack_all:
                leaves_sampled_actions1 = np.eye(config.attacker_action_dim)
                leaves_sampled_actions1 = leaves_sampled_actions1.reshape(1, *leaves_sampled_actions1.shape)
                leaves_sampled_actions1 = leaves_sampled_actions1.repeat(hidden_states.shape[0], axis=0)
            else:
                num = config.mcts_num_policy_samples + config.mcts_num_random_samples + config.mcts_num_expert_samples
                leaves_sampled_actions1 = F.gumbel_softmax(leaves_attacker_policy.unsqueeze(1).repeat(1, num, 1), hard=True,
                                                dim=-1).detach().cpu().numpy()

            actions_pool.append(leaves_sampled_actions)
            attacker_actions_pool.append(leaves_sampled_actions1)
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value,
                                      min_max_stats_lst, results,
                                      to_plays,
                                      torch_utils.tensor_to_numpy(leaves_attacker_policy).tolist(),
                                      int(config.add_attacker))


    return roots.get_values(), roots.get_distributions(), \
           actions_pool[0], attacker_actions_pool[0]