import torch
import numpy as np
import mcts_tree_sample.cytree as tree
import torch_utils
import ray
from torch_utils import profile
from torch.cuda.amp import autocast
from game.gridsim.utils import *

# @profile
def run_multi_support(observations, model, config,
                      ready_masks=None, closable_masks=None,
                      action_highs=None, action_lows=None, origin_states=None,
                      # expert_model=None,
                      train_steps=0
                      ):
    root_num = observations.shape[0]

    with torch.no_grad():
        model.eval()

        pb_c_base, pb_c_init, discount = config.pb_c_base, config.pb_c_init, config.discount

        hidden_states_pool = []  # [NODE_ID, BATCHSIZE, H_DIM], CUDA_TENSORS
        actions_pool = []        # [NODE_ID, BATCHSIZE, N_ACTION, ACTION_DIM], CUDA_TENSORS>
        real_gen_ps_pool = []

        if config.use_amp:
            with autocast():
                _, root_reward, policy_info, roots_hidden_state = \
                   model.initial_inference(torch_utils.numpy_to_tensor(observations))
                # if expert_model is not None:
                #     _, _, expert_policy, _ = expert_model.initial_inference(torch.utils.numpy_to_tensor(observations))
        else:
            _, root_reward, policy_info, roots_hidden_state = \
                model.initial_inference(torch_utils.numpy_to_tensor(observations))
            # if expert_model is not None:
            #     _, _, expert_policy, _ = expert_model.initial_inference(torch.utils.numpy_to_tensor(observations))

        if root_reward.size(-1) != 1:
            root_reward = torch_utils.support_to_scalar(root_reward,
                                                        config.reward_support_size,
                                                        config.reward_support_step)

        # root_reward shape          [256]
        # roots_hidden_state_shape = [[256, h]]

        hidden_states_pool.append(torch_utils.tensor_to_numpy(roots_hidden_state))
        real_gen_ps_pool.append(origin_states[:, 1:55])

        if config.use_amp:
            with autocast():
                action_temp = model.sample_mixed_actions(policy_info, config, True,
                                                       origin_states, action_highs, action_lows, ready_masks, closable_masks, train_steps=train_steps)
        else:
            action_temp = model.sample_mixed_actions(policy_info, config, True,
                                                     origin_states, action_highs, action_lows, ready_masks,
                                                     closable_masks, train_steps=train_steps)

        cnt=0
        while np.isnan(action_temp).any():
            # idx_0, idx_1, idx_2 = np.where(np.isnan(action_temp))
            # for i_0 in idx_0:
            #     for i_1 in idx_1:
            #         root_actions[i_0, i_1, :] = root_actions[i_0, i_1-1, :]
            if config.use_amp:
                with autocast():
                    action_temp = model.sample_mixed_actions(policy_info, config, True,
                                                             origin_states, action_highs, action_lows, ready_masks,
                                                             closable_masks, train_steps=train_steps)
            else:
                action_temp = model.sample_mixed_actions(policy_info, config, True,
                                                         origin_states, action_highs, action_lows, ready_masks,
                                                         closable_masks, train_steps=train_steps)
            cnt+=1
            print(f'dummy error, {cnt}')
        actions_pool.append(action_temp)

        hidden_state_idx_1 = 0

        if config.efficient_imitation:
            n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples + config.mcts_num_expert_samples
        else:
            n_total_actions = config.mcts_num_policy_samples + config.mcts_num_random_samples

        roots = tree.Roots(root_num, n_total_actions, config.num_simulations)
        noises = [np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size).astype(
            np.float32).tolist() for _ in range(root_num)]
        # noises = [np.zeros_like(np.random.dirichlet([config.root_dirichlet_alpha] * config.action_space_size)).astype(np.float32).tolist() for _ in range(root_num)]

        root_actions = actions_pool[0]

        if config.use_amp:
            with autocast():
                q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                        torch_utils.numpy_to_tensor(root_actions),
                                        origin_states[:, 1:55],
                                        action_highs, action_lows, ready_masks, closable_masks
                                        )
        else:
            q_values = model.eval_q(torch_utils.numpy_to_tensor(observations),
                                    torch_utils.numpy_to_tensor(root_actions),
                                    origin_states[:, 1:55],
                                    action_highs, action_lows, ready_masks, closable_masks
                                    )

        if root_num == 1:
            q_values = q_values.reshape(1, *q_values.shape)
        # print('batch_worker eval_q NAN', torch.any(torch.isnan(q_values)).tolist())
        # print('Q_value', q_values.abs().max())

        # During preparing, set Q_init.
        # roots.prepare(config.root_exploration_fraction,
        #               noises,
        #               q_values.tolist(),
        #               root_reward.reshape(-1).tolist())
        roots.prepare_no_noise(
            root_reward.reshape(-1).tolist(), q_values.tolist()
        )

        min_max_stats_lst = tree.MinMaxStatsList(root_num)
        min_max_stats_lst.set_delta(0.01)

        for index_simulation in range(config.num_simulations):
            hidden_states = []
            selected_actions = []
            real_gen_ps = []
            results = tree.ResultsWrapper(root_num)
            data_idxes_0, data_idxes_1, last_actions = \
                tree.multi_traverse(roots, pb_c_base, pb_c_init, discount, min_max_stats_lst, results)
            # TODO: add return of is_error, add ipdb after flag.

            ptr = 0
            for idx_0, idx_1 in zip(data_idxes_0, data_idxes_1):
                hidden_states.append(hidden_states_pool[idx_1][idx_0])
                selected_actions.append(actions_pool[idx_1][idx_0][last_actions[ptr]])
                real_gen_ps.append(real_gen_ps_pool[idx_1][idx_0])
                ptr += 1

            hidden_states = torch.from_numpy(np.asarray(hidden_states)).to('cuda').float()
            # selected_actions = torch.from_numpy(np.asarray(selected_actions)).to('cuda').float()
            selected_actions = np.asarray(selected_actions)
            real_gen_ps = np.asarray(real_gen_ps)

            modified_adjust_gen = action_mapping(selected_actions, config, action_highs, action_lows, ready_masks, closable_masks)

            real_gen_ps += modified_adjust_gen
            selected_actions = torch.from_numpy(selected_actions).float().to('cuda')

            if config.use_amp:
                with autocast():
                    leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                        model.recurrent_inference(hidden_states, selected_actions)
            else:
                leaves_value, leaves_reward, leaves_policy, leaves_hidden_state = \
                    model.recurrent_inference(hidden_states, selected_actions)

            if config.multi_reward:
                leaf_reward = 0
                cnt = 0
                for coeff, reward_item in zip(config.reward_coeffs, leaves_reward):
                    if cnt == 2 and config.ground_truth_running_cost_reward:
                        leaf_reward += coeff * torch.from_numpy(calc_running_cost_rew(real_gen_ps)).unsqueeze(1).float().to('cuda')
                    else:
                        leaf_reward += coeff * torch_utils.support_to_scalar(reward_item,
                                                                     config.reward_support_size,
                                                                     config.reward_support_step)
                    cnt += 1
                leaves_reward = leaf_reward
            else:
                if leaves_reward.size(-1) != 1:
                    leaves_reward = torch_utils.support_to_scalar(leaves_reward,
                                                                  config.reward_support_size,
                                                                  config.reward_support_step)

            leaves_reward = leaves_reward.reshape(-1).tolist()

            if leaves_value.size(-1) != 1:
                leaves_value = torch_utils.support_to_scalar(leaves_value, config.support_size, config.value_support_step)

            leaves_value = leaves_value.reshape(-1).tolist()

            # Update the database
            hidden_states_pool.append(torch_utils.tensor_to_numpy(leaves_hidden_state))
            real_gen_ps_pool.append(real_gen_ps)
            leaves_sampled_actions = model.sample_mixed_actions(leaves_policy, config, False, train_steps=train_steps)

            cnt=0
            while np.isnan(leaves_sampled_actions).any():
                leaves_sampled_actions = model.sample_mixed_actions(leaves_policy, config, False, train_steps=train_steps)
                cnt += 1
                print(f'dummy error, {cnt}')

            actions_pool.append(leaves_sampled_actions)
            hidden_state_idx_1 += 1

            # Back-propagate the reward information.
            tree.multi_back_propagate(hidden_state_idx_1, discount, leaves_reward,
                                      leaves_value, min_max_stats_lst, results)

    return roots.get_values(), roots.get_distributions(), actions_pool[0]

