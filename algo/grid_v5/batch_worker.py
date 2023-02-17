import copy
import numpy
import numpy as np
import ray
from model.utils import *
import time
from ray.util.queue import Queue
from algo.grid_v5.rzero import MLPModel
from algo.grid_v5.self_play import GameHistory
from mcts_tree_sample.dummy import run_multi_support_adversarial
from torch_utils import profile
from torch.cuda.amp import autocast


class BatchBufferFast(object):
    def __init__(self, threshold=20, size=25):
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, items):
        for item in items:
            if self.queue.qsize() <= self.threshold:
                self.queue.put(item)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()

        return None

    def get_len(self):
        return self.queue.qsize()


def calculate_mean_std(val, visit):
    '''

    :param val: [B, N, VAL]
    :param visit: [B, N]. The last dim is the visit count of each value.
    :return:
    '''
    # print(val.shape, visit.shape)
    visit = visit / np.sum(visit, axis=-1, keepdims=True)
    visit = visit.reshape(*visit.shape, -1)

    val_mean = np.sum(visit * val, axis=-2)
    val_mean_result = np.array(val_mean)

    val_mean = val_mean.reshape((val_mean.shape[0], 1, val_mean.shape[-1]))
    val_mean = val_mean.repeat(val.shape[1], axis=1)

    val_dev = (val_mean - val) * (val_mean - val)
    val_dev_mean = np.sqrt(np.sum(visit * val_dev, axis=-2))

    return val_mean_result, val_dev_mean


# @ray.remote(num_gpus=0.08)
@ray.remote(num_gpus=0.12)
class LowDimFastBatchTargetWorker:
    """
    Class which run in a dedicated thread to calculate the mini-batch for training
    """

    def __init__(self, rank, initial_checkpoint, batch_buffer, replay_buffer, shared_storage, config):
        self.rank = rank
        self.config = config
        self.batch_buffer = batch_buffer
        self.replay_buffer = replay_buffer
        self.shared_storage = shared_storage
        self.buffer = []
        self.model = MLPModel(config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["target_weights"]))
        self.model.to("cuda")
        self.model.eval()
        self.weight_step = 0
        # Fix random generator seed
        numpy.random.seed(self.config.seed)

        self.last_target_update_index = -1

    def set_weights(self, weights, weight_step=0):
        print("Setting weight!", weight_step)
        self.model.set_weights(weights)
        self.weight_step = weight_step
        return self.weight_step

    # @profile
    def calculate_bootstrap(self, observations, masks,
                            ready_masks=None, closable_masks=None,
                            action_highs=None, action_lows=None, origin_states=None, train_steps=0, is_attackers=None,
                            root_rewards=None
                            ):
        """
        Run several MCTS for the given observations.

        :param observations:
        :param masks:
        :return:
        """
        # print('BOOTSTRAP_CALC', observations.shape)
        root_rewards = root_rewards.squeeze()
        root_values, root_distributions, root_actions, root_attacker_actions = run_multi_support_adversarial(
            observations, self.model, self.config, ready_masks, closable_masks, action_highs, action_lows,
            origin_states, train_steps=train_steps, is_attackers=is_attackers, root_rewards=root_rewards)
        values = np.array(root_values)
        actions = np.array(root_actions)
        attacker_actions = np.array(root_attacker_actions)
        policies = np.array(root_distributions).astype(np.float32)
        policies /= policies.sum(axis=-1, keepdims=True)

        values = values * masks
        # [256, ], [256, N, ACTION_DIM], [256, N]
        return values, actions, attacker_actions, policies

    def buffer_add_batch(self, batch):
        self.buffer.append(batch)
        if len(self.buffer) > 5:
            self.buffer = self.buffer[1:]

    def buffer_get_batch(self):
        if len(self.buffer) > 0:
            data = copy.deepcopy(self.buffer[0])
            self.buffer = self.buffer[1:]
            return data
        return None

    def spin(self):
        while ray.get(self.shared_storage.get_info.remote("num_played_games")) < 5:  # previous 5
            time.sleep(1)

        if self.config.efficient_imitation:
            while ray.get(self.shared_storage.get_info.remote("num_expert_games")) < 1:
                time.sleep(1)

        print("start making batches...")

        batches = []
        while True:

            training_step = ray.get(self.shared_storage.get_info.remote("training_step"))
            target_update_index = training_step // self.config.target_update_interval
            # try:
                # x = time.time()
            if target_update_index > self.last_target_update_index:
                self.last_target_update_index = target_update_index
                target_weights = ray.get(self.shared_storage.get_info.remote("target_weights"))
                self.model.set_weights(target_weights)
                self.model.to('cuda')
                self.model.eval()
                print("batch worker model updated !!!")
            batch = self.get_batch(train_steps=training_step)
            self.batch_buffer.push([batch])
                # print(f'prepare batch time:{time.time()-x:.3f}')
            # except:
            #     continue
                # print(f'batchQ={self.batch_buffer.get_len()}')

            # time.sleep(0.2)

    # @profile
    def get_batch(self, train_steps=0):
        """
        :return:
        """
        (
            index_batch,
            observation_batch,
            next_observation_batch,
            action_batch,
            attacker_action_batch,
            reward_batch,
            line_overflow_reward_batch,
            renewable_consumption_reward_batch,
            running_cost_reward_batch,
            balanced_gen_reward_batch,
            reactive_power_reward_batch,
            value_batch,
            target_mu_batch,
            target_std_batch,
            gradient_scale_batch,
            raw_actions_batch,
            raw_attacker_action_batch,
            raw_policies_batch,
            mask_batch,
            ready_mask_batch,
            closable_mask_batch,
            action_high_batch,
            action_low_batch,
            attacker_flag_batch
        ) = ([], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], [], [], [], [], [], [], [], [], []
             )

        weight_batch = [] if self.config.PER else None
        # x = time.time()
        game_samples = ray.get(self.replay_buffer.sample_n_games.remote(self.config.batch_size, force_uniform=False))
        n_total_samples = len(game_samples)

        """ 
            Reanalyzed
        """
        all_bootstrap_values = []
        for i in range(self.config.num_unroll_steps_reanalyze + 1):
            # print(f'reanalyze {i}')

            begin_observation = []
            begin_origin_state = []
            begin_ready_mask = []
            begin_closable_mask = []
            begin_action_high = []
            begin_action_low = []
            begin_attacker_flags = []
            begin_reward = []

            bootstrap_observation = []
            bootstrap_origin_state = []
            bootstrap_mask = []
            bootstrap_ready_mask = []
            bootstrap_closable_mask = []
            bootstrap_action_high = []
            bootstrap_action_low = []
            bootstrap_attacker_flags = []
            bootstrap_reward = []

            for (game_id, game_history, game_prob, game_pos, pos_prob) in game_samples:
                game_history = game_history.subset(max([0, game_pos - self.config.stacked_observations + 1]), 16)

                begin_index = self.config.stacked_observations - 1 + i
                bootstrap_index = self.config.stacked_observations - 1 + i + self.config.td_steps
                begin_observation.append(game_history.get_stacked_observations(
                        begin_index, self.config.stacked_observations)
                )
                begin_origin_state.append(
                    game_history.get_stacked_origin_states(
                        begin_index, self.config.stacked_observations
                    )
                )
                ready_masks, closable_masks = game_history.get_stacked_masks(
                    begin_index, self.config.stacked_observations
                )
                begin_ready_mask.append(ready_masks)
                begin_closable_mask.append(closable_masks)

                action_highs, action_lows = game_history.get_stacked_action_space(
                    begin_index, self.config.stacked_observations
                )
                begin_action_high.append(action_highs)
                begin_action_low.append(action_lows)

                begin_attacker_flags.append(
                    game_history.get_stacked_attacker_flag(
                        begin_index, self.config.stacked_observations
                    )
                )

                begin_reward.append(
                    game_history.get_stacked_reward(
                        begin_index, self.config.stacked_observations
                    )
                )

                if bootstrap_index > len(game_history.root_values):
                    bootstrap_mask.append(0)
                    bootstrap_observation.append(
                        game_history.get_stacked_observations(
                        0, self.config.stacked_observations)
                    )
                    bootstrap_origin_state.append(
                        game_history.get_stacked_origin_states(
                            0, self.config.stacked_observations
                        )
                    )
                    ready_masks, closable_masks = game_history.get_stacked_masks(
                        0, self.config.stacked_observations
                    )
                    bootstrap_ready_mask.append(ready_masks)
                    bootstrap_closable_mask.append(closable_masks)

                    action_highs, action_lows = game_history.get_stacked_action_space(
                        0, self.config.stacked_observations
                    )
                    bootstrap_action_high.append(action_highs)
                    bootstrap_action_low.append(action_lows)

                    bootstrap_attacker_flags.append(
                        game_history.get_stacked_attacker_flag(
                            0, self.config.stacked_observations
                        )
                    )

                    bootstrap_reward.append(
                        game_history.get_stacked_reward(
                            0, self.config.stacked_observations
                        )
                    )

                else:
                    bootstrap_mask.append(1)
                    bootstrap_observation.append(
                        game_history.get_stacked_observations(
                            bootstrap_index, self.config.stacked_observations
                        )
                    )
                    bootstrap_origin_state.append(
                        game_history.get_stacked_origin_states(
                            bootstrap_index, self.config.stacked_observations
                        )
                    )
                    ready_masks, closable_masks = game_history.get_stacked_masks(
                        bootstrap_index, self.config.stacked_observations
                    )
                    bootstrap_ready_mask.append(ready_masks)
                    bootstrap_closable_mask.append(closable_masks)

                    action_highs, action_lows = game_history.get_stacked_action_space(
                        bootstrap_index, self.config.stacked_observations
                    )
                    bootstrap_action_high.append(action_highs)
                    bootstrap_action_low.append(action_lows)

                    bootstrap_attacker_flags.append(
                        game_history.get_stacked_attacker_flag(
                            bootstrap_index, self.config.stacked_observations
                        )
                    )

                    bootstrap_reward.append(
                        game_history.get_stacked_reward(
                            bootstrap_index, self.config.stacked_observations
                        )
                    )

            bootstrap_mask = np.array(bootstrap_mask)
            bootstrap_observation = np.array(bootstrap_observation)
            bootstrap_origin_state = np.array(bootstrap_origin_state)
            bootstrap_ready_mask = np.array(bootstrap_ready_mask)
            bootstrap_closable_mask = np.array(bootstrap_closable_mask)
            bootstrap_action_high = np.array(bootstrap_action_high)
            bootstrap_action_low = np.array(bootstrap_action_low)
            bootstrap_attacker_flags = np.array(bootstrap_attacker_flags).astype(np.int32).squeeze()
            bootstrap_reward = np.array(bootstrap_reward)

            # x = time.time()
            bootstrap_values, _, _, _ = self.calculate_bootstrap(bootstrap_observation, bootstrap_mask,
                bootstrap_ready_mask, bootstrap_closable_mask, bootstrap_action_high, bootstrap_action_low,
                bootstrap_origin_state, train_steps=train_steps, is_attackers=bootstrap_attacker_flags,
                root_rewards=bootstrap_reward)
            # print(f'bootstrap time:{time.time() - x:.3f}')

            # Reanalyze result.
            all_bootstrap_values.append(bootstrap_values.reshape(*bootstrap_values.shape, 1))

            begin_observation = np.array(begin_observation)
            begin_origin_state = np.array(begin_origin_state)
            begin_ready_mask = np.array(begin_ready_mask)
            begin_closable_mask = np.array(begin_closable_mask)
            begin_action_high = np.array(begin_action_high)
            begin_action_low = np.array(begin_action_low)
            begin_mask = np.ones(begin_observation.shape[0])
            begin_attacker_flags = np.array(begin_attacker_flags).astype(np.int32).squeeze()
            begin_reward = np.array(begin_reward)

            # x = time.time()
            _, begin_actions, begin_attacker_actions, begin_policies = self.calculate_bootstrap(
                begin_observation, begin_mask, begin_ready_mask, begin_closable_mask, begin_action_high,
                begin_action_low, begin_origin_state, train_steps=train_steps, is_attackers=begin_attacker_flags,
                root_rewards=begin_reward)
            # print(f'begin time:{time.time()-x:.3f}')

            raw_actions_batch.append(begin_actions)
            raw_attacker_action_batch.append(begin_attacker_actions)
            raw_policies_batch.append(begin_policies)
            policy_mu, policy_std = calculate_mean_std(begin_actions, begin_policies)
            target_mu_batch.append(policy_mu)
            target_std_batch.append(policy_std)

        all_bootstrap_values = np.concatenate(all_bootstrap_values, axis=1)
        # print('BS Value', all_bootstrap_values)
        """
            Compute the targets.
        """
        for idx, (game_id, game_history, game_prob, game_pos, pos_prob) in enumerate(game_samples):

            game_history = game_history.subset(max([0, game_pos - self.config.stacked_observations + 1]), 16)

            values, rewards, line_overflow_rewards, renewable_consumption_rewards, \
            running_cost_rewards, balanced_gen_rewards, reactive_power_rewards, \
            actions, attacker_actions, ready_masks, closable_masks, action_highs, action_lows, \
            next_observations, masks, attacker_flags = self.make_target(
                game_history, self.config.stacked_observations - 1, all_bootstrap_values[idx]
            )

            index_batch.append([game_id, game_pos])

            observation_batch.append(
                game_history.get_stacked_observations(
                    self.config.stacked_observations - 1, self.config.stacked_observations
                )
            )

            next_observation_batch.append(next_observations)
            action_batch.append(actions)
            attacker_action_batch.append(attacker_actions)
            ready_mask_batch.append(ready_masks)
            closable_mask_batch.append(closable_masks)
            action_high_batch.append(action_highs)
            action_low_batch.append(action_lows)
            value_batch.append(values)
            reward_batch.append(rewards)
            line_overflow_reward_batch.append(line_overflow_rewards)
            renewable_consumption_reward_batch.append(renewable_consumption_rewards)
            running_cost_reward_batch.append(running_cost_rewards)
            balanced_gen_reward_batch.append(balanced_gen_rewards)
            reactive_power_reward_batch.append(reactive_power_rewards)
            mask_batch.append(masks)
            attacker_flag_batch.append(attacker_flags)

            gradient_scale_batch.append(
                [
                    min(
                        self.config.num_unroll_steps,
                        len(game_history.action_history) - (self.config.stacked_observations - 1),
                    )
                ]
                * len(actions)
            )

            if self.config.PER:
                weight_batch.append((n_total_samples * game_prob * pos_prob) ** (-self.config.PER_beta))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        observation_batch = np.array(observation_batch)
        next_observation_batch = np.array(next_observation_batch)
        mask_batch = np.array(mask_batch)
        action_batch = np.array(action_batch)
        attacker_action_batch = np.array(attacker_action_batch)
        ready_mask_batch = np.array(ready_mask_batch)
        closable_mask_batch = np.array(closable_mask_batch)
        action_high_batch = np.array(action_high_batch)
        action_low_batch = np.array(action_low_batch)
        value_batch = np.array(value_batch)
        reward_batch = np.array(reward_batch)
        # line_overflow_reward_batch = np.array(line_overflow_reward_batch)
        # renewable_consumption_reward_batch = np.array(renewable_consumption_reward_batch)
        # running_cost_reward_batch = np.array(running_cost_reward_batch)
        # balanced_gen_reward_batch = np.array(balanced_gen_reward_batch)
        # reactive_power_reward_batch = np.array(reactive_power_reward_batch)
        # target_mu_batch = np.array(target_mu_batch)
        # target_std_batch = np.array(target_std_batch)
        weight_batch = np.array(weight_batch)
        # gradient_scale_batch = np.array(gradient_scale_batch)
        raw_actions_batch = np.array(raw_actions_batch)
        raw_attacker_action_batch = np.array(raw_attacker_action_batch)
        raw_policies_batch = np.array(raw_policies_batch)
        attacker_flag_batch = np.array(attacker_flag_batch)
     #   print('TOT', time.time() - x)

        return (
            index_batch,
            self.weight_step,
            (
                observation_batch,          # [B, O_DIM]             * s0
                next_observation_batch,     # [B, STEP + 1, O_DIM]   * s0,    s1,     s2,     ..., s_unroll
                action_batch,               # [B, STEP + 1, A_DIM]   * X ,    a0,     a1,     ..., a_unroll-1
                attacker_action_batch,
                value_batch,                # [B, STEP_R + 1]
                reward_batch,               # [B, STEP + 1]          * X,     r0,     r1,     ...,
                # line_overflow_reward_batch,
                # renewable_consumption_reward_batch,
                # running_cost_reward_batch,
                # balanced_gen_reward_batch,
                # reactive_power_reward_batch,
                # target_mu_batch,            # [B, STEP_R + 1, A_DIM]   mu_0,  mu_1,   mu_2,   ...,
                # target_std_batch,           # [B, STEP_R + 1, A_DIM]   std_0, std_1,  std_2,  ...,
                weight_batch,               # ...
                # gradient_scale_batch,       # ...
                mask_batch,                  # [B, STEP + 1]
                raw_actions_batch,
                raw_attacker_action_batch,
                raw_policies_batch,
                ready_mask_batch,
                closable_mask_batch,
                action_high_batch,
                action_low_batch,
                attacker_flag_batch
            )
        )

    def compute_target_value(self, game_history, index=0, bootstrap_value=None):
        bootstrap_index = index + self.config.td_steps
        if bootstrap_value is not None:
            value = bootstrap_value * self.config.discount ** self.config.td_steps
        else:

            if bootstrap_index < len(game_history.root_values):
            # if bootstrap_index < len(game_history.observation_history):
                root_values = (
                    game_history.root_values
                    if game_history.reanalysed_predicted_root_values is None
                    else game_history.reanalysed_predicted_root_values
                )

                last_step_value = (
                    root_values[bootstrap_index]
                )
                value = last_step_value * self.config.discount ** self.config.td_steps
            else:
                value = 0

        for i, reward in enumerate(
            game_history.reward_history[index + 1: bootstrap_index + 1]
        ):
            # The value is oriented from the perspective of the current player
            value += (reward) * self.config.discount ** i

        return value

    def make_target(self, game_history, state_index=0, bootstrap_value=None):
        """
        Generate targets for every unroll steps.
        """
        target_values = []
        target_rewards = []

        target_line_of_rewards = []
        target_renewable_consump_rewards = []
        target_run_cost_rewards = []
        target_bal_gen_rewards = []
        target_reac_pow_rewards = []

        actions = []
        actions_attacker = []
        ready_masks = []
        closable_masks = []
        action_highs = []
        action_lows = []
        target_masks = []
        target_next_observations = []
        attacker_flags = []
        """
            Target actions,     Target policies
            -----------------------------------
            action_vec_0        scalar_prob_0
            action_vec_1        scalar_prob_1
            action_vec_2        scalar_prob_2
            ...                 ...
        """

        """
            Obs_cur
            target_action_cur-1, target_action_cur,
        """

        # [UNROLL, NUM_ACTION]
        # [UNROLL, NUM_ACTION, ACTION_DIM]
        # is_attacker = False
        for current_index in range(state_index, state_index + self.config.num_unroll_steps + 1):
           #print('Current Calculation', current_index - state_index)
            if current_index - state_index <= self.config.num_unroll_steps_reanalyze:
                calculate_value = True
                value = self.compute_target_value(game_history, current_index,
                                                  bootstrap_value[current_index - state_index])
               # print('Current Calculation', current_index - state_index,
                  #    bootstrap_value[current_index - state_index], value)
            else:
                calculate_value = False

            if current_index < len(game_history.root_values):
                if calculate_value:
                    target_values.append(value)

                target_rewards.append(game_history.reward_history[current_index])

                target_line_of_rewards.append(game_history.line_overflow_rewards[current_index])
                target_renewable_consump_rewards.append(game_history.renewable_consumption_rewards[current_index])
                target_run_cost_rewards.append(game_history.running_cost_rewards[current_index])
                target_bal_gen_rewards.append(game_history.balanced_gen_rewards[current_index])
                target_reac_pow_rewards.append(game_history.reactive_power_rewards[current_index])

                target_next_observations.append(
                    game_history.get_stacked_observations(
                        current_index,
                        self.config.stacked_observations)
                )

                target_masks.append(1.)
                actions.append(game_history.action_history[current_index])
                actions_attacker.append(game_history.attacker_action_history[current_index])
                ready_masks.append(game_history.ready_mask_history[current_index])
                closable_masks.append(game_history.closable_mask_history[current_index])
                action_highs.append(game_history.action_high_history[current_index])
                action_lows.append(game_history.action_low_history[current_index])
                attacker_flags.append(game_history.attacker_flag_history[current_index])

            elif current_index == len(game_history.root_values):
                if calculate_value:
                    target_values.append(0)
                target_rewards.append(game_history.reward_history[current_index])

                target_line_of_rewards.append(game_history.line_overflow_rewards[current_index])
                target_renewable_consump_rewards.append(game_history.renewable_consumption_rewards[current_index])
                target_run_cost_rewards.append(game_history.running_cost_rewards[current_index])
                target_bal_gen_rewards.append(game_history.balanced_gen_rewards[current_index])
                target_reac_pow_rewards.append(game_history.reactive_power_rewards[current_index])

                target_next_observations.append(
                    game_history.get_stacked_observations(
                        0, self.config.stacked_observations)
                )
                actions.append(game_history.action_history[current_index])
                actions_attacker.append(game_history.attacker_action_history[current_index])
                ready_masks.append(game_history.ready_mask_history[current_index])
                closable_masks.append(game_history.closable_mask_history[current_index])
                action_highs.append(game_history.action_high_history[current_index])
                action_lows.append(game_history.action_low_history[current_index])
                target_masks.append(0.)
                attacker_flags.append(game_history.attacker_flag_history[current_index])
                # is_attacker = not bool(game_history.attacker_flag_history[current_index])
            else:
                if calculate_value:
                    target_values.append(0)

                target_rewards.append(0)

                target_line_of_rewards.append(0)
                target_renewable_consump_rewards.append(0)
                target_run_cost_rewards.append(0)
                target_bal_gen_rewards.append(0)
                target_reac_pow_rewards.append(0)

                target_next_observations.append(
                    game_history.get_stacked_observations(
                        0, self.config.stacked_observations)
                )
                if len(game_history.child_visits) > 0:
                    random_actions = self.config.sample_random_actions(len(game_history.child_visits[0]),
                                                                       is_attacker=False)
                    random_actions1 = self.config.sample_random_actions(len(game_history.child_visits[0]),
                                                                       is_attacker=True)
                else:
                    random_actions = self.config.sample_random_actions(self.config.mcts_num_policy_samples +
                                                                       self.config.mcts_num_random_samples +
                                                                       self.config.mcts_num_expert_samples,
                                                                       is_attacker=False)
                    random_actions1 = self.config.sample_random_actions(self.config.mcts_num_policy_samples +
                                                                       self.config.mcts_num_random_samples +
                                                                       self.config.mcts_num_expert_samples,
                                                                       is_attacker=True)
                actions.append(random_actions[0])
                actions_attacker.append(random_actions1[0])
                ready_masks.append(game_history.ready_mask_history[0])
                closable_masks.append(game_history.closable_mask_history[0])
                action_highs.append(game_history.action_high_history[0])
                action_lows.append(game_history.action_low_history[0])
                target_masks.append(0.)
                last_flag = attacker_flags[-1]
                attacker_flags.append(int(1 - last_flag) if self.config.add_attacker else 0)

        target_values = np.array(target_values)
        target_rewards = np.array(target_rewards)

        target_line_of_rewards = np.array(target_line_of_rewards)
        target_renewable_consump_rewards = np.array(target_renewable_consump_rewards)
        target_run_cost_rewards = np.array(target_run_cost_rewards)
        target_bal_gen_rewards = np.array(target_bal_gen_rewards)
        target_reac_pow_rewards = np.array(target_reac_pow_rewards)

        target_masks = np.array(target_masks)
        target_next_observations = np.array(target_next_observations)
        actions = np.array(actions)
        ready_masks = np.array(ready_masks)
        closable_masks = np.array(closable_masks)
        action_highs = np.array(action_highs)
        action_lows = np.array(action_lows)
        attacker_flags = np.array(attacker_flags)

        return target_values, target_rewards, \
               target_line_of_rewards, target_renewable_consump_rewards, target_run_cost_rewards, \
               target_bal_gen_rewards, target_reac_pow_rewards, \
               actions, actions_attacker, ready_masks, closable_masks, action_highs, action_lows, \
               target_next_observations, target_masks, attacker_flags


