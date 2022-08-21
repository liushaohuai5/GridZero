import copy
import numpy
import numpy as np
import ray
from model.utils import *
import time
from ray.util.queue import Queue
from algo.grid_v5.rzero import MLPModel
from algo.grid_v5.self_play import GameHistory
from mcts_tree_sample.dummy import run_multi_support
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
        self.model.to(torch.device("cuda"))
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
                            action_highs=None, action_lows=None, origin_states=None, train_steps=0
                            ):
        """
        Run several MCTS for the given observations.

        :param observations:
        :param masks:
        :return:
        """
        # print('BOOTSTRAP_CALC', observations.shape)
        root_values, root_distributions, root_actions = run_multi_support(observations, self.model, self.config,
                                                                          ready_masks, closable_masks,
                                                                          action_highs, action_lows, origin_states, train_steps=train_steps
                                                                          )
        values = np.array(root_values)
        actions = np.array(root_actions)
        policies = np.array(root_distributions).astype(np.float32)
        policies /= policies.sum(axis=-1, keepdims=True)

        values = values * masks
        # [256, ], [256, N, ACTION_DIM], [256, N]
        return values, actions, policies

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
        while ray.get(self.shared_storage.get_info.remote("num_played_games")) < 10:
            time.sleep(1)

        batches = []
        while True:

            training_step = ray.get(self.shared_storage.get_info.remote("training_step"))
            target_update_index = training_step // self.config.target_update_interval
            if target_update_index > self.last_target_update_index:
                self.last_target_update_index = target_update_index
                target_weights = ray.get(self.shared_storage.get_info.remote("target_weights"))
                self.model.set_weights(target_weights)
                self.model.to('cuda')
                self.model.eval()
                # print("batch worker model updated !!!")

            x = time.time()
            try:
                batch = self.get_batch(train_steps=training_step)
                self.batch_buffer.push([batch])
                # print(f'batchQ={self.batch_buffer.get_len()}')
            except:
                pass
            # print(f'PREPARE={time.time()-x:.3f}')

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
            raw_policies_batch,
            mask_batch,
            ready_mask_batch,
            closable_mask_batch,
            # next_ready_mask_batch,
            # next_closable_mask_batch,
            action_high_batch,
            action_low_batch
        ) = ([], [], [], [], [], [], [], [], [], [], [], [],
             [], [], [], [], [], [], [], [], []
             )

        weight_batch = [] if self.config.PER else None

        # x = time.time()
        # n_total_samples, game_samples = ray.get([
        #     self.replay_buffer.get_n_total_samples.remote(),
        #     self.replay_buffer.sample_n_games.remote(self.config.batch_size)]
        # )
        game_samples = ray.get(self.replay_buffer.sample_n_games.remote(self.config.batch_size, force_uniform=False))
        n_total_samples = len(game_samples)
        # state_min, state_max = ray.get(self.replay_buffer.get_state_min_max.remote())
        # print(f'get time={time.time()-x:.3f}')

        """ 
            Reanalyzed
        """
        # import ipdb
        # ipdb.set_trace()
        all_bootstrap_values = []
        x = time.time()
        for i in range(self.config.num_unroll_steps_reanalyze + 1):
            begin_observation = []
            begin_origin_state = []
            begin_ready_mask = []
            begin_closable_mask = []
            begin_action_high = []
            begin_action_low = []
            bootstrap_observation = []
            bootstrap_origin_state = []
            bootstrap_mask = []
            bootstrap_ready_mask = []
            bootstrap_closable_mask = []
            bootstrap_action_high = []
            bootstrap_action_low = []


            for (game_id, game_history, game_prob, game_pos, pos_prob) in game_samples:
                game_history = game_history.subset(max([0, game_pos - self.config.stacked_observations + 1]), 16)
                # game_history = GameHistory()
                # game_history.from_list(game_history_list)
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

                if bootstrap_index > len(game_history.root_values):
                # if bootstrap_index > len(game_history.observation_history):
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

            bootstrap_mask = np.array(bootstrap_mask)
            bootstrap_observation = np.array(bootstrap_observation)
            # bootstrap_observation = (bootstrap_observation - state_min) / (state_max - state_min + 1e-4)
            bootstrap_origin_state = np.array(bootstrap_origin_state)
            bootstrap_ready_mask = np.array(bootstrap_ready_mask)
            bootstrap_closable_mask = np.array(bootstrap_closable_mask)
            bootstrap_action_high = np.array(bootstrap_action_high)
            bootstrap_action_low = np.array(bootstrap_action_low)

            x = time.time()
            bootstrap_values, _, _ = self.calculate_bootstrap(bootstrap_observation, bootstrap_mask,
                                                              bootstrap_ready_mask, bootstrap_closable_mask,
                                                              bootstrap_action_high, bootstrap_action_low, bootstrap_origin_state, train_steps=train_steps
                                                              )
            bootstrap_values = bootstrap_values.reshape(-1, 1)

            # Reanalyze result.
            all_bootstrap_values.append(bootstrap_values)

            begin_observation = np.array(begin_observation)
            # begin_observation = (begin_observation - state_min) / (state_max - state_min + 1e-4)
            begin_origin_state = np.array(begin_origin_state)
            begin_ready_mask = np.array(begin_ready_mask)
            begin_closable_mask = np.array(begin_closable_mask)
            begin_action_high = np.array(begin_action_high)
            begin_action_low = np.array(begin_action_low)
            begin_mask = np.ones(begin_observation.shape[0])


            _, begin_actions, begin_policies = self.calculate_bootstrap(begin_observation, begin_mask,
                                                                        begin_ready_mask, begin_closable_mask,
                                                                        begin_action_high, begin_action_low, begin_origin_state, train_steps=train_steps
                                                                        )

            raw_actions_batch.append(begin_actions)
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

            # game_history = GameHistory()
            # game_history.from_list(game_history_list)
            game_history = game_history.subset(max([0, game_pos - self.config.stacked_observations + 1]), 16)

            values, rewards, line_overflow_rewards, renewable_consumption_rewards, \
            running_cost_rewards, balanced_gen_rewards, reactive_power_rewards, \
            actions, ready_masks, closable_masks, action_highs, action_lows, \
            next_observations, masks = self.make_target(
                game_history, self.config.stacked_observations - 1, all_bootstrap_values[idx]
            )

            index_batch.append([game_id, game_pos])

            observation_batch.append(
                game_history.get_stacked_observations(
                    self.config.stacked_observations - 1, self.config.stacked_observations
                )
            )
            # stacked_ready_masks, stacked_closable_masks = game_history.get_stacked_masks(
            #     self.config.stacked_observations - 1,
            #     self.config.stacked_observations
            # )
            # ready_mask_batch.append(stacked_ready_masks)
            # closable_mask_batch.append(stacked_closable_masks)

            next_observation_batch.append(next_observations)
            # next_ready_mask_batch.append(next_ready_masks)
            # next_closable_mask_batch.append(next_closable_masks)
            action_batch.append(actions)
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
                # weight_batch.append(1 / n_total_samples * game_prob * pos_prob)
                weight_batch.append((n_total_samples * game_prob * pos_prob) ** (-self.config.PER_beta))

        if self.config.PER:
            weight_batch = numpy.array(weight_batch, dtype="float32") / max(
                weight_batch
            )

        observation_batch = np.array(observation_batch)
        # observation_batch = (observation_batch - state_min) / (state_max - state_min + 1e-4)
        next_observation_batch = np.array(next_observation_batch)
        # next_observation_batch = (next_observation_batch - state_min) / (state_max - state_min + 1e-4)
        mask_batch = np.array(mask_batch)
        action_batch = np.array(action_batch)
        ready_mask_batch = np.array(ready_mask_batch)
        closable_mask_batch = np.array(closable_mask_batch)
        action_high_batch = np.array(action_high_batch)
        action_low_batch = np.array(action_low_batch)
        value_batch = np.array(value_batch)
        reward_batch = np.array(reward_batch)
        line_overflow_reward_batch = np.array(line_overflow_reward_batch)
        renewable_consumption_reward_batch = np.array(renewable_consumption_reward_batch)
        running_cost_reward_batch = np.array(running_cost_reward_batch)
        balanced_gen_reward_batch = np.array(balanced_gen_reward_batch)
        reactive_power_reward_batch = np.array(reactive_power_reward_batch)
        target_mu_batch = np.array(target_mu_batch)
        target_std_batch = np.array(target_std_batch)
        weight_batch = np.array(weight_batch)
        gradient_scale_batch = np.array(gradient_scale_batch)
        raw_actions_batch = np.array(raw_actions_batch)
        raw_policies_batch = np.array(raw_policies_batch)
     #   print('TOT', time.time() - x)

        return (
            index_batch,
            self.weight_step,
            (
                observation_batch,          # [B, O_DIM]             * s0
                next_observation_batch,     # [B, STEP + 1, O_DIM]   * s0,    s1,     s2,     ..., s_unroll
                action_batch,               # [B, STEP + 1, A_DIM]   * X ,    a0,     a1,     ..., a_unroll-1
                value_batch,                # [B, STEP_R + 1]
                reward_batch,               # [B, STEP + 1]          * X,     r0,     r1,     ...,
                line_overflow_reward_batch,
                renewable_consumption_reward_batch,
                running_cost_reward_batch,
                balanced_gen_reward_batch,
                reactive_power_reward_batch,
                target_mu_batch,            # [B, STEP_R + 1, A_DIM]   mu_0,  mu_1,   mu_2,   ...,
                target_std_batch,           # [B, STEP_R + 1, A_DIM]   std_0, std_1,  std_2,  ...,
                weight_batch,               # ...
                gradient_scale_batch,       # ...
                mask_batch,                  # [B, STEP + 1]
                raw_actions_batch,
                raw_policies_batch,
                ready_mask_batch,
                closable_mask_batch,
                # next_ready_mask_batch,
                # next_closable_mask_batch
                action_high_batch,
                action_low_batch
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
        ready_masks = []
        closable_masks = []
        action_highs = []
        action_lows = []
        target_masks = []
        target_next_observations = []
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

        for current_index in range(
                state_index, state_index + self.config.num_unroll_steps + 1
        ):
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
            # if current_index < len(game_history.observation_history):
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
                ) # 20
                # stacked_ready_masks, stacked_closable_masks = game_history.get_stacked_masks(
                #     current_index,
                #     self.config.stacked_observations
                # )
                # target_next_ready_masks.append(stacked_ready_masks)
                # target_next_closable_masks.append(stacked_closable_masks)

                target_masks.append(1.)
                actions.append(game_history.action_history[current_index])
                ready_masks.append(game_history.ready_mask_history[current_index])
                closable_masks.append(game_history.closable_mask_history[current_index])
                action_highs.append(game_history.action_high_history[current_index])
                action_lows.append(game_history.action_low_history[current_index])

            elif current_index == len(game_history.root_values):
            # elif current_index == len(game_history.observation_history):
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
                # stacked_ready_masks, stacked_closable_masks = game_history.get_stacked_masks(
                #     0,
                #     self.config.stacked_observations
                # )
                # target_next_ready_masks.append(stacked_ready_masks)
                # target_next_closable_masks.append(stacked_closable_masks)
                actions.append(game_history.action_history[current_index])
                ready_masks.append(game_history.ready_mask_history[current_index])
                closable_masks.append(game_history.closable_mask_history[current_index])
                action_highs.append(game_history.action_high_history[current_index])
                action_lows.append(game_history.action_low_history[current_index])
                target_masks.append(0.)

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
                # stacked_ready_masks, stacked_closable_masks = game_history.get_stacked_masks(
                #     0,
                #     self.config.stacked_observations
                # )
                # target_next_ready_masks.append(stacked_ready_masks)
                # target_next_closable_masks.append(stacked_closable_masks)

                # import ipdb
                # ipdb.set_trace()
                if len(game_history.child_visits) > 0:
                    random_actions = self.config.sample_random_actions(len(game_history.child_visits[0]))
                    # if self.config.parameters['only_power']:
                    #     random_actions = np.concatenate((random_actions, random_actions, random_actions), axis=-1)
                    # else:
                    #     random_actions = np.concatenate((random_actions, random_actions, random_actions[:, :2]), axis=-1)
                else:
                    random_actions = self.config.sample_random_actions(17)
                    # if self.config.parameters['only_power']:
                    #     random_actions = np.concatenate((random_actions, random_actions, random_actions), axis=-1)
                    # else:
                    #     random_actions = np.concatenate((random_actions, random_actions, random_actions[:, :2]),
                    #                                     axis=-1)
                # random_actions = random_actions.clip(-0.999, 0.999)
                # import ipdb
                # ipdb.set_trace()
                actions.append(random_actions[0])
                ready_masks.append(game_history.ready_mask_history[0])
                closable_masks.append(game_history.closable_mask_history[0])
                action_highs.append(game_history.action_high_history[0])
                action_lows.append(game_history.action_low_history[0])
                target_masks.append(0.)

        target_values = np.array(target_values)
        target_rewards = np.array(target_rewards)

        target_line_of_rewards = np.array(target_line_of_rewards)
        target_renewable_consump_rewards = np.array(target_renewable_consump_rewards)
        target_run_cost_rewards = np.array(target_run_cost_rewards)
        target_bal_gen_rewards = np.array(target_bal_gen_rewards)
        target_reac_pow_rewards = np.array(target_reac_pow_rewards)

        target_masks = np.array(target_masks)
        target_next_observations = np.array(target_next_observations)
        # target_next_ready_masks = np.array(target_next_ready_masks)
        # target_next_closable_masks = np.array(target_next_closable_masks)
        actions = np.array(actions)
        ready_masks = np.array(ready_masks)
        closable_masks = np.array(closable_masks)
        action_highs = np.array(action_highs)
        action_lows = np.array(action_lows)

        return target_values, target_rewards, \
               target_line_of_rewards, target_renewable_consump_rewards, target_run_cost_rewards, \
               target_bal_gen_rewards, target_reac_pow_rewards, \
               actions, ready_masks, closable_masks, action_highs, action_lows, target_next_observations, target_masks


