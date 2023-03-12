import sys
# sys.path.append('/workspace/AdversarialGridZero')
# print(sys.path)
import os

import numpy as np
from utilize.settings import settings
from Reward.rewards import *
from game.gridsim.utils import *
from experiments.gridsim_v2 import GridSimExperientConfig
# from Environment.base_env import Environment
import copy
import time

import pandas as pd

import gurobipy as gp
from gurobipy import GRB

from ori_opf_SG126 import traditional_solver
import matplotlib
# matplotlib.use('TkAgg')
# print(matplotlib.get_backend())
import matplotlib.pyplot as plt

up_or_down = '+'      # 0 for down, 1 for up
percentage = 0.1
force_rerun = False
start_sample_idx = [
    # 22753,
    # 16129,
    # 74593,
    # 45793,
    # 32257,
                    53569,
    # 13826, 26785,
    # 1729, 17281,
    # 34273,
    # 36289,
    # 44353, 52417,
    # 67105,
    # 75169,
    # 289, 4897,
    # 15841,
    # 31969
]
scores = []
for epi_idx, start_idx in enumerate(start_sample_idx):
    # start_idx = start_sample_idx[0]
    config = GridSimExperientConfig(env_id='grid', task_name='balance')
    config.initialize()
    game = config.new_game()
    obs = game.reset(ori_obs=True, start_sample_idx=start_idx)

    all_load_p = pd.read_csv('./test_data/load_p.csv').values
    all_renewable_max = pd.read_csv('./test_data/max_renewable_gen_p.csv').values
    thermal_units = settings.thermal_ids
    renewable_units = settings.renewable_ids
    balanced_units = [settings.balanced_id]
    nb_periods = 288
    max_gen = np.expand_dims(np.array(settings.max_gen_p), axis=1).repeat(nb_periods, axis=1)
    max_gen[settings.renewable_ids, :] = all_renewable_max[start_idx:start_idx+nb_periods, :].swapaxes(0, 1)
    tmp = np.array([i*percentage/80 for i in range(nb_periods)]).clip(0, percentage)
    # errors = max_gen[settings.renewable_ids] * np.array([np.random.normal(i*0.5/nb_periods, 0.1, len(settings.renewable_ids))[0] for i in range(nb_periods)])
    # errors = max_gen[settings.renewable_ids] * np.expand_dims(tmp, axis=0).repeat(len(settings.renewable_ids), axis=0)
    errors = np.expand_dims(np.array(settings.max_gen_p), axis=1).repeat(nb_periods, axis=1)[settings.renewable_ids] * np.expand_dims(tmp, axis=0).repeat(len(settings.renewable_ids), axis=0)
    # TODO: check errors
    ori_max_gen = copy.deepcopy(max_gen)
    if up_or_down == '+':
        max_gen[settings.renewable_ids] += errors
    else:
        max_gen[settings.renewable_ids] -= errors
    max_gen = max_gen.clip(np.zeros_like(max_gen), np.expand_dims(np.array(settings.max_gen_p), axis=1).repeat(nb_periods, axis=1))

    min_gen = np.expand_dims(np.array(settings.min_gen_p), axis=1).repeat(nb_periods, axis=1)
    constant_cost = np.array(settings.constant_cost)
    constant_cost[settings.renewable_ids] = 0
    first_order_cost = np.array(settings.first_order_cost)
    first_order_cost[settings.renewable_ids] = 0
    second_order_cost = np.array(settings.second_order_cost)
    second_order_cost[settings.renewable_ids] = 0
    initial_p = np.array(obs.gen_p)
    adjust = (sum(initial_p) - sum(obs.load_p)) / (sum(initial_p) - sum(np.array(settings.min_gen_p)*obs.gen_status)) * (initial_p - np.array(settings.min_gen_p)*obs.gen_status)
    initial_p_1 = initial_p - adjust


    if os.path.exists(f'./expert_uc_{start_idx}_{up_or_down}{percentage}.npy') and not force_rerun:
        expert_uc = np.load(f'./expert_uc_{start_idx}_{up_or_down}{percentage}.npy')
    else:
        df_up = {
            "initial": initial_p_1,
            "min_gen": min_gen,
            "max_gen": max_gen,
            "min_uptime": [40 for _ in range(settings.num_gen)],
            # "min_uptime": settings.max_steps_to_recover_gen,
            "min_downtime": [40 for _ in range(settings.num_gen)],
            # "min_downtime": settings.max_steps_to_close_gen,
            "ramp_up": np.array(settings.max_gen_p) * settings.ramp_rate,
            "ramp_down": -np.array(settings.max_gen_p) * settings.ramp_rate,
            "start_cost": settings.startup_cost,
            "fixed_cost": constant_cost,
            "first_order_cost": first_order_cost,
            "second_order_cost": second_order_cost
        }

        raw_demand = all_load_p[start_idx:start_idx+nb_periods, :]
        print("nb periods = {}".format(nb_periods))

        model = gp.Model('Unit Commitment')
        model.Params.Threads = 75
        model.Params.MIPGap = 5e-3
        in_use = model.addVars(len(thermal_units), nb_periods, vtype=GRB.BINARY, name='in_use')
        turn_on = model.addVars(len(thermal_units), nb_periods, vtype=GRB.BINARY, name='turn_on')
        turn_off = model.addVars(len(thermal_units), nb_periods, vtype=GRB.BINARY, name='turn_off')
        # turn_off = model.addVars(len(thermal_units), nb_periods, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='turn_off')
        production = model.addVars(settings.num_gen, nb_periods, vtype=GRB.CONTINUOUS, name='p')
        bus_ang = model.addVars(len(settings.busname), nb_periods, vtype=GRB.CONTINUOUS, lb=-math.pi, ub=math.pi, name='theta')
        # line_p = model.addVars(len(settings.lnname), nb_periods, vtype=GRB.CONTINUOUS, lb=float('inf'), ub=float('inf'), name='line_p')



        # When in use, the production level is constrained to be between min and max generation.
        for i, u in enumerate(thermal_units):
            for p in range(nb_periods):
                model.addConstr(production[u, p] <= df_up['max_gen'][u, p] * in_use[i, p])
                model.addConstr(production[u, p] >= df_up['min_gen'][u, p] * in_use[i, p])
                # model.addConstr(production[u, p] >= 0)
                # model.addGenConstrIndicator(in_use[i, p], True, production[u, p] <= df_up['max_gen'][u, p])
                # model.addGenConstrIndicator(in_use[i, p], True, production[u, p] >= df_up['min_gen'][u, p])
                # model.addGenConstrIndicator(in_use[i, p], False, production[u, p] == 0)

        for i, u in enumerate(renewable_units):
            for p in range(nb_periods):
                model.addConstr(production[u, p] <= df_up['max_gen'][u, p])
                model.addConstr(production[u, p] >= df_up['min_gen'][u, p])

        for i, u in enumerate(balanced_units):
            for p in range(nb_periods):
                model.addConstr(production[u, p] <= df_up['max_gen'][u, p])
                model.addConstr(production[u, p] >= df_up['min_gen'][u, p])


        # Initial state
        # If initial production is nonzero, then period #1 is not a turn_on
        # else turn_on equals in_use
        # Dual logic is implemented for turn_off
        for i, u in enumerate(thermal_units):
            if df_up['initial'][u] > 0:
                # if u is already running, not starting up
                model.addConstr(turn_on[i, 0] == 0)
                # turnoff iff not in use
                model.addConstr(turn_off[i, 0] + in_use[i, 0] == 1)
                # model.addConstr(in_use[i, 0] == 1)
                # model.addConstr(turn_off[i, 0] == 0)
            else:
                # turn on at 0 iff in use at 0
                model.addConstr(turn_on[i, 0] == in_use[i, 0])
                # model.addConstr(turn_on[i, 0] == 0)
                # already off, not switched off at t==0
                model.addConstr(turn_off[i, 0] == 0)
                # model.addConstr(in_use[i, 0] == 0)


        # Thermal Units Ramping Constraints
        for i, u in enumerate(thermal_units):
            u_ramp_up = df_up['ramp_up'][u]
            u_ramp_down = df_up['ramp_down'][u]
            u_initial = df_up['initial'][u]
            # model.addConstr(production[u, 0] - u_initial <= u_ramp_up)
            model.addGenConstrIndicator(turn_on[i, 0], False, production[u, 0] - u_initial <= u_ramp_up)
            # model.addConstr(production[u, 0] - u_initial >= u_ramp_down)
            model.addGenConstrIndicator(turn_off[i, 0], False, production[u, 0] - u_initial >= u_ramp_down)
            for p in range(nb_periods - 1):
                # model.addConstr(production[u, p+1] - production[u, p] <= u_ramp_up)
                model.addGenConstrIndicator(turn_on[i, p+1], False, production[u, p+1] - production[u, p] <= u_ramp_up)
                # model.addConstr(production[u, p+1] - production[u, p] >= u_ramp_down)
                model.addGenConstrIndicator(turn_off[i, p+1], False, production[u, p+1] - production[u, p] >= u_ramp_down)


        # Thermal Units Open/Close Limits
        for i, u in enumerate(thermal_units):
            # for (in_use_curr, in_use_next, turn_on_next, turn_off_next) in zip(in_use[i, :-1], in_use[i, 1:], turn_on[i, 1:], turn_off[i, 1:]):
            for p in range(nb_periods - 1):
                # if unit is off at time t and on at time t+1, then it was turned on at time t+1
                # model.addConstr(in_use_next - in_use_curr <= turn_on_next)
                # model.addConstr(in_use[i, p+1] - in_use[i, p] <= turn_on[i, p+1])
                model.addConstr(in_use[i, p+1] - in_use[i, p] + turn_off[i, p+1] == turn_on[i, p+1])

                # if unit is on at time t and time t+1, then it was not turned on at time t+1
                # mdl.add_constraint(in_use_next + in_use_curr + turn_on_next <= 2)
                # model.addConstr(in_use[i, p+1] + in_use[i, p] + turn_on[i, p+1] <= 2)

                # if unit is on at time t and off at time t+1, then it was turned off at time t+1
                # model.addConstr(in_use_curr - in_use_next + turn_on_next == turn_off_next)
                model.addConstr(in_use[i, p] - in_use[i, p+1] + turn_on[i, p+1] == turn_off[i, p+1])

                # forbid turn_on and turn_off at the same time
                model.addConstr(turn_on[i, p+1] + turn_off[i, p+1] <= 1)


        # Iff reaches minimum power, a thermal gen could be turn off
        for p in range(nb_periods-1):
            for i, u in enumerate(thermal_units):
                model.addGenConstrIndicator(turn_off[i, p+1], True, production[u, p] == min_gen[u, p])


        # Thermal Gen is not allowed to turn on before step 40
        for i, u in enumerate(thermal_units):
            min_uptime = df_up['min_uptime'][u]
            for p in range(min_uptime):
                model.addConstr(turn_on[i, p] == 0)

        for p in range(nb_periods):
            model.addConstr(gp.quicksum([turn_on[i, p] for i, u in enumerate(thermal_units)]) <= 1)
            model.addConstr(gp.quicksum([turn_off[i, p] for i, u in enumerate(thermal_units)]) <= 1)

        # for p in range(nb_periods):
        #     model.addConstr(gp.quicksum([in_use[i, p] for i, u in enumerate(thermal_units)]) >= 20)

        # Thermal Units Open/Close Waiting Limits
        for i, u in enumerate(thermal_units):
            min_uptime = df_up['min_uptime'][u]
            min_downtime = df_up['min_downtime'][u]
            # Note that r.turn_on and r.in_use are Series that can be indexed as arrays (ie: first item index = 0)
            for t in range(min_uptime, nb_periods):
                model.addConstr(gp.quicksum([turn_on[i, j] for j in range((t - min_uptime)+1, t+1)]) <= in_use[i, t])

            for t in range(min_downtime, nb_periods):
                model.addConstr(gp.quicksum([turn_off[i, j] for j in range((t - min_downtime)+1, t+1)]) <= 1 - in_use[i, t])


        # Period Power Balancing Limits
        # for period in range(nb_periods):
        #     total_demand = sum(raw_demand[period, :])
        #     # model.addConstr(gp.quicksum([production[i, period] for i in range(settings.num_gen)]) >= total_demand)
        #     # model.addConstr(gp.quicksum([production[i, period] for i in range(settings.num_gen)]) <= 1.05 * total_demand)
        #     model.addConstr(gp.quicksum([production[i, period] for i in range(settings.num_gen)]) == total_demand)


        # TODO: Network Equations Constraints
        gen2busM = gen_to_bus_matrix(obs, settings)
        load2busM = load_to_bus_matrix(obs, settings)
        bus2lineM = bus_to_line_matrix(obs, settings)
        data = pd.read_csv(r'./model_jm/SG126.csv'
                             ).values
        branch_data = data[135:135+194, 1:]
        # b_line = branch_data[:, 71].astype(np.float32)
        b_line = -1/branch_data[:, 8].astype(np.float32)
        bus_num = len(settings.busname)
        B = np.zeros((bus_num, bus_num))
        for j in range(bus2lineM.shape[1]):
            s = np.where(bus2lineM[:, j] == 1)[0][0]
            t = np.where(bus2lineM[:, j] == -1)[0][0]
            B[s, t] = b_line[j]
            B[t, s] = b_line[j]
        tmp = -np.matmul(np.abs(bus2lineM), np.expand_dims(b_line, axis=1)).squeeze()
        for i in range(len(settings.busname)):
            B[i, i] = tmp[i]


        for p in range(nb_periods):
            total_load = np.matmul(load2busM.swapaxes(0, 1), raw_demand[p, :])
            tmp = []
            for bus, name in enumerate(settings.busname):
                total_gen_p = gp.quicksum([production[i, p] * gen2busM[i, bus] for i in range(settings.num_gen)])
                total_injection = gp.quicksum([bus_ang[i, p] * B[bus, i] for i, n in enumerate(settings.busname)])
                tmp.append(total_injection)
                model.addConstr(total_injection + total_gen_p / 100 - total_load[bus] / 100 == 0)
                # model.addConstr(total_injection + total_gen_p / 100 - total_load[bus] / 100 <= 0.001)
                # model.addConstr(total_injection + total_gen_p / 100 - total_load[bus] / 100 >= -0.001)
            model.addConstr(gp.quicksum(tmp) == 0)

        for p in range(nb_periods):
            for l, n in enumerate(settings.lnname):
                s = np.where(bus2lineM[:, l] == 1)[0][0]
                t = np.where(bus2lineM[:, l] == -1)[0][0]
                model.addConstr(B[s, t] * (bus_ang[s, p] - bus_ang[t, p]) <= settings.line_thermal_limit[l])
                model.addConstr(B[s, t] * (bus_ang[s, p] - bus_ang[t, p]) >= -settings.line_thermal_limit[l])


        # objective
        total_fixed_cost = gp.quicksum(gp.quicksum([in_use[i, p] * df_up['fixed_cost'][u] for i, u in enumerate(thermal_units)]) for p in range(nb_periods))
        # total_first_order_cost = gp.quicksum(gp.quicksum([production[u, p] * df_up['first_order_cost'][u] for u in thermal_units]) for p in range(nb_periods))
        total_first_order_cost = gp.quicksum(gp.quicksum([production[u, p] * df_up['first_order_cost'][u] for u in range(settings.num_gen)]) for p in range(nb_periods))
        # total_second_order_cost = gp.quicksum(gp.quicksum([production[u, p] * production[u, p] * df_up['second_order_cost'][u] for u in thermal_units]) for p in range(nb_periods))
        total_second_order_cost = gp.quicksum(gp.quicksum([production[u, p] * production[u, p] * df_up['second_order_cost'][u] for u in range(settings.num_gen)]) for p in range(nb_periods))
        total_startup_cost = gp.quicksum(gp.quicksum([turn_on[i, p] * df_up['start_cost'][u] for i, u in enumerate(thermal_units)]) for p in range(nb_periods))
        # total_startup_cost = 0
        total_close_cost = gp.quicksum(gp.quicksum([turn_off[i, p] * df_up['start_cost'][u] for i, u in enumerate(thermal_units)]) for p in range(nb_periods))
        # total_close_cost = 0
        total_economic_cost = total_fixed_cost + total_first_order_cost + total_second_order_cost + total_startup_cost + total_close_cost

        model.setObjective(total_economic_cost)

        # minimize sum of all costs
        x = time.time()
        model.write('junk.lp')
        model.optimize()
        print(f'time consuming={time.time()-x:.3f}s')

        gen_in_use = np.zeros((len(thermal_units), nb_periods))
        gen_turn_on = np.zeros((len(thermal_units), nb_periods))
        gen_turn_off = np.zeros((len(thermal_units), nb_periods))
        for i, u in enumerate(thermal_units):
            for p in range(nb_periods):
                gen_in_use[i, p] = int(in_use[i, p].x)
                gen_turn_on[i, p] = int(turn_on[i, p].x)
                gen_turn_off[i, p] = int(turn_off[i, p].x)

        gen_production = np.zeros((settings.num_gen, nb_periods))
        for i in range(settings.num_gen):
            for p in range(nb_periods):
                gen_production[i, p] = production[i, p].x
        bus_angle = np.zeros((len(settings.busname), nb_periods))
        for i in range(len(settings.busname)):
            for p in range(nb_periods):
                bus_angle[i, p] = bus_ang[i, p].x
        expert_uc = np.array([gen_in_use, gen_turn_on, gen_turn_off])
        np.save(f'./expert_uc_{start_idx}_{up_or_down}{percentage}', expert_uc)

        for period in range(nb_periods):
            total_load = np.matmul(load2busM.swapaxes(0, 1), raw_demand[period, :])
            tmp = []
            for bus, name in enumerate(settings.busname):
                total_gen_p = sum([gen_production[i, period] * gen2busM[i, bus] for i in range(settings.num_gen)])
                total_injection = sum([bus_ang[i, period].x * B[bus, i] for i, n in enumerate(settings.busname)])
                tmp.append(total_injection)
                if abs(total_injection + total_gen_p / 100 - total_load[bus] / 100) > 0.01:
                    import ipdb
                    ipdb.set_trace()
                    print(f'error, T={period}, bus={bus}')

    print('finish')
    T_solver = traditional_solver(config, obs, dc_opf_flag=True, unit_comb_flag=False, for_training_flag=False)
    step, score, epi_vol_voilations, epi_reac_violations, epi_bal_p_violations, epi_soft_overflows, epi_hard_overflows, \
        epi_running_cost, epi_renewable_consumption = T_solver.play_game(start_idx=start_idx, epi_idx=epi_idx,
                                                                         expert_uc=expert_uc,
                                                                         opt_name=f'{up_or_down}{percentage}',
                                                                         ori_renewable_max=max_gen)
    import ipdb
    ipdb.set_trace()
    scores.append(score)

import ipdb
ipdb.set_trace()
data = pd.read_csv(f'./results/gridsim_v2/trad_expert/new_data_{start_idx}.csv').values
steps = np.asarray([i for i in range(data.shape[0])])
colors = ['sienna', 'darkgrey', 'lightpink', 'red', 'royalblue']
plt.plot(steps, data[:, 0], color='darkorange', label='Load')
plt.plot(steps, data[:, 1], color=colors[4], linestyle='dashdot', label='GridZero consumption')
plt.plot(steps, data[:, 2], color='mediumseagreen', linestyle='dotted', label='Renewable maximum')
plt.show()



# small system
# Bus 23 71 69
selected_bus = [23, 69, 70, 71, 72]
selected_load = [53]
selected_gen = [9, 30, 31, 32]
selected_line = [108, 109, 110, 111, 112]

if os.path.exists(f'./expert_uc_{start_idx}.npy') and not force_rerun:
    expert_uc = np.load(f'./expert_uc_{start_idx}.npy')
else:
    # import ipdb
    # ipdb.set_trace()
    all_load_p = pd.read_csv('./test_data/load_p.csv').values
    all_renewable_max = pd.read_csv('./test_data/max_renewable_gen_p.csv').values

    thermal_units = [1, 2]
    renewable_units = [0]
    balanced_units = [3]
    nb_periods = 288

    max_gen = np.expand_dims(np.array(settings.max_gen_p)[selected_gen], axis=1).repeat(nb_periods, axis=1)
    max_gen[renewable_units, :] = all_renewable_max[start_idx:start_idx+nb_periods, 8:9].swapaxes(0, 1)
    # errors = max_gen[settings.renewable_ids] * np.array([np.random.normal(i*0.5/nb_periods, 0.1, len(settings.renewable_ids))[0] for i in range(nb_periods)])
    errors = max_gen[renewable_units] * np.expand_dims(np.array([i*0.5/nb_periods for i in range(nb_periods)]), axis=0).repeat(len(renewable_units), axis=0)
    ori_max_gen = copy.deepcopy(max_gen)
    # max_gen[settings.renewable_ids] -= errors
    min_gen = np.expand_dims(np.array(settings.min_gen_p)[selected_gen], axis=1).repeat(nb_periods, axis=1)
    constant_cost = np.array(settings.constant_cost)[selected_gen]
    constant_cost[renewable_units] = 0
    first_order_cost = np.array(settings.first_order_cost)[selected_gen]
    first_order_cost[renewable_units] = 0
    second_order_cost = np.array(settings.second_order_cost)[selected_gen]
    second_order_cost[renewable_units] = 0
    initial_p = np.array(obs.gen_p)[selected_gen]
    raw_demand = all_load_p[start_idx:start_idx + nb_periods, 53:56].sum(1, keepdims=True)
    print("nb periods = {}".format(nb_periods))
    tmp = (raw_demand[0] - initial_p[0]) / 3
    initial_p[1:4] = tmp
    df_up = {
        "initial": initial_p,
        "min_gen": min_gen,
        "max_gen": max_gen,
        "min_uptime": [40 for _ in range(len(selected_gen))],
        "min_downtime": [40 for _ in range(len(selected_gen))],
        "ramp_up": np.array(settings.max_gen_p)[selected_gen] * settings.ramp_rate,
        "ramp_down": -np.array(settings.max_gen_p)[selected_gen] * settings.ramp_rate,
        "start_cost": settings.startup_cost,
        "fixed_cost": constant_cost,
        "first_order_cost": first_order_cost,
        "second_order_cost": second_order_cost
    }

    model = gp.Model('Unit Commitment')
    in_use = model.addVars(len(thermal_units), nb_periods, vtype=GRB.BINARY, name='in_use')
    turn_on = model.addVars(len(thermal_units), nb_periods, vtype=GRB.BINARY, name='turn_on')
    turn_off = model.addVars(len(thermal_units), nb_periods, vtype=GRB.BINARY, name='turn_off')
    # turn_off = model.addVars(len(thermal_units), nb_periods, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name='turn_off')
    production = model.addVars(len(selected_gen), nb_periods, vtype=GRB.CONTINUOUS, name='p')
    bus_ang = model.addVars(len(selected_bus), nb_periods, vtype=GRB.CONTINUOUS, lb=-math.pi, ub=math.pi, name='theta')
    # line_p = model.addVars(len(settings.lnname), nb_periods, vtype=GRB.CONTINUOUS, lb=float('inf'), ub=float('inf'), name='line_p')



    # When in use, the production level is constrained to be between min and max generation.
    for i, u in enumerate(thermal_units):
        for p in range(nb_periods):
            model.addConstr(production[u, p] <= df_up['max_gen'][u, p] * in_use[i, p])
            model.addConstr(production[u, p] >= df_up['min_gen'][u, p] * in_use[i, p])
            # model.addConstr(production[u, p] >= 0)
            # model.addGenConstrIndicator(in_use[i, p], True, production[u, p] <= df_up['max_gen'][u, p])
            # model.addGenConstrIndicator(in_use[i, p], True, production[u, p] >= df_up['min_gen'][u, p])
            # model.addGenConstrIndicator(in_use[i, p], False, production[u, p] == 0)

    for i, u in enumerate(renewable_units):
        for p in range(nb_periods):
            model.addConstr(production[u, p] <= df_up['max_gen'][u, p])
            model.addConstr(production[u, p] >= df_up['min_gen'][u, p])

    for i, u in enumerate(balanced_units):
        for p in range(nb_periods):
            model.addConstr(production[u, p] <= df_up['max_gen'][u, p])
            model.addConstr(production[u, p] >= df_up['min_gen'][u, p])


    # Initial state
    # If initial production is nonzero, then period #1 is not a turn_on
    # else turn_on equals in_use
    # Dual logic is implemented for turn_off
    for i, u in enumerate(thermal_units):
        if df_up['initial'][u] > 0:
            # if u is already running, not starting up
            model.addConstr(turn_on[i, 0] == 0)
            # turnoff iff not in use
            model.addConstr(turn_off[i, 0] + in_use[i, 0] == 1)
            # model.addConstr(in_use[i, 0] == 1)
            # model.addConstr(turn_off[i, 0] == 0)
        else:
            # turn on at 0 iff in use at 0
            model.addConstr(turn_on[i, 0] == in_use[i, 0])
            # model.addConstr(turn_on[i, 0] == 0)
            # already off, not switched off at t==0
            model.addConstr(turn_off[i, 0] == 0)
            # model.addConstr(in_use[i, 0] == 0)


    # Thermal Units Ramping Constraints
    for i, u in enumerate(thermal_units):
        u_ramp_up = df_up['ramp_up'][u]
        u_ramp_down = df_up['ramp_down'][u]
        u_initial = df_up['initial'][u]
        # model.addConstr(production[u, 0] - u_initial <= u_ramp_up)
        model.addGenConstrIndicator(turn_on[i, 0], False, production[u, 0] - u_initial <= u_ramp_up)
        # model.addConstr(production[u, 0] - u_initial >= u_ramp_down)
        model.addGenConstrIndicator(turn_off[i, 0], False, production[u, 0] - u_initial >= u_ramp_down)
        for p in range(nb_periods - 1):
            # model.addConstr(production[u, p+1] - production[u, p] <= u_ramp_up)
            model.addGenConstrIndicator(turn_on[i, p+1], False, production[u, p+1] - production[u, p] <= u_ramp_up)
            # model.addConstr(production[u, p+1] - production[u, p] >= u_ramp_down)
            model.addGenConstrIndicator(turn_off[i, p+1], False, production[u, p+1] - production[u, p] >= u_ramp_down)


    # Thermal Units Open/Close Limits
    for i, u in enumerate(thermal_units):
        # for (in_use_curr, in_use_next, turn_on_next, turn_off_next) in zip(in_use[i, :-1], in_use[i, 1:], turn_on[i, 1:], turn_off[i, 1:]):
        for p in range(nb_periods - 1):
            # if unit is off at time t and on at time t+1, then it was turned on at time t+1
            # model.addConstr(in_use_next - in_use_curr <= turn_on_next)
            # model.addConstr(in_use[i, p+1] - in_use[i, p] <= turn_on[i, p+1])
            model.addConstr(in_use[i, p+1] - in_use[i, p] + turn_off[i, p+1] == turn_on[i, p+1])

            # if unit is on at time t and time t+1, then it was not turned on at time t+1
            # mdl.add_constraint(in_use_next + in_use_curr + turn_on_next <= 2)
            # model.addConstr(in_use[i, p+1] + in_use[i, p] + turn_on[i, p+1] <= 2)

            # if unit is on at time t and off at time t+1, then it was turned off at time t+1
            # model.addConstr(in_use_curr - in_use_next + turn_on_next == turn_off_next)
            model.addConstr(in_use[i, p] - in_use[i, p+1] + turn_on[i, p+1] == turn_off[i, p+1])

            # forbid turn_on and turn_off at the same time
            model.addConstr(turn_on[i, p+1] + turn_off[i, p+1] <= 1)


    # Iff reaches minimum power, a thermal gen could be turn off
    for p in range(nb_periods-1):
        for i, u in enumerate(thermal_units):
            model.addGenConstrIndicator(turn_off[i, p+1], True, production[u, p] == min_gen[u, p])


    # Thermal Gen is not allowed to turn on before step 40
    for i, u in enumerate(thermal_units):
        min_up_time = df_up['min_uptime'][u]
        for p in range(min_up_time):
            model.addConstr(turn_on[i, p] == 0)

    for p in range(nb_periods):
        model.addConstr(gp.quicksum([turn_on[i, p] for i, u in enumerate(thermal_units)]) <= 2)
        model.addConstr(gp.quicksum([turn_off[i, p] for i, u in enumerate(thermal_units)]) <= 2)

    # Thermal Units Open/Close Waiting Limits
    for i, u in enumerate(thermal_units):
        min_uptime = df_up['min_uptime'][u]
        min_downtime = df_up['min_downtime'][u]
        # Note that r.turn_on and r.in_use are Series that can be indexed as arrays (ie: first item index = 0)
        for t in range(min_uptime, nb_periods):
            model.addConstr(gp.quicksum([turn_on[i, j] for j in range((t - min_uptime)+1, t+1)]) <= in_use[i, t])

        for t in range(min_downtime, nb_periods):
            model.addConstr(gp.quicksum([turn_off[i, j] for j in range((t - min_downtime)+1, t+1)]) <= 1 - in_use[i, t])


    # Period Power Balancing Limits
    # for period in range(nb_periods):
    #     total_demand = sum(raw_demand[period, :])
    #     # model.addConstr(gp.quicksum([production[i, period] for i in range(settings.num_gen)]) >= total_demand)
    #     # model.addConstr(gp.quicksum([production[i, period] for i in range(settings.num_gen)]) <= 1.05 * total_demand)
    #     model.addConstr(gp.quicksum([production[i, period] for i in range(settings.num_gen)]) == total_demand)


    # TODO: Network Equations Constraints
    gen2busM = gen_to_bus_matrix(obs, settings)[selected_gen][:, selected_bus]
    load2busM = load_to_bus_matrix(obs, settings)[selected_load][:, selected_bus]
    bus2lineM = bus_to_line_matrix(obs, settings)[selected_bus][:, selected_line]
    data = pd.read_csv(r'./model_jm/SG126.csv'
                         ).values
    branch_data = data[135:135+194, 1:]
    # b_line = branch_data[:, 71].astype(np.float32)
    b_line = -1/branch_data[:, 8].astype(np.float32)
    b_line = b_line[selected_line]
    bus_num = len(selected_bus)
    B = np.zeros((bus_num, bus_num))
    for j in range(bus2lineM.shape[1]):
        s = np.where(bus2lineM[:, j] == 1)[0][0]
        t = np.where(bus2lineM[:, j] == -1)[0][0]
        B[s, t] = b_line[j]
        B[t, s] = b_line[j]
    tmp = -np.matmul(np.abs(bus2lineM), np.expand_dims(b_line, axis=1)).squeeze()
    for i in range(len(selected_bus)):
        B[i, i] = tmp[i]


    for p in range(nb_periods):
        total_load = np.matmul(load2busM.swapaxes(0, 1), raw_demand[p, :])
        tmp = []
        for bus, name in enumerate(selected_bus):
            total_gen_p = gp.quicksum([production[i, p] * gen2busM[i, bus] for i, gen in enumerate(selected_gen)])
            total_injection = gp.quicksum([bus_ang[i, p] * B[bus, i] for i, n in enumerate(selected_bus)])
            tmp.append(total_injection)
            model.addConstr(total_injection + total_gen_p / 100 - total_load[bus] / 100 == 0)
            # model.addConstr(total_injection + total_gen_p / 100 - total_load[bus] / 100 <= 0.001)
            # model.addConstr(total_injection + total_gen_p / 100 - total_load[bus] / 100 >= -0.001)
        model.addConstr(gp.quicksum(tmp) == 0)

    for p in range(nb_periods):
        for l, n in enumerate(selected_line):
            s = np.where(bus2lineM[:, l] == 1)[0][0]
            t = np.where(bus2lineM[:, l] == -1)[0][0]
            model.addConstr(B[s, t] * (bus_ang[s, p] - bus_ang[t, p]) <= settings.line_thermal_limit[n])
            model.addConstr(B[s, t] * (bus_ang[s, p] - bus_ang[t, p]) >= -settings.line_thermal_limit[n])


    # objective
    total_fixed_cost = gp.quicksum(gp.quicksum([in_use[i, p] * df_up['fixed_cost'][u] for i, u in enumerate(thermal_units)]) for p in range(nb_periods))
    # total_first_order_cost = gp.quicksum(gp.quicksum([production[u, p] * df_up['first_order_cost'][u] for u in thermal_units]) for p in range(nb_periods))
    total_first_order_cost = gp.quicksum(gp.quicksum([production[i, p] * df_up['first_order_cost'][i] for i, u in enumerate(selected_gen)]) for p in range(nb_periods))
    # total_second_order_cost = gp.quicksum(gp.quicksum([production[u, p] * production[u, p] * df_up['second_order_cost'][u] for u in thermal_units]) for p in range(nb_periods))
    total_second_order_cost = gp.quicksum(gp.quicksum([production[i, p] * production[i, p] * df_up['second_order_cost'][i] for i, u in enumerate(selected_gen)]) for p in range(nb_periods))
    # total_startup_cost = gp.quicksum(gp.quicksum([turn_on[i, p] * df_up['start_cost'][u] for i, u in enumerate(thermal_units)]) for p in range(nb_periods))
    total_startup_cost = 0
    # total_close_cost = gp.quicksum(gp.quicksum([turn_off[i, p] * df_up['start_cost'][u] for i, u in enumerate(thermal_units)]) for p in range(nb_periods))
    total_close_cost = 0
    total_economic_cost = total_fixed_cost + total_first_order_cost + total_second_order_cost + total_startup_cost + total_close_cost

    model.setObjective(total_economic_cost)

    # minimize sum of all costs
    x = time.time()
    model.write('junk.lp')
    model.optimize()
    print(f'time consuming={time.time()-x:.3f}s')
    # import ipdb
    # ipdb.set_trace()

    gen_in_use = np.zeros((len(thermal_units), nb_periods))
    gen_turn_on = np.zeros((len(thermal_units), nb_periods))
    gen_turn_off = np.zeros((len(thermal_units), nb_periods))
    for i, u in enumerate(thermal_units):
        for p in range(nb_periods):
            gen_in_use[i, p] = int(in_use[i, p].x)
            gen_turn_on[i, p] = int(turn_on[i, p].x)
            gen_turn_off[i, p] = int(turn_off[i, p].x)

    gen_production = np.zeros((len(selected_gen), nb_periods))
    for i in range(len(selected_gen)):
        for p in range(nb_periods):
            gen_production[i, p] = production[i, p].x
    bus_angle = np.zeros((len(selected_bus), nb_periods))
    for i in range(len(selected_bus)):
        for p in range(nb_periods):
            bus_angle[i, p] = bus_ang[i, p].x

    expert_uc = np.array([gen_in_use, gen_turn_on, gen_turn_off])
    np.save(f'./expert_uc_{start_idx}', expert_uc)

    import ipdb
    ipdb.set_trace()
    for period in range(nb_periods):
        total_load = np.matmul(load2busM.swapaxes(0, 1), raw_demand[period, :])
        tmp = []
        for bus, name in enumerate(selected_bus):
            total_gen_p = sum([production[i, period].x * gen2busM[i, bus] for i, gen in enumerate(selected_gen)])
            total_injection = sum([bus_ang[i, period].x * B[bus, i] for i, n in enumerate(selected_bus)])
            tmp.append(total_injection)
            if abs(total_injection + total_gen_p / 100 - total_load[bus] / 100) > 0.01:
                import ipdb
                ipdb.set_trace()
                print(f'error, T={period}, bus={bus}')

import ipdb
ipdb.set_trace()
print('finish')
