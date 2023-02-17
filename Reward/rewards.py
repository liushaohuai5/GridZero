import math


def line_over_flow_reward(obs, settings):
    r = 1 - sum([min(i, 1) for i in obs.rho])/settings.num_line
    return r

def line_over_flow_reward_v2(obs, settings):
    # r = 1 - sum([min(i, 1) for i in obs.rho])/settings.num_line
    r = 1 - sum([rho for rho in obs.rho])/settings.num_line
    return r

def line_disconnect_reward(obs, settings):
    disc_num = settings.num_line - sum(obs.line_status)
    r = -0.3 * (1.5) ** disc_num
    return r

def renewable_consumption_reward(obs, settings):
    all_gen_p = 0.0
    all_gen_p_max = 0.0
    for i, j in enumerate(settings.renewable_ids):
        all_gen_p += obs.gen_p[j]
        all_gen_p_max += obs.curstep_renewable_gen_p_max[i]
    r = all_gen_p / all_gen_p_max
    return r

def thermal_backup_reward(obs, settings):
    backup_gen_p = 0.0
    backup_gen_p_max = sum(settings.min_gen_p)
    for i, j in enumerate(settings.thermal_ids):
        backup_gen_p += max(abs(obs.action_space['adjust_gen_p'].high[j]), abs(obs.action_space['adjust_gen_p'].low[j]))
    r = backup_gen_p / backup_gen_p_max
    return r

def balanced_gen_reward(obs, settings):
    r = 0.0
    idx = settings.balanced_id
    max_val = settings.max_gen_p[idx]
    min_val = settings.min_gen_p[idx]
    gen_p_val = obs.gen_p[idx]
    if gen_p_val > max_val:
        r += abs((gen_p_val - max_val) /
                 ((max_val - min_val)/2)
                 # max_val
                 )
    if gen_p_val < min_val:
        r += abs((gen_p_val - min_val) /
                 ((max_val - min_val)/2)
                 # min_val
                 )
    r = -10 * r   # Ensure the range of r is [-1,0]
    return r

def balanced_gen_reward_v2(obs, settings):
    r = 0.0
    i = settings.balanced_id
    r += (inv_funnel_func(obs.gen_p[i], settings.max_gen_p[i], settings.min_gen_p[i]) - 0.75)
    return r

def running_cost_reward(obs, last_obs, settings):
    r = 0.0
    for i, name in enumerate(settings.gen_name_list):
        idx = obs.unnameindex[name]
        if idx not in settings.renewable_ids:
            r -= settings.second_order_cost[i] * (obs.gen_p[idx]) ** 2 + \
                settings.first_order_cost[i] * \
                obs.gen_p[idx] + settings.constant_cost[i]
        if obs.gen_status[idx] != last_obs.gen_status[idx] and idx in settings.thermal_ids:
            r -= settings.startup_cost[i]
    # print(r)
    temp = 10000
    # print((r/temp+3)/5)
    # r = math.exp((r/temp+3)/5) - 1
    r = r / (10 * temp)
    # r = math.exp(r) - 1
    return r


def running_cost_reward_v2(obs, last_obs, settings):
    r = 0.0
    for i, name in enumerate(settings.gen_name_list):
        idx = obs.unnameindex[name]
        if idx not in settings.renewable_ids:
            r -= settings.second_order_cost[i] * (obs.gen_p[idx]) ** 2 + \
                settings.first_order_cost[i] * \
                obs.gen_p[idx] + settings.constant_cost[i]
        if obs.gen_status[idx] != last_obs.gen_status[idx] and idx in settings.thermal_ids:
            r -= settings.startup_cost[i]
    temp = 10000
    r = r / (10 * temp)
    # r = math.exp(r) - 1
    # r = math.log(1 + r)
    # r = -r**2
    return r


def gen_reactive_power_reward(obs, settings):
    r = 0.0
    for i in range(settings.num_gen):
        if obs.gen_q[i] > settings.max_gen_q[i]:
            r -= abs((obs.gen_q[i] - settings.max_gen_q[i]) /
                     ((settings.max_gen_q[i] - settings.min_gen_q[i])/2)
                     # settings.max_gen_q[i]
                     )
        if obs.gen_q[i] < settings.min_gen_q[i]:
            r -= abs((obs.gen_q[i] - settings.min_gen_q[i]) /
                     ((settings.max_gen_q[i] - settings.min_gen_q[i])/2)
                     # settings.min_gen_q[i]
                     )
    r = math.exp(r) - 1
    return r

# TODO: V2 is a dense version of reactive power reward(penalty)
def gen_reactive_power_reward_v2(obs, settings):
    # r = 0.0
    # for i in range(settings.num_gen):
    #     r += (inv_funnel_func(obs.gen_q[i], settings.max_gen_q[i], settings.min_gen_q[i]) - 0.75)
    # r /= settings.num_gen
    # return r

    r = []
    for i in range(settings.num_gen):
        r.append(inv_funnel_func(obs.gen_q[i], settings.max_gen_q[i], settings.min_gen_q[i]) - 0.75)
    return min(r)


def inv_funnel_func(x, upperB, lowerB):
    mu = (upperB + lowerB) / 2
    sigma = (upperB - lowerB) / 3
    return math.exp((-(x-mu)**2)/(sigma**2))


def sub_voltage_reward(obs, settings):
    r = 0.0
    for i in range(len(settings.max_bus_v)):
        if obs.bus_v[i] > settings.max_bus_v[i]:
            r -= abs((obs.bus_v[i] - settings.max_bus_v[i]) /
                     ((settings.max_bus_v[i] - settings.min_bus_v[i])*10)
                     # settings.max_bus_v[i]
                     )
        if obs.bus_v[i] < settings.min_bus_v[i]:
            r -= abs((obs.bus_v[i] - settings.min_bus_v[i]) /
                     ((settings.max_bus_v[i] - settings.min_bus_v[i])*10)
                     # settings.min_bus_v[i]
                     )
    r = math.exp(r) - 1
    return r


def EPRIReward(obs, last_obs, settings):
    r = settings.coeff_line_over_flow * line_over_flow_reward(obs, settings) + \
        settings.coeff_renewable_consumption * renewable_consumption_reward(obs, settings) + \
        settings.coeff_running_cost * running_cost_reward(obs, last_obs, settings) + \
        settings.coeff_balanced_gen * balanced_gen_reward(obs, settings) + \
        settings.coeff_gen_reactive_power * gen_reactive_power_reward(obs, settings) + \
        settings.coeff_sub_voltage * sub_voltage_reward(obs, settings)
    return r


def self_reward(obs, last_obs, config, settings):
    # r = config.coeff_renewable_consumption * renewable_consumption_reward(obs, settings)

    # config.coeff_line_disconnect * line_disconnect_reward(obs, settings) #+ \
    #
    r = config.coeff_line_over_flow * line_over_flow_reward_v2(obs, settings) + \
        config.coeff_renewable_consumption * renewable_consumption_reward(obs, settings) + \
        config.coeff_running_cost * running_cost_reward_v2(obs, last_obs, settings) + \
        config.coeff_balanced_gen * balanced_gen_reward(obs, settings) + \
        config.coeff_gen_reactive_power * gen_reactive_power_reward(obs, settings) + \
        config.coeff_line_disconnect * line_disconnect_reward(obs, settings)
        # config.coeff_balanced_gen * balanced_gen_reward(obs, settings) + \
        # config.coeff_gen_reactive_power * gen_reactive_power_reward(obs, settings)
        # config.coeff_thermal_backup * thermal_backup_reward(obs, settings)
        # config.coeff_sub_voltage * sub_voltage_reward(obs, settings)
    #
    return r
