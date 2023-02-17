from Environment.base_env import Environment
from utilize.settings import settings
from game.gridsim.env_wrapper import GridSimWrapper

def make_gridsim(config=None, rule_agent=None, reward_func=None, is_test=False, two_player=False, attack_all=False):
    env = Environment(settings, "EPRIReward", is_test=is_test, two_player=two_player, attack_all=attack_all)
    return GridSimWrapper(env, rule_agent=rule_agent, config=config, reward_func=reward_func)