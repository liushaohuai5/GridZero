from game.gridsim import make_gridsim

env = make_gridsim()
obs = env.reset()
print(obs)

import ipdb
ipdb.set_trace()