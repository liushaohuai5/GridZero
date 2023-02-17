import yaml

def check_gen_type(gen_type):
    return all([ele == 1 or ele == 5 or ele == 2 for ele in gen_type])

# get dict attribute with 'obj.attr' format
class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

with open('utilize/parameters/main.yml', 'r') as f:
    dict_ = yaml.load(f, Loader=yaml.Loader)
    name_index = {}
    for key, val in zip(dict_["un_nameindex_key"], dict_["un_nameindex_value"]):
        name_index[key] = val
    dict_['name_index'] = name_index

settings = dotdict(dict_)
for i in settings.thermal_ids:
    settings.min_gen_p[i] = float(int(settings.max_gen_p[i] * 0.5))

settings.max_steps_to_recover_gen = []
settings.max_steps_to_close_gen = []
for i in range(settings.num_gen):
    if i in settings.thermal_ids:
        if settings.max_gen_p[i] < 80:
            settings.max_steps_to_recover_gen.append(10)
            settings.max_steps_to_close_gen.append(10)
        if settings.max_gen_p[i] >= 80:
            settings.max_steps_to_recover_gen.append(40)
            settings.max_steps_to_close_gen.append(40)
    else:
        settings.max_steps_to_recover_gen.append(40)
        settings.max_steps_to_close_gen.append(40)

settings.hard_overflow_bound = 1.5
settings.soft_overflow_bound = 1.1
settings.max_steps_soft_overflow = 8

del dict_

if not check_gen_type(settings.gen_type):
    raise NotImplemented

