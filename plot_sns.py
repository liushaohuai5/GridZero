import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from utilize.settings import settings
import copy
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Circle
sns.set()

colors = ['sienna', 'darkgrey', 'lightpink', 'red', 'royalblue']

# plt.rcParams['axes.facecolor'] = 'white'
plt.rcdefaults()
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13

# Fixing random state for reproducibility
np.random.seed(20220105)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ens = ['DAS-TFR', 'DAS-ECS', 'DAS-PHS', 'RTS-GridZero*']
Colors = ['lightcoral', 'brown', 'b', 'y']
stas = [
    # ' ',
    '10%/30%', '20%/60%',
    # '  '
]
weight = {
          '10%/30%': np.array([29.9, 257.2, 155.1, 5]),
          '20%/60%': np.array([60.1, 514.7, 309.7, 5]),
}
_x = np.arange(len(ens))
_y = np.arange(len(stas))
_xx, _yy = np.meshgrid(_x, _y)
width = depth = 0.5
for ith, ista in enumerate(stas):
    x = _xx[ith]
    y = _yy[ith]
    top = []
    for i in range(len(ens)):
        top.append(weight[ista][i])
    bottom = np.zeros_like(top)
    cs = [Colors[ith]] * len(x)
    ax.bar3d(x, y, bottom, width, depth, top, alpha=0.8, color=cs)

ax.set_xlabel('Type', fontsize=15, labelpad=15)
ax.set_ylabel('Retrofit level / Renewable penetration', fontsize=15, labelpad=20)
ax.set_zlabel('Cost (Million $)', fontsize=15, labelpad=15)

ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(ens)
ax.set_ylim(0, 2)
ax.set_yticks([0.5, 1.5])
ax.set_yticklabels(stas)
plt.show()


# for ith, ista in enumerate(stas):
#     xs = np.arange(len(ens))
#     ys = weight[ista]
#
#     cs = [colors[ith]] * len(xs)
#     # Plot the bar graph given by xs and ys on the plane y=k with 80% opacity.
#     ax.bar(xs, ys, zs=ith+0.5, zdir='y', color=cs, alpha=0.8)
#
# ax.set_xlabel('Type', fontsize=15, labelpad=15)
# ax.set_ylabel('Retrofit level / Renewable penetration', fontsize=15, labelpad=15)
# ax.set_zlabel('Cost (Million $)', fontsize=15, labelpad=15)
#
# ax.set_xticks([0, 1, 2, 3])
# ax.set_xticklabels(ens)
# ax.set_ylim(0, 2)
# ax.set_yticks([0.5, 1.5])
# ax.set_yticklabels(stas)
# plt.show()


percentage = 0.4
start_idx = 53569
all_renewable_max = pd.read_csv('./test_data/max_renewable_gen_p.csv').values

# all_load = pd.read_csv('./test_data/load_p.csv').values
# pure_load = all_load.sum(1) - all_renewable_max.sum(1)
# import ipdb
# ipdb.set_trace()
# step = 53450
# # plt.plot(np.arange(288), all_renewable_max.sum(1)[16980-144:16980+144])
# # plt.plot(np.arange(288), all_load.sum(1)[16980-144:16980+144])
# # plt.plot(np.arange(288), pure_load[2900:2900+288])
# plt.plot(np.arange(288), all_renewable_max.sum(1)[step:step+288])
# plt.plot(np.arange(288), all_load.sum(1)[step:step+288])
# plt.show()
# i = 0
# delta = 288
# while i < pure_load.shape[0]:
#     plt.plot(np.arange(delta), all_load.sum(1)[i:i+delta])
#     plt.plot(np.arange(delta), all_renewable_max.sum(1)[i:i+delta])
#     plt.title(f'i={i}')
#     plt.show()
#     i += delta

thermal_units = settings.thermal_ids
renewable_units = settings.renewable_ids
balanced_units = [settings.balanced_id]
nb_periods = 288
max_gen = np.expand_dims(np.array(settings.max_gen_p), axis=1).repeat(nb_periods, axis=1)
max_gen[settings.renewable_ids, :] = all_renewable_max[start_idx:start_idx+nb_periods, :].swapaxes(0, 1)
tmp = np.array([i*percentage/150 for i in range(nb_periods)]).clip(0, percentage)
# errors = max_gen[settings.renewable_ids] * np.array([np.random.normal(i*0.5/nb_periods, 0.1, len(settings.renewable_ids))[0] for i in range(nb_periods)])
# errors = max_gen[settings.renewable_ids] * np.expand_dims(tmp, axis=0).repeat(len(settings.renewable_ids), axis=0)
errors = np.expand_dims(np.array(settings.max_gen_p), axis=1).repeat(nb_periods, axis=1)[settings.renewable_ids] * np.expand_dims(tmp, axis=0).repeat(len(settings.renewable_ids), axis=0)
# TODO: check errors
ori_max_gen = copy.deepcopy(max_gen)
max_gen[settings.renewable_ids] += 0.8*errors
ori_max_gen[settings.renewable_ids] -= errors

fig, axs = plt.subplots(2, 2, sharey=True, sharex=True, constrained_layout=False)

data = pd.read_csv(f'./new_data_53569_+0.3_1.csv').values
gen_status = np.array(data[:, -1])
gen_status_over = np.array([[float(i) for i in gen_status[j][1:-1].split(' ')] for j in range(data.shape[0])])
# gen_status_over = np.concatenate((gen_status_over, np.zeros((288 - gen_status_over.shape[0], gen_status_over.shape[1]))), axis=0)
data = data[:, :-1].astype(np.float32)
steps_1 = np.array([i for i in range(data.shape[0])])
max_gen_p = data[:, -2] - 50
min_gen_p = data[:, -1]

g_data = pd.read_csv(f'./data_53569.csv').values
gen_status = np.array(g_data[:, -1])
gen_status_gridzero = np.array([[float(i) for i in gen_status[j][1:-1].split(',')] for j in range(g_data.shape[0])])
g_data = g_data[:, :-1].astype(np.float32)

data1 = pd.read_csv(f'./results/gridsim_v2/selected_episodes/2/data_53569.csv').values
steps = np.asarray([i for i in range(288)])

axs[0, 0].plot(steps, data1[:, 0], color='darkorange', label='Load consumption')
axs[0, 0].plot(steps, data1[:, 2], color='mediumseagreen', label='Ground-truth renewable maximum')
axs[0, 0].fill_between(steps_1, data[:, 2], data[:, 1], color='mediumseagreen', alpha=0.3, label='Renewable curtailment')
axs[0, 0].plot(steps, max_gen[settings.renewable_ids, :].sum(0), color=colors[4], linestyle='dotted',
         label='Renewable prediction with +30% errors')

axs[0, 0].fill_between(steps_1, max_gen_p, min_gen_p, color=colors[4], alpha=0.3, label='UC adjust capacity area')
axs[0, 0].plot(steps_1, data[:, 1], label='Renewable integration', color=colors[4], linestyle='dashdot')
axs[0, 0].set_title(f'(A1) DAS: renewable day-ahead overestimated',)

# axs[0, 1].imread('./figs/active_power/DAS_53569_84.png')
# axs[0, 1].imshow()
# axs[0, 2].imread('./figs/active_power/DAS_53569_85.png')
# axs[0, 2].imshow()


# TODO: check errors
data = pd.read_csv(f'./new_data_53569_-0.5_2.csv').values
gen_status = np.array(data[:, -1])
gen_status_under = np.array([[float(i) for i in gen_status[j][1:-1].split(' ')] for j in range(data.shape[0])])
data = data[:, :-1].astype(np.float32)
steps_1 = np.array([i for i in range(data.shape[0])])
max_gen_p = data[:, 6]
min_gen_p = data[:, 7]
data1 = pd.read_csv(f'./results/gridsim_v2/selected_episodes/2/data_53569.csv').values
steps = np.asarray([i for i in range(nb_periods)])

axs[0, 1].plot(steps, data1[:, 0], color='darkorange',)
axs[0, 1].plot(steps, ori_max_gen[settings.renewable_ids].sum(0), color=colors[4], linestyle='dotted',
         label='Renewable prediction with -30% errors')
axs[0, 1].plot(steps, data1[:, 2], color='mediumseagreen',)

axs[0, 1].fill_between(steps_1, max_gen_p, min_gen_p, color=colors[4], alpha=0.3, label='Adjust capacity area')
axs[0, 1].fill_between(steps_1, data1[:, 2], data[:, 1], color='mediumseagreen', alpha=0.3)
axs[0, 1].plot(steps_1, data[:, 1], color=colors[4], linestyle='dashdot')
axs[0, 1].set_title(f'(A2) DAS: renewable day-ahead underestimated',)


steps = np.asarray([i for i in range(nb_periods)])
axs[1, 1].plot(steps, g_data[:, 0], color='darkorange',
            # label='Load consumption'
            )

axs[1, 1].plot(steps, g_data[:, 2], color='mediumseagreen',
            # label='Ground-truth renewable maximum'
            )
axs[1, 1].fill_between(steps, g_data[:, 5], g_data[:, 6], color=colors[4], alpha=0.3, label='GridZero adjust capacity area')
axs[1, 1].fill_between(steps, g_data[:, 2], g_data[:, 1], color='mediumseagreen', alpha=0.3)
axs[1, 1].plot(steps, g_data[:, 1], color=colors[4], linestyle='dashdot', label='GridZero consumption')
axs[1, 1].set_title(f'(C) RTS-GridZero: renewable nowcasting')

# axs[1, 1].imread('./figs/active_power/DAS_53569_84.png')
# axs[1, 1].imshow()
# axs[1, 2].imread('./figs/active_power/DAS_53569_85.png')
# axs[1, 2].imshow()

sac_data = pd.read_csv('data_53569_sac.csv').values
gen_status = np.array(sac_data[:, -1])
sac_data = sac_data[:, :-1].astype(np.float32)
gen_status_sac = np.array([[float(i) for i in gen_status[j][1:-1].split(',')] for j in range(sac_data.shape[0])])
axs[1, 0].plot(steps, sac_data[:, 0], color='darkorange',
            )

axs[1, 0].plot(steps, sac_data[:, 2], color='mediumseagreen',
            # label='Ground-truth renewable maximum'
            )

axs[1, 0].fill_between(steps, sac_data[:, 5], sac_data[:, 6], color=colors[4], alpha=0.3, label='SAC adjust capacity area')
axs[1, 0].fill_between(steps, sac_data[:, 2], sac_data[:, 1], color='mediumseagreen', alpha=0.3)
axs[1, 0].plot(steps, sac_data[:, 1], color=colors[4], linestyle='dashdot')
axs[1, 0].set_title(f'(B) RTS-SAC: renewable nowcasting',
                        # fontsize='small'
                        )
# im = axs[3, 1].imshow(gen_status_sac[:, settings.thermal_ids].swapaxes(0, 1), cmap=matplotlib.colormaps['Greys'])
# plt.colorbar(im, ax=
# axs[3, 1].plot(steps_1, gen_status_sac.sum(1), color=colors[1])

fig.supxlabel('Steps', fontsize=20)
fig.supylabel('Power', fontsize=20)
# fig.legend(loc='lower right', fontsize=12)
fig.legend(
    fontsize=12,
    loc='upper center',
           bbox_to_anchor=(0.5, 1.0),
          fancybox=True, shadow=True,
    ncol=5
)
# fig.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 3, sharey=False, sharex=False, constrained_layout=True)
position = 0.0
data = pd.read_csv(f'./new_data_53569_+0.3_1.csv').values
gen_status = np.array(data[:, -1])
gen_status_over = np.array([[float(i) for i in gen_status[j][1:-1].split(' ')] for j in range(data.shape[0])])
# gen_status_over = np.concatenate((gen_status_over, np.zeros((288 - gen_status_over.shape[0], gen_status_over.shape[1]))), axis=0)
data = data[:, :-1].astype(np.float32)
steps_1 = np.array([i for i in range(data.shape[0])])
max_gen_p = data[:, -2] - 50
min_gen_p = data[:, -1]

g_data = pd.read_csv(f'./data_53569.csv').values
gen_status = np.array(g_data[:, -1])
gen_status_gridzero = np.array([[float(i) for i in gen_status[j][1:-1].split(',')] for j in range(g_data.shape[0])])
g_data = g_data[:, :-1].astype(np.float32)

data1 = pd.read_csv(f'./results/gridsim_v2/selected_episodes/2/data_53569.csv').values
steps = np.asarray([i for i in range(288)])

axs[0, 0].plot(steps, data1[:, 0], color='darkorange', label='Load consumption')
axs[0, 0].plot(steps, data1[:, 2], color='mediumseagreen', label='Real renewable maximum')
axs[0, 0].fill_between(steps_1, data[:, 2], data[:, 1], color='mediumseagreen', alpha=0.3, label='Renewable curtailment')
axs[0, 0].plot(steps, max_gen[settings.renewable_ids, :].sum(0), color=colors[4], linestyle='dotted',
         label='Renewable overestimation')

axs[0, 0].fill_between(steps_1, max_gen_p, min_gen_p, color=colors[4], alpha=0.3, label='Adjust capacity area')
axs[0, 0].plot(steps_1, data[:, 1], label='Renewable consumption', color=colors[4], linestyle='dashdot')
axs[0, 0].set_title(f'(A1) DAS: renewable day-ahead overestimated', fontsize=15
                    # y=position
                    )
# axs[0, 0].set_ylabel('Power', fontsize=18)

img = plt.imread('./figs/active_power/DAS_53569_84_cut.png')
axs[0, 1].imshow(img)
axs[0, 1].axis('off')
axs[0, 1].set_title(f'(A2) DAS operational snapshot', fontsize=15
                    # y=position
                    )

img = plt.imread('./figs/active_power/DAS_53569_85_cut.png')
axs[0, 2].imshow(img)
axs[0, 2].axis('off')
axs[0, 2].set_title(f'(A3) DAS snapshot (no thermal unit restarts)', fontsize=15
                    # y=position
                    )

steps = np.asarray([i for i in range(nb_periods)])
axs[1, 0].plot(steps, g_data[:, 0], color='darkorange')

axs[1, 0].plot(steps, g_data[:, 2], color='mediumseagreen')
axs[1, 0].fill_between(steps, g_data[:, 5], g_data[:, 6], color=colors[4], alpha=0.3)
axs[1, 0].fill_between(steps, g_data[:, 2], g_data[:, 1], color='mediumseagreen', alpha=0.3)
axs[1, 0].plot(steps, g_data[:, 1], color=colors[4], linestyle='dashdot')
axs[1, 0].set_title(f'(B1) RTS-GridZero: renewable nowcasting', fontsize=15
                    # y=position
                    )
# axs[1, 0].set_ylabel('Power', fontsize=18)
axs[1, 0].set_xlabel('Steps', fontsize=18)

img = plt.imread('./figs/active_power/GridZero_53569_84_cut.png')
axs[1, 1].imshow(img)
axs[1, 1].axis('off')
axs[1, 1].set_title(f'(B2) RTS-GridZero operational snapshot', fontsize=15
                    # y=position
                    )

img = plt.imread('./figs/active_power/GridZero_53569_85_cut.png')
axs[1, 2].imshow(img)
axs[1, 2].axis('off')
axs[1, 2].set_title(f'(B3) RTS-GridZero snapshot (thermal unit restarts)', fontsize=15)
fig.supylabel('Power', fontsize=18)
# elements = [
#             # Line2D([0], [0], marker='o', color='w', label='Thermal units',
#             #               markerfacecolor='red', markersize=10),
#             # Line2D([0], [0], marker='o', color='w', label='Renewable units',
#             #        markerfacecolor='g', markersize=10),
#             Line2D([0], [0], color='darkorange', label='Load consumption'),
#             Line2D([0], [0], color='mediumseagreen', label='Real renewable capacity'),
#             Line2D([0], [0], linestyle='dashdot', color=colors[4], label='Renewable intgration'),
#             Line2D([0], [0], linestyle='dotted', color=colors[4], label='Renewable estimation'),
#             Patch(facecolor=colors[4], edgecolor=colors[4], label='Adjust capacity area', alpha=0.3),
#             Patch(facecolor='mediumseagreen', edgecolor='mediumseagreen', label='Renewable curtailment', alpha=0.3)
#             ]
# axs[1, 0].legend(
#     handles=elements,
#     #        bbox_to_anchor=(0.3, 0.05),
#     #       fancybox=True, shadow=True,
#     # ncol=5
#     fontsize=10, loc='lower right', ncol=1
# )
# elements = [
#     Line2D([0], [0], marker='o', color='w', label='Thermal units',
#                   markerfacecolor='red', markersize=10),
#     Line2D([0], [0], marker='o', color='w', label='Renewable units',
#            markerfacecolor='g', markersize=10),
#     # Line2D([0], [0], marker='o', color='w', linestyle='dotted', label='Critical difference',
#     #        markerfacecolor='w', markeredgecolor='darkred', markersize=10),
#     Circle((0, 0), radius=5, joinstyle='round', linestyle='dotted', capstyle='round', label='Critical difference', facecolor='w', edgecolor='darkred'),
#     Line2D([0], [0], color='tan', label='Power flow'),
# ]
# axs[1, 2].legend(
#     handles=elements,
#     #        bbox_to_anchor=(0.3, 0.05),
#     #       fancybox=True, shadow=True,
#     # ncol=5
#     fontsize=10, loc='lower right',
#     ncol=3
# )
plt.show()




fig, axs = plt.subplots(2, 3, constrained_layout=True)
img = plt.imread('./figs/active_power/SAC_53569_175_cut.png')
axs[0, 0].imshow(img)
axs[0, 0].axis('off')
axs[0, 0].set_title(f'(A1) RTS-SAC operational snapshot', fontsize=15)

img = plt.imread('./figs/active_power/DAS_53569_175_cut.png')
axs[0, 1].imshow(img)
axs[0, 1].axis('off')
axs[0, 1].set_title(f'(B1) DAS operational snapshot', fontsize=15)

img = plt.imread('./figs/active_power/GridZero_53569_175_cut.png')
axs[0, 2].imshow(img)
axs[0, 2].axis('off')
axs[0, 2].set_title(f'(C1) RTS-GridZero operational snapshot', fontsize=15)

sac_data = pd.read_csv('data_53569_sac.csv').values
gen_status = np.array(sac_data[:, -1])
sac_data = sac_data[:, :-1].astype(np.float32)
gen_status_sac = np.array([[float(i) for i in gen_status[j][1:-1].split(',')] for j in range(sac_data.shape[0])])
axs[1, 0].plot(steps, sac_data[:, 0], color='darkorange')
axs[1, 0].plot(steps, sac_data[:, 2], color='mediumseagreen')
axs[1, 0].fill_between(steps, sac_data[:, 5], sac_data[:, 6], color=colors[4], alpha=0.3, label='SAC adjust capacity area')
axs[1, 0].fill_between(steps, sac_data[:, 2], sac_data[:, 1], color='mediumseagreen', alpha=0.3)
axs[1, 0].plot(steps, sac_data[:, 1], color=colors[4], linestyle='dashdot')
axs[1, 0].set_ylabel('Power', fontsize=18)
axs[1, 0].set_title(f'(A2) RTS-SAC: renewable nowcasting', fontsize=15)

data = pd.read_csv(f'./new_data_53569_-0.5_2.csv').values
gen_status = np.array(data[:, -1])
gen_status_under = np.array([[float(i) for i in gen_status[j][1:-1].split(' ')] for j in range(data.shape[0])])
data = data[:, :-1].astype(np.float32)
steps_1 = np.array([i for i in range(data.shape[0])])
max_gen_p = data[:, 6]
min_gen_p = data[:, 7]
data1 = pd.read_csv(f'./results/gridsim_v2/selected_episodes/2/data_53569.csv').values
steps = np.asarray([i for i in range(nb_periods)])

axs[1, 1].plot(steps, data1[:, 0], color='darkorange',)
axs[1, 1].plot(steps, ori_max_gen[settings.renewable_ids].sum(0), color=colors[4], linestyle='dotted',
         label='Renewable prediction with -30% errors')
axs[1, 1].plot(steps, data1[:, 2], color='mediumseagreen')
axs[1, 1].fill_between(steps_1, max_gen_p, min_gen_p, color=colors[4], alpha=0.3, label='Adjust capacity area')
axs[1, 1].fill_between(steps_1, data1[:, 2], data[:, 1], color='mediumseagreen', alpha=0.3)
axs[1, 1].plot(steps_1, data[:, 1], color=colors[4], linestyle='dashdot')
axs[1, 1].set_title(f'(B2) DAS: renewable day-ahead underestimated', fontsize=15)

steps = np.asarray([i for i in range(nb_periods)])
axs[1, 2].plot(steps, g_data[:, 0], color='darkorange')
axs[1, 2].plot(steps, g_data[:, 2], color='mediumseagreen')
axs[1, 2].fill_between(steps, g_data[:, 5], g_data[:, 6], color=colors[4], alpha=0.3, label='GridZero adjust capacity area')
axs[1, 2].fill_between(steps, g_data[:, 2], g_data[:, 1], color='mediumseagreen', alpha=0.3)
axs[1, 2].plot(steps, g_data[:, 1], color=colors[4], linestyle='dashdot')
axs[1, 2].set_title(f'(C2) RTS-GridZero: renewable nowcasting', fontsize=15)
fig.supxlabel('Steps', fontsize=18)
plt.show()



fig, axs = plt.subplots(2, 3, constrained_layout=True)
img = plt.imread('./figs/line_loading/SAC_53569_33_cut.png')
axs[0, 0].imshow(img)
axs[0, 0].axis('off')
axs[0, 0].set_title(f'(A1) RTS-SAC operational snapshot', fontsize=15)

img = plt.imread('./figs/line_loading/DAS_53569_33_cut.png')
axs[0, 1].imshow(img)
axs[0, 1].axis('off')
axs[0, 1].set_title(f'(A2) DAS operational snapshot', fontsize=15)

img = plt.imread('./figs/line_loading/GridZero_53569_33_cut.png')
axs[0, 2].imshow(img)
axs[0, 2].axis('off')
axs[0, 2].set_title(f'(A3) RTS-GridZero operational snapshot', fontsize=15)

img = plt.imread('./figs/reactive_power/SAC_53569_23_cut.png')
axs[1, 0].imshow(img)
axs[1, 0].axis('off')
axs[1, 0].set_title(f'(B1) RTS-SAC operational snapshot', fontsize=15)

img = plt.imread('./figs/reactive_power/DAS_53569_23_cut.png')
axs[1, 1].imshow(img)
axs[1, 1].axis('off')
axs[1, 1].set_title(f'(B2) DAS operational snapshot', fontsize=15)

img = plt.imread('./figs/reactive_power/GridZero_53569_23_cut.png')
axs[1, 2].imshow(img)
axs[1, 2].axis('off')
axs[1, 2].set_title(f'(B3) RTS-GridZero operational snapshot', fontsize=15)
plt.show()


# reduce flexibility retrofit fee
g_data = pd.read_csv(f'./data_53569.csv').values
# gen_status = np.array(g_data[:, -1])
# gen_status_gridzero = np.array([[float(i) for i in gen_status[j][1:-1].split(',')] for j in range(g_data.shape[0])])
g_data = g_data[:, :-1].astype(np.float32)
data_min50 = pd.read_csv(f'./new_data_53569_-0.5_2.csv').values
# gen_status = np.array(data_min50[:, -1])
# gen_status_under = np.array([[float(i) for i in gen_status[j][1:-1].split(' ')] for j in range(data.shape[0])])
data_min50 = data_min50[:, :-1].astype(np.float32)
data_min40 = pd.read_csv(f'./new_data_53569_-0.5_min0.4.csv').values
# gen_status = np.array(data_min40[:, -1])
# gen_status_gridzero = np.array([[float(i) for i in gen_status[j][1:-1].split(',')] for j in range(g_data.shape[0])])
data_min40 = data_min40[:, :-1].astype(np.float32)
data_min30 = pd.read_csv(f'./new_data_53569_-0.5_min0.3.csv').values
# gen_status = np.array(data_min30[:, -1])
# gen_status_gridzero = np.array([[float(i) for i in gen_status[j][1:-1].split(',')] for j in range(g_data.shape[0])])
data_min30 = data_min30[:, :-1].astype(np.float32)

fig, axs = plt.subplots(2, 2, sharey=True, sharex=True, constrained_layout=True, figsize=(8, 4))
axs[0, 0].plot(steps, data_min50[:, 0], color='darkorange',
            label='Load consumption'
            )
axs[0, 0].plot(steps, ori_max_gen[settings.renewable_ids].sum(0), color=colors[4], linestyle='dotted',
         label='Renewable forecasts')
# axs[0, 1].plot(steps, data[:, 5], color=colors[0], linestyle='dotted', label='Renewable prediction with -30% errors')
axs[0, 0].plot(steps, data_min50[:, 2], color='mediumseagreen',
            label='Real renewable capacity'
            )
axs[0, 0].fill_between(steps_1, data_min50[:, 6], data_min50[:, 7], color=colors[4], alpha=0.3, label='Adjust capacity area')
axs[0, 0].fill_between(steps_1, data_min50[:, 2], data_min50[:, 1], color='mediumseagreen', alpha=0.3, label='Renewable curtailment')
axs[0, 0].plot(steps_1, data_min50[:, 1], label='Renewable integration', color=colors[4], linestyle='dashdot')
axs[0, 0].set_title(r'(A1) DAS: $P_{min} = 0.5P_{max}$')

axs[0, 1].plot(steps, data_min40[:, 0], color='darkorange'
            )
axs[0, 1].plot(steps, ori_max_gen[settings.renewable_ids].sum(0), color=colors[4], linestyle='dotted')
# axs[0, 1].plot(steps, data[:, 5], color=colors[0], linestyle='dotted', label='Renewable prediction with -30% errors')
axs[0, 1].plot(steps, data_min40[:, 2], color='mediumseagreen',
            )
axs[0, 1].fill_between(steps_1, data_min40[:, 6], data_min40[:, 7], color=colors[4], alpha=0.3)
axs[0, 1].fill_between(steps_1, data_min40[:, 2], data_min40[:, 1], color='mediumseagreen', alpha=0.3)
axs[0, 1].plot(steps_1, data_min40[:, 1], color=colors[4], linestyle='dashdot')
axs[0, 1].set_title(r'(A2) DAS: $P_{min} = 0.4P_{max}$')

axs[1, 0].plot(steps, data_min30[:, 0], color='darkorange',
            )
axs[1, 0].plot(steps, ori_max_gen[settings.renewable_ids].sum(0), color=colors[4], linestyle='dotted',
         )
# axs[1, 0].plot(steps, data[:, 5], color=colors[0], linestyle='dotted', label='Renewable prediction with -30% errors')
axs[1, 0].plot(steps, data_min30[:, 2], color='mediumseagreen'
            )
axs[1, 0].fill_between(steps_1, data_min30[:, 6], data_min30[:, 7], color=colors[4], alpha=0.3)
axs[1, 0].fill_between(steps_1, data_min30[:, 2], data_min30[:, 1], color='mediumseagreen', alpha=0.3)
axs[1, 0].plot(steps_1, data_min30[:, 1], color=colors[4], linestyle='dashdot')
axs[1, 0].set_title(r'(A3) DAS: $P_{min} = 0.3P_{max}$')

axs[1, 1].plot(steps, g_data[:, 0], color='darkorange'
            )
# axs[1, 0].plot(steps, data[:, 5], color=colors[0], linestyle='dotted', label='Renewable prediction with -30% errors')
axs[1, 1].plot(steps, g_data[:, 2], color='mediumseagreen'
            )
axs[1, 1].fill_between(steps_1, g_data[:, 5], g_data[:, 6], color=colors[4], alpha=0.3)
axs[1, 1].fill_between(steps_1, g_data[:, 2], g_data[:, 1], color='mediumseagreen', alpha=0.3)
axs[1, 1].plot(steps_1, g_data[:, 1], color=colors[4], linestyle='dashdot')
axs[1, 1].set_title(r'(B) RTS-GridZero: $P_{min}=0.5P_{max}$')
fig.supxlabel('Steps', fontsize=18)
fig.supylabel('Power', fontsize=18)
# fig.legend(loc='lower right', fontsize=12)
# fig.legend(
#     loc='upper center',
#     fontsize=12,
#            bbox_to_anchor=(0.5, 1.0),
#           fancybox=True, shadow=True,
#     ncol=5
# )
plt.show()






# data1 = pd.read_csv(f'./new_data_53569_+{percentage}.csv').values
# data2 = pd.read_csv(f'./new_data_53569_-{percentage}.csv').values
# data = pd.read_csv(f'./results/gridsim_v2/selected_episodes/2/data_53569.csv').values
#
# steps = np.asarray([i for i in range(288)])
# plt.plot(steps, data[:, 0], color='darkorange', label='Load consumption')
# plt.plot(steps, data[:, 1], color=colors[4], linestyle='dashdot', label='GridZero consumption')
#
# plt.plot(steps, max_gen[settings.renewable_ids].sum(0), color=colors[0], linestyle='dotted',
#          label='Renewable prediction with +30% errors')
# try:
#     plt.plot(steps, data1[:, 1], color=colors[0], label='Renewable consumption with +30% error')
# except:
#     num = 288 - data1[:, 1].shape[0]
#     tmp = np.concatenate([data1[:, 1], np.zeros(num)])
#     plt.plot(steps, tmp, color=colors[0], label='Renewable consumption with +30% error')
#
# plt.plot(steps, ori_max_gen[settings.renewable_ids].sum(0), color=colors[1], linestyle='dotted',
#              label='Renewable prediction with -30% errors')
# try:
#     plt.plot(steps, data2[:, 1], color=colors[1], label='Renewable consumption with -30% error')
# except:
#     num = 288 - data2[:, 1].shape[0]
#     tmp = np.concatenate([data2[:, 1], np.zeros(num)])
#     plt.plot(steps, tmp, color=colors[1], label='Renewable consumption with -30% error')
#
# plt.plot(steps, data[:, 2], color='mediumseagreen', linestyle='dotted', label='Renewable maximum')
# plt.title(f'start_idx=53569',
#                         # fontsize='small'
#                         )
# plt.xlabel('Steps', fontsize=20)
# plt.ylabel('Power', fontsize=20)
# plt.legend(loc='upper right', fontsize=15)
# plt.show()


# row_num = 2
# col_num = 4
# # fig, axs = plt.subplots(row_num, col_num, sharey=False, sharex=True, constrained_layout=True)
# data = pd.read_csv('./constraint_data.csv').values
# train_steps = np.concatenate([data[:, 0] for _ in range(3)])
# bal_p_vio_data = np.concatenate([data[:, i] for i in range(1, 4)])
# hard_overflow_data = np.concatenate([data[:, i] for i in range(4, 7)])
# reactive_p_vio_data = np.concatenate([data[:, i] for i in range(7, 10)])
# renewable_consump_data = np.concatenate([data[:, i] for i in range(10, 13)])
# running_cost = np.concatenate([data[:, i] for i in range(13, 16)])
# soft_overflow_data = np.concatenate([data[:, i] for i in range(16, 19)])
# vol_vio_data = np.concatenate([data[:, i] for i in range(19, 22)])
# plt.subplot(3, 3, 1)
# sns.lineplot(x=train_steps/1000, y=100*bal_p_vio_data)
# plt.title('Balanced power constraint violation',
#           fontsize=15
#           )
# # plt.grid()
# # plt.xlabel('training steps (K steps)')
# plt.ylabel('balanced power constraint violation rate(%)', fontsize=12)
# plt.subplot(3, 3, 2)
# sns.lineplot(x=train_steps/1000, y=100*hard_overflow_data)
# plt.title('Line hard overflow', fontsize=15)
# # plt.grid()
# # plt.xlabel('training steps (K steps)')
# plt.ylabel('line hard overflow rate(%)', fontsize=12)
# plt.subplot(3, 3, 3)
# sns.lineplot(x=train_steps/1000, y=100*reactive_p_vio_data)
# plt.title('Reactive power constraints violation', fontsize=15)
# # plt.grid()
# # plt.xlabel('training steps (K steps)')
# plt.ylabel('reactive power constraint violation rate(%)', fontsize=12)
# plt.subplot(3, 3, 4)
# sns.lineplot(x=train_steps/1000, y=100*renewable_consump_data)
# plt.title('Renewable consumption', fontsize=15)
# # plt.grid()
# # plt.xlabel('training steps (K steps)')
# plt.ylabel('renewable consumption rate(%)', fontsize=12)
# plt.subplot(3, 3, 5)
# sns.lineplot(x=train_steps/1000, y=running_cost)
# plt.title('Running cost', fontsize=15)
# # plt.grid()
# plt.xlabel('training steps (K steps)', fontsize=12)
# plt.ylabel('running cost', fontsize=12)
# plt.subplot(3, 3, 6)
# sns.lineplot(x=train_steps/1000, y=100*soft_overflow_data)
# plt.title('Line soft overflow', fontsize=15)
# # plt.grid()
# plt.xlabel('training steps (K steps)', fontsize=12)
# plt.ylabel('line soft overflow rate(%)', fontsize=12)
# plt.subplot(3, 3, 7)
# sns.lineplot(x=train_steps/1000, y=100*vol_vio_data)
# plt.title('Bus voltage constraint violation', fontsize=15)
# # plt.grid()
# plt.xlabel('training steps (K steps)', fontsize=12)
# plt.ylabel('bus volatge violation rate(%)', fontsize=12)
#
# # sub = sns.lineplot(x=train_steps/1000, y=hard_overflow_data, ax=axs[0, 1])
# # sub.set_title('Line hard overflow')
# # sub.grid()
# # sub = sns.lineplot(x=train_steps/1000, y=reactive_p_vio_data, ax=axs[0, 2])
# # sub.set_title('Reactive power constraints violation')
# # sub.grid()
# # sub = sns.lineplot(x=train_steps/1000, y=renewable_consump_data, ax=axs[0, 3])
# # sub.set_title('Renewable consumption')
# # sub.grid()
# # sub = sns.lineplot(x=train_steps/1000, y=running_cost, ax=axs[1, 0])
# # sub.set_title('Running cost')
# # sub.grid()
# # sub = sns.lineplot(x=train_steps/1000, y=soft_overflow_data, ax=axs[1, 1])
# # sub.set_title('Line soft overflow')
# # sub.grid()
# # sub = sns.lineplot(x=train_steps/1000, y=vol_vio_data, ax=axs[1, 2])
# # sub.set_title('Bus voltage constraints violation')
# # sub.grid()
# plt.show()


# start_sample_idx = [
#     53569, 36289, 67105, 4897
# ]
# row_num = 2
# col_num = 2
# fig, axs = plt.subplots(row_num, col_num, sharey=True, sharex=True, constrained_layout=True)
# for epi in range(row_num*col_num):
#     row = epi // col_num
#     col = epi % col_num
#     data = pd.read_csv(f'./results/gridsim_v2/selected_episodes/2/data_{start_sample_idx[epi]}.csv').values
#     ac_data = pd.read_csv(f'./results/gridsim_v2/trad_expert/ac_opf/data_{start_sample_idx[epi]}.csv').values
#     ddpg_data = pd.read_csv(f'./results/gridsim_v2/ddpg_log/data_{start_sample_idx[epi]}.csv').values
#     sac_data = pd.read_csv(f'./results/gridsim_v2/sac_log/data_{start_sample_idx[epi]}.csv').values
#     steps = np.asarray([i for i in range(288)])
#     axs[row, col].plot(steps, data[:, 0], color='darkorange', label='Load' if row==0 and col==0 else None)
#     axs[row, col].plot(steps, data[:, 1], color=colors[4], linestyle='dashdot', label='GridZero consumption' if row==0 and col==0 else None)
#     axs[row, col].plot(steps, ac_data[:, 1], color=colors[3], linestyle='dashed', label='AC-UC-ED consumption' if row==0 and col==0 else None)
#     try:
#         axs[row, col].plot(steps, ddpg_data[:, 1], color=colors[0], label='DDPG consumption' if row==0 and col==0 else None)
#     except:
#         num = 288 - ddpg_data[:, 1].shape[0]
#         tmp = np.concatenate([ddpg_data[:, 1], np.zeros(num)])
#         axs[row, col].plot(steps, tmp, color=colors[0], label='DDPG consumption' if row==0 and col==0 else None)
#
#     try:
#         axs[row, col].plot(steps, sac_data[:, 1], color=colors[1], label='SAC consumption' if row==0 and col==0 else None)
#     except:
#         num = 288 - sac_data[:, 1].shape[0]
#         tmp = np.concatenate([sac_data[:, 1], np.zeros(num)])
#         axs[row, col].plot(steps, tmp, color=colors[1], label='SAC consumption' if row==0 and col==0 else None)
#     axs[row, col].plot(steps, data[:, 2], color='mediumseagreen', linestyle='dotted', label='Renewable maximum' if row==0 and col==0 else None)
#     axs[row, col].set_title(f'start_idx={start_sample_idx[epi]}',
#                             # fontsize='small'
#                             )
#     # axs[row, col].grid()
#     # axs[row, col].set_xlabel("Steps")
#     # axs[row, col].set_ylabel("Power")
#
#     # if row == 0 and col == 1:
#     #     axs[row, col].legend(loc='center right',
#     #                          fontsize=15
#     #                          )
# # axs.set_xlabel("Steps")
# # axs.set_ylabel("Power")
# fig.supxlabel('Steps', fontsize=20)
# fig.supylabel('Power', fontsize=20)
# fig.legend(loc='upper right', fontsize=15)
# plt.show()


# start_sample_idx = [
#     53569, 36289, 67105, 4897
# ]
# row_num = 2
# col_num = 2
# fig, axs = plt.subplots(row_num, col_num, sharey=True, sharex=True, constrained_layout=True)
# for epi in range(row_num*col_num):
#     row = epi // col_num
#     col = epi % col_num
#     data = pd.read_csv(f'./results/gridsim_v2/selected_episodes/2/data_{start_sample_idx[epi]}.csv').values
#     ac_data = pd.read_csv(f'./results/gridsim_v2/trad_expert/ac_opf/data_{start_sample_idx[epi]}.csv').values
#     ddpg_data = pd.read_csv(f'./results/gridsim_v2/ddpg_log/data_{start_sample_idx[epi]}.csv').values
#     sac_data = pd.read_csv(f'./results/gridsim_v2/sac_log/data_{start_sample_idx[epi]}.csv').values
#     steps = np.asarray([i for i in range(288)])
#     axs[row, col].plot(steps, data[:, 0], color='darkorange', label='Load' if row==0 and col==0 else None)
#     axs[row, col].plot(steps, data[:, 1], color=colors[4], linestyle='dashdot', label='GridZero consumption' if row==0 and col==0 else None)
#     axs[row, col].plot(steps, data[:, 2], color='mediumseagreen', linestyle='dotted', label='Renewable maximum' if row==0 and col==0 else None)
#     axs[row, col].set_title(f'start_idx={start_sample_idx[epi]}',
#                             # fontsize='small'
#                             )
#     # axs[row, col].grid()
#     # axs[row, col].set_xlabel("Steps")
#     # axs[row, col].set_ylabel("Power")
#     axs2 = axs[row, col].twinx()
#     axs2.plot(steps, data[:, -1], color='grey', label='Closable gen num' if row==0 and col==0 else None)
#     axs2.plot(steps, data[:, -2], color='black', label='Closed gen num' if row==0 and col==0 else None)
#     axs2.set_ylim(0, 30)
#     if col==1:
#         axs2.set_ylabel('Gen num')
#
#     # if row == 0 and col == 1:
#     #     axs[row, col].legend(loc='center right',
#     #                          fontsize=15
#     #                          )
# # axs.set_xlabel("Steps")
# # axs.set_ylabel("Power")
# fig.supxlabel('Steps', fontsize=20)
# fig.supylabel('Power', fontsize=20)
# fig.legend(loc='upper right', fontsize=15)
# plt.show()


start_sample_idx = [
            22753, 16129, 74593, 45793, 32257, 53569, 13826,
            26785,
            1729, 17281,
            34273, 36289, 44353, 52417, 67105, 75169, 289, 4897, 15841, 31969,]
# row_num = 5
# col_num = 4
# fig, axs = plt.subplots(row_num, col_num, sharey=True, sharex=True, constrained_layout=True)
# for epi in range(20):
#     row = epi // col_num
#     col = epi % col_num
#     data = pd.read_csv(f'./results/gridsim_v2/selected_episodes/2/data_{start_sample_idx[epi]}.csv').values
#     ac_data = pd.read_csv(f'./results/gridsim_v2/trad_expert/ac_opf/data_{start_sample_idx[epi]}.csv').values
#     ddpg_data = pd.read_csv(f'./results/gridsim_v2/ddpg_log/data_{start_sample_idx[epi]}.csv').values
#     sac_data = pd.read_csv(f'./results/gridsim_v2/sac_log/data_{start_sample_idx[epi]}.csv').values
#     steps = np.asarray([i for i in range(288)])
#     axs[row, col].plot(steps, data[:, 0], color='darkorange', label='Load' if row==0 and col==0 else None)
#     axs[row, col].plot(steps, data[:, 1], color=colors[4], linestyle='dashdot', label='GridZero consumption' if row==0 and col==0 else None)
#     axs[row, col].plot(steps, ac_data[:, 1], color=colors[3], linestyle='dashed', label='UC-ED consumption' if row==0 and col==0 else None)
#     # try:
#     #     axs[row, col].plot(steps, ddpg_data[:, 1], color=colors[0], label='DDPG consumption' if row==0 and col==0 else None)
#     # except:
#     #     num = 288 - ddpg_data[:, 1].shape[0]
#     #     tmp = np.concatenate([ddpg_data[:, 1], np.zeros(num)])
#     #     axs[row, col].plot(steps, tmp, color=colors[0], label='DDPG consumption' if row==0 and col==0 else None)
#     #
#     # try:
#     #     axs[row, col].plot(steps, sac_data[:, 1], color=colors[1], label='SAC consumption' if row==0 and col==0 else None)
#     # except:
#     #     num = 288 - sac_data[:, 1].shape[0]
#     #     tmp = np.concatenate([sac_data[:, 1], np.zeros(num)])
#     #     axs[row, col].plot(steps, tmp, color=colors[1], label='SAC consumption' if row==0 and col==0 else None)
#     axs[row, col].plot(steps, data[:, 2], color='mediumseagreen', linestyle='dotted', label='Renewable maximum' if row==0 and col==0 else None)
#     axs[row, col].set_title(f'start_idx={start_sample_idx[epi]}',
#                             fontsize=12
#                             )
#
# fig.supxlabel('Steps', fontsize=20)
# fig.supylabel('Power', fontsize=20)
# fig.legend(loc='upper right')
# plt.show()


row_num = 5
col_num = 4
fig, axs = plt.subplots(row_num, col_num, sharey=True, sharex=True, constrained_layout=True)
for epi in range(20):
    row = epi // col_num
    col = epi % col_num
    data = pd.read_csv(f'./results/gridsim_v2/selected_episodes/6/data_{start_sample_idx[epi]}.csv').values
    gen_status = np.array(data[:, -1])
    gen_status_gridzero = np.array([[float(i) for i in gen_status[j][1:-1].split(',')] for j in range(data.shape[0])])
    data = data[:, :-1].astype(np.float32)
    # ac_data = pd.read_csv(f'./results/gridsim_v2/trad_expert/ac_opf/data_{start_sample_idx[epi]}.csv').values
    # ddpg_data = pd.read_csv(f'./results/gridsim_v2/ddpg_log/data_{start_sample_idx[epi]}.csv').values
    # sac_data = pd.read_csv(f'./results/gridsim_v2/sac_log/data_{start_sample_idx[epi]}.csv').values
    steps = np.asarray([i for i in range(288)])
    axs[row, col].plot(steps, data[:, 0], color='darkorange', label='Load consumption' if row == 0 and col == 0 else None)
    axs[row, col].plot(steps, data[:, 1], color=colors[4], linestyle='dashdot',
                       label='GridZero integration' if row == 0 and col == 0 else None)
    # axs[row, col].plot(steps, ac_data[:, 1], color=colors[3], linestyle='dashed',
    #                    label='UC-ED consumption' if row == 0 and col == 0 else None)
    # try:
    #     axs[row, col].plot(steps, ddpg_data[:, 1], color=colors[0], label='DDPG consumption' if row==0 and col==0 else None)
    # except:
    #     num = 288 - ddpg_data[:, 1].shape[0]
    #     tmp = np.concatenate([ddpg_data[:, 1], np.zeros(num)])
    #     axs[row, col].plot(steps, tmp, color=colors[0], label='DDPG consumption' if row==0 and col==0 else None)
    #
    # try:
    #     axs[row, col].plot(steps, sac_data[:, 1], color=colors[1], label='SAC consumption' if row==0 and col==0 else None)
    # except:
    #     num = 288 - sac_data[:, 1].shape[0]
    #     tmp = np.concatenate([sac_data[:, 1], np.zeros(num)])
    #     axs[row, col].plot(steps, tmp, color=colors[1], label='SAC consumption' if row==0 and col==0 else None)
    axs[row, col].fill_between(steps, data[:, 5], data[:, 6], color=colors[4], alpha=0.3,
                           label='Adjust capacity area' if row == 0 and col == 0 else None)
    axs[row, col].fill_between(steps, data[:, 2], data[:, 1], color='mediumseagreen', alpha=0.3,
                           label='Renewable curtailment' if row == 0 and col == 0 else None)
    axs[row, col].plot(steps, data[:, 2], color='mediumseagreen',
                       label='Real renewable capacity' if row == 0 and col == 0 else None)
    axs[row, col].set_title(f'start_idx={start_sample_idx[epi]}',
                            fontsize=12
                            )

fig.supxlabel('Steps', fontsize=18)
fig.supylabel('Power', fontsize=18)
fig.legend(loc='upper right')
# fig.legend(bbox_to_anchor=(0.5, 1.1), loc=9, borderaxespad=0, ncol=5)
plt.show()


T = np.asarray([20.8, 21.2, 900.1, 8557.3, 45.3])
# T = np.asarray([21.2, 8557.3, 45.3])
scores = np.asarray([331.2, 256.8, 649.4, 690.1, 628.3])
# scores = np.asarray([331.2, 649.4, 628.3])
speed = 1/T
types = []
types.append(plt.scatter(math.log10(T[1]), scores[1], c=colors[1]))
types.append(plt.scatter(math.log10(T[3]), scores[2], c=colors[3]))
types.append(plt.scatter(math.log10(T[4]), scores[4], c=colors[4], marker='*', s=80))
# for i in range(len(T)):
#     # if i == 4:
#     if i == 2:
#         tmp = plt.scatter(math.log(T[i]), scores[i], c=colors[i], marker='*', s=80)
#     else:
#         tmp = plt.scatter(math.log(T[i]), scores[i], c=colors[i])
#     types.append(tmp)
# plt.xlabel("Computation Speed (1 / episode time)")
plt.xlabel("log(episode time)", fontsize=15)
plt.ylabel("Scores", fontsize=15)
# plt.legend(types, ["DDPG", "SAC", "DC-UC-ED", "AC-UC-ED", "GridZero"], loc='lower right')
plt.legend(types, ["SAC", "UC-ED", "GridZero"], loc='lower right')
# plt.grid()
plt.show()


is_envstep = False
gridzero_data = pd.read_csv('./gridzero.csv').values
train_steps_baseline = np.concatenate([gridzero_data[:, 0]*5 for _ in range(3)])
train_steps = np.concatenate([gridzero_data[:, 0] for _ in range(4)])
train_steps_expert = np.concatenate([gridzero_data[:, 0]*5 for _ in range(4)])

env_steps = [gridzero_data[:, 5] for i in range(5, 9)]
env_steps = np.concatenate(env_steps)
scores = [gridzero_data[:, i] for i in range(1, 5)]
scores = np.concatenate(scores)
sac_scores = np.concatenate([gridzero_data[:, i] for i in range(9, 12)])
ddpg_scores = np.concatenate([gridzero_data[:, i] for i in range(12, 15)])
env_steps_baseline = np.concatenate([gridzero_data[:, 5] for _ in range(9, 12)])
dc_opf_scores = 649.4 * np.ones_like(scores)
import ipdb
ipdb.set_trace()
# ac_opf_scores = 690.1 * np.ones_like(scores)
sns.lineplot(x=env_steps/1e6, y=scores, color=colors[4], label='GridZero')
# sns.lineplot(x=env_steps/1e6, y=dc_opf_scores, color=colors[3], label='UC-ED')
# sns.lineplot(x=env_steps/1e6, y=ac_opf_scores, color=colors[3], label='AC-UC-ED')
sns.lineplot(x=env_steps_baseline/1e6, y=sac_scores, color=colors[1], label='SAC')
# sns.lineplot(x=env_steps_baseline/1e6, y=ddpg_scores, color=colors[0], label='DDPG')
plt.xlabel("env steps (million steps)", fontsize=15)
plt.ylabel("Scores", fontsize=15)
plt.legend(
    # types, ["DDPG", "SAC", "DC-UC-ED", "AC-UC-ED", "GridZero"],
    loc='lower right')
# plt.grid()
plt.show()

scores = [gridzero_data[:, i] for i in range(1, 5)]
scores = np.concatenate(scores)
sac_scores = np.concatenate([gridzero_data[:, i] for i in range(9, 12)])
ddpg_scores = np.concatenate([gridzero_data[:, i] for i in range(12, 15)])
env_steps_baseline = np.concatenate([gridzero_data[:, 5] for _ in range(9, 12)])
dc_opf_scores = 649.4 * np.ones_like(scores)
ac_opf_scores = 690.1 * np.ones_like(scores)
sns.lineplot(x=train_steps/1e3, y=scores, color=colors[4], label='GridZero')
plt.xlabel("training steps (K steps)", fontsize=15)
plt.ylabel("Scores", fontsize=15)
plt.legend(loc='lower right')
# plt.grid()
plt.show()


