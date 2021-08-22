import numpy as np
import gym

import scipy as sc
from scipy.special import comb
import torch

from sds.distributions.gamma import Gamma
from sds.models import HybridController
from reps.mb_hbreps_hbvf import hbREPS_hbVf

# np.random.seed(1337)

torch.set_num_threads(1)

env = gym.make('Pole-RL-v0')
env._max_episode_steps = 5000
env.unwrapped.dt = 0.01
env.unwrapped.sigma = 1e-8
# env.seed(1337)

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

dyn = torch.load(open('rarhmm_pole.pkl', 'rb'))
nb_modes = dyn.nb_states

# ctl type
ctl_type = 'ard'
ctl_degree = 1

# ctl_prior
feat_dim = int(comb(ctl_degree + state_dim, ctl_degree)) - 1
input_dim = feat_dim + 1
output_dim = act_dim

likelihood_precision_prior = Gamma(dim=1, alphas=np.ones((1,)) + 1e-8,
                                   betas=25. * np.ones((1,)))
parameter_precision_prior = Gamma(dim=input_dim, alphas=np.ones((input_dim,)) + 1e-8,
                                  betas=1e1 * np.ones((input_dim,)))

ctl_prior = {'likelihood_precision_prior': likelihood_precision_prior,
             'parameter_precision_prior': parameter_precision_prior}

ctl_kwargs = {'degree': ctl_degree}
ctl = HybridController(dynamics=dyn, ctl_type=ctl_type,
                       ctl_prior=ctl_prior, ctl_kwargs=ctl_kwargs)

# init controller
Ks = np.stack([np.zeros((output_dim, input_dim))] * nb_modes, axis=0)
lmbdas = np.stack([1. / 25. * np.eye(output_dim)] * nb_modes, axis=0)
ctl.controls.params = Ks, lmbdas

from reps.mb_hbreps_hbvf import RFFVfunction, PolyVfunction

# vfunc = RFFVfunction(nb_modes, state_dim, nb_feat=25,
#                      scale=[0.25, 1.5], mult=0.5)
vfunc = PolyVfunction(nb_modes, state_dim, degree=2)

hbreps = hbREPS_hbVf(env=env, dyn=dyn, ctl=ctl,
                     kl_bound=0.25, discount=0.98,
                     vfunc=vfunc, vf_reg=1e-8)

ctl_mstep_kwargs = {'nb_iter': 5}
hbreps.run(nb_iter=10, nb_train_samples=2500,
           nb_eval_rollouts=25, nb_eval_steps=100,
           ctl_mstep_kwargs=ctl_mstep_kwargs)

rollouts, _ = hbreps.evaluate(nb_rollouts=25, nb_steps=250)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=hbreps.state_dim + hbreps.act_dim, figsize=(12, 4))
for roll in rollouts:
    for k, col in enumerate(ax[:-1]):
        col.plot(roll['x'][:, k])
    ax[-1].plot(roll['uc'])
plt.show()

# def beautify(ax):
#     ax.set_frame_on(True)
#     ax.minorticks_on()
#
#     ax.grid(True)
#     ax.grid(linestyle=':')
#
#     ax.tick_params(which='both', direction='in',
#                    bottom=True, labelbottom=True,
#                    top=True, labeltop=False,
#                    right=True, labelright=False,
#                    left=True, labelleft=True)
#
#     ax.tick_params(which='major', length=6)
#     ax.tick_params(which='minor', length=3)
#
#     ax.autoscale(tight=True)
#     # ax.set_aspect('equal')
#
#     if ax.get_legend():
#         ax.legend(loc='best')
#
#     return ax
#
#
# xlim = (-0.25, 0.25)
# ylim = (-1.5, 1.5)
#
# npts = 18
#
# x = np.linspace(*xlim, npts)
# y = np.linspace(*ylim, npts)
#
# X, Y = np.meshgrid(x, y)
# XYi = np.stack((X, Y))
# XYn = np.zeros((2, npts, npts))
#
# hr = 3
# XYh = np.zeros((hr, 2, npts, npts))
#
# ##
# env.reset()
# for i in range(npts):
#     for j in range(npts):
#         XYh[0, :, i, j] = XYi[:, i, j]
#         for t in range(1, hr):
#             XYh[t, :, i, j] = env.unwrapped.fake_step(XYh[t - 1, :, i, j], np.array([0.0]))
#
# env.reset()
# for i in range(npts):
#     for j in range(npts):
#         hist_obs, hist_act = XYh[..., i, j], np.zeros((hr, act_dim))
#         u = hbreps.ctl.action(hist_obs, hist_act, False, False)[-1]
#         XYn[:, i, j] = env.unwrapped.fake_step(XYh[-1, :, i, j], u)
#
# dXY = XYn - XYh[-1, ...]
#
# # re-interpolate data for streamplot
# xh, yh = XYh[-1, 0, 1, :], XYh[-1, 1, :, 0]
# xi = np.linspace(xh.min(), xh.max(), x.size)
# yi = np.linspace(yh.min(), yh.max(), y.size)
#
# from scipy.interpolate import interp2d
#
# dxh, dyh = dXY[0, ...], dXY[1, ...]
# dxi = interp2d(xh, yh, dxh)(xi, yi)
# dyi = interp2d(xh, yh, dyh)(xi, yi)
#
# import matplotlib.pyplot as plt
#
# fig = plt.figure(figsize=(5, 5), frameon=True)
# ax = fig.gca()
#
# ax.streamplot(xi, yi, dxi, dyi,
#               color='magenta', linewidth=1, density=1.25,
#               arrowstyle='-|>', arrowsize=1.,
#               minlength=0.25)
#
# ax = beautify(ax)
# ax.grid(False)
#
# ax.set_xlim((xh.min(), xh.max()))
# ax.set_ylim((yh.min(), yh.max()))
#
# plt.show()
# #
# # from tikzplotlib import save
# # save("hbreps_pole_rl.tex")
