import numpy as np
import gym

import scipy as sc
from scipy.special import comb
import torch

from sds.distributions.gamma import Gamma
from sds.models import HybridController
from reps.mf_hbreps_hbvf import hbREPS_hbVf

# np.random.seed(1337)
# torch.manual_seed(1337)

torch.set_num_threads(1)

env = gym.make('Pendulum-RL-v1')
env._max_episode_steps = 5000
env.unwrapped.dt = 0.02
env.unwrapped.sigma = 1e-4
# env.seed(1337)

state_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

dyn = torch.load(open('./rarhmm_pendulum_cart.pkl', 'rb'))
nb_modes = dyn.nb_states

# ctl type
ctl_type = 'ard'
ctl_degree = 3

# ctl_prior
feat_dim = int(comb(ctl_degree + state_dim, ctl_degree)) - 1
input_dim = feat_dim + 1
output_dim = act_dim

likelihood_precision_prior = Gamma(dim=1, alphas=np.ones((1,)) + 1e-8,
                                   betas=1. * np.ones((1,)))
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

from reps.mf_hbreps_hbvf import RFFVfunction, PolyVfunction

# vfunc, vf_reg = RFFVfunction(nb_modes, state_dim, nb_feat=75,
#                              scale=[1., 1., 8.0], mult=0.5), 1e-8
vfunc, vf_reg = PolyVfunction(nb_modes, state_dim, degree=3), 1e-8

hbreps = hbREPS_hbVf(env=env, dyn=dyn, ctl=ctl,
                     kl_bound=0.5, discount=0.985,
                     vfunc=vfunc, vf_reg=vf_reg)

ctl_mstep_kwargs = {'nb_iter': 5}
hbreps.run(nb_iter=15, nb_train_samples=5000,
           nb_eval_rollouts=25, nb_eval_steps=250,
           ctl_mstep_kwargs=ctl_mstep_kwargs)

rollouts, _ = hbreps.evaluate(nb_rollouts=25, nb_steps=250)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=hbreps.state_dim + hbreps.act_dim, figsize=(12, 4))
for roll in rollouts[10:]:
    for k, col in enumerate(ax[:-1]):
        col.plot(roll['x'][:, k])
    ax[-1].plot(roll['uc'])
plt.show()
