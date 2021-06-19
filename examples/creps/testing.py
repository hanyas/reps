#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename: testing
# @Date: 2019-07-08-21-45
# @Author: Hany Abdulsamad
# @Contact: hany@robot-learning.de

from rl.creps import cREPS
from rl.envs import CSphere


creps = cREPS(func=CSphere(context_dim=3, act_dim=3),
              n_episodes=100,
              kl_bound=0.1,
              vdgr=2, pdgr=1,
              vreg=1e-16, preg=1e-16,
              cov0=100.0)

creps.run(nb_iter=250, verbose=True)
