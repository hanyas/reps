import numpy as np
import gym

from reps.reps import REPS

np.random.seed(1337)

env = gym.make('Pole-RL-v0')
env._max_episode_steps = 5000
env.unwrapped.dt = 0.01
env.unwrapped.sigma = 1e-8
# env.seed(1337)

reps = REPS(env=env, kl_bound=0.25, discount=0.98,
            scale=[0.25, 1.5], mult=0.5,
            nb_pfeat=25, nb_vfeat=25, vf_reg=1e-8)

reps.run(nb_iter=10, nb_train_samples=2500,
         nb_eval_rollouts=25, nb_eval_steps=100)

# evaluate deterministic policy
rollouts, _ = reps.evaluate(nb_rollouts=25, nb_steps=100)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=reps.state_dim + reps.act_dim, figsize=(12, 4))
for roll in rollouts:
    for k, col in enumerate(ax[:-1]):
        col.plot(roll['x'][:, k])
    ax[-1].plot(roll['uc'])
plt.show()
