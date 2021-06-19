import numpy as np
import gym

from reps.reps import REPS

np.random.seed(1337)

env = gym.make('Pendulum-RL-v1')
env._max_episode_steps = 5000
env.unwrapped.dt = 0.02
env.unwrapped.sigma = 1e-4
env.seed(1337)

reps = REPS(env=env, kl_bound=0.1, discount=0.985,
            scale=[0.5, 0.5, 4.0], mult=1.,
            nb_pfeat=75, nb_vfeat=75, vf_reg=1e-12)

reps.run(nb_iter=10, nb_train_samples=5000,
         nb_eval_rollouts=25, nb_eval_steps=250,
         iterative=False)

# evaluate deterministic policy
rollouts, _ = reps.evaluate(nb_rollouts=25, nb_steps=250)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=reps.state_dim + reps.act_dim, figsize=(12, 4))
for roll in rollouts:
    for k, col in enumerate(ax[:-1]):
        col.plot(roll['x'][:, k])
    ax[-1].plot(roll['uc'])
plt.show()
