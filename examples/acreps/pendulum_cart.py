import numpy as np
import gym

from reps.acreps import acREPS

np.random.seed(1337)

env = gym.make('Pendulum-RL-v1')
env._max_episode_steps = 250
env.unwrapped.dt = 0.05
env.unwrapped.sigma = 1e-4
# env.seed(1337)

acreps = acREPS(env=env, kl_bound=0.1, discount=0.985, lmbda=0.95,
                scale=[1., 1., 8.0, 2.5], mult=0.5,
                nb_vfeat=75, nb_pfeat=75, vf_reg=1e-12)

acreps.run(nb_iter=15, nb_train_samples=5000,
           nb_eval_rollouts=25, nb_eval_steps=100)

# evaluate
rollouts, _ = acreps.evaluate(nb_rollouts=25, nb_steps=100)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=acreps.state_dim + acreps.act_dim, figsize=(12, 4))
for roll in rollouts:
    for k, col in enumerate(ax[:-1]):
        col.plot(roll['x'][:, k])
    ax[-1].plot(roll['uc'])
plt.show()
