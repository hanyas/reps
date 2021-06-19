import os
import torch

from gym.envs.registration import register

register(
    id='LQR-RL-v0',
    entry_point='reps.envs:LQR',
    max_episode_steps=1000,
)

register(
    id='Pendulum-RL-v0',
    entry_point='reps.envs:Pendulum',
    max_episode_steps=1000,
)

register(
    id='Pendulum-RL-v1',
    entry_point='reps.envs:PendulumWithCartesianObservation',
    max_episode_steps=1000,
)

register(
    id='Pole-RL-v0',
    entry_point='reps.envs:PoleWithWall',
    max_episode_steps=1000,
)

register(
    id='Cartpole-RL-v0',
    entry_point='reps.envs:Cartpole',
    max_episode_steps=1000,
)

register(
    id='Cartpole-RL-v1',
    entry_point='reps.envs:CartpoleWithCartesianObservation',
    max_episode_steps=1000,
)

try:
    register(
        id='HybridPole-RL-v0',
        entry_point='reps.envs:HybridPoleWithWall',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/control/hybrid/models/rarhmm_pole.pkl', 'rb'),
                                     map_location='cpu')}
    )
except :
    pass

try:
    register(
        id='HybridPendulum-RL-v1',
        entry_point='reps.envs:HybridPendulumWithCartesianObservation',
        max_episode_steps=1000,
        kwargs={'rarhmm': torch.load(open(os.path.dirname(__file__)
                                          + '/envs/control/hybrid/models/rarhmm_pendulum_cart.pkl', 'rb'),
                                     map_location='cpu')}
    )
except:
    pass
