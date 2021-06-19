import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class BouncingBall(gym.Env):

    def __init__(self):
        self.state_dim = 2
        self.act_dim = 1
        self.obs_dim = 2

        self.dt = 0.01

        self.sigma = 1e-8

        # x = [x, xd]
        self.xmax = np.array([np.inf, np.inf])
        self.state_space = spaces.Box(low=-self.xmax,
                                      high=self.xmax,
                                      dtype=np.float64)

        self.ymax = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.ymax,
                                            high=self.ymax,
                                            dtype=np.float64)

        self.umax = 0.
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax, shape=(1,),
                                       dtype=np.float64)

        self.state = None
        self.np_random = None

        self.seed()

    @property
    def xlim(self):
        return self.xmax

    @property
    def ulim(self):
        return self.umax

    def dynamics(self, x, u):
        k, g = 0.8, 9.81

        def f(x, u):
            h, dh = x
            return np.array([dh, -g])

        c1 = f(x, u)
        c2 = f(x + 0.5 * self.dt * c1, u)
        c3 = f(x + 0.5 * self.dt * c2, u)
        c4 = f(x + self.dt * c3, u)

        xn = x + self.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)

        if xn[0] <= 0. and xn[1] < 0.:
            xn[0] = 0.
            xn[1] = - k * xn[1]

        return xn

    def observe(self, x):
        return x

    def noise(self, x=None, u=None):
        return self.sigma * np.eye(self.obs_dim)

    def rewrad(self, x, u):
        return NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        self.state = self.dynamics(self.state, u)
        rwrd = self.rewrad(self.state, u)
        sigma = self.noise(self.state, u)
        obs = self.np_random.multivariate_normal(self.observe(self.state), sigma)
        return obs, rwrd, False, {}

    def reset(self):
        low = np.array([8.0, 1.0])
        high = np.array([10.0, 2.0])
        self.state = self.np_random.uniform(low=low, high=high)
        return self.observe(self.state)

    # for plotting
    def fake_step(self, x, u):
        xn = self.dynamics(x, u)
        return self.observe(xn)
