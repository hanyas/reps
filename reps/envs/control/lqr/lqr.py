import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class LQR(gym.Env):

    def __init__(self):
        self.state_dim = 2
        self.act_dim = 1
        self.obs_dim = 2

        self.dt = 0.01

        self.sigma = 1e-8

        self.g = np.array([0., 0.])
        self.gw = - np.array([1.e2, 1.e1])

        self.xmax = np.array([1., 1.])
        self.observation_space = spaces.Box(low=-self.xmax,
                                            high=self.xmax,
                                            dtype=np.float64)

        self.uw = - 1e-1 * np.ones((self.act_dim, ))
        self.umax = np.array([np.inf])
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax,
                                       dtype=np.float64)

        self.A = np.array([[0., 1.], [0., 0.]])
        self.B = np.array([[0., 1.]])
        self.c = np.zeros((2, ))

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
        def f(x, u):
            return np.einsum('kh,h->k', self.A, x)\
                   + np.einsum('kh,h->k', self.B, u)\
                   + self.c

        k1 = f(x, u)
        k2 = f(x + 0.5 * self.dt * k1, u)
        k3 = f(x + 0.5 * self.dt * k2, u)
        k4 = f(x + self.dt * k3, u)

        xn = x + self.dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)

        return xn

    def observe(self, x):
        return x

    def noise(self, x=None, u=None):
        return self.sigma * np.eye(self.obs_dim)

    def rewrad(self, x, u):
        return (x - self.g).T @ np.diag(self.gw) @ (x - self.g)\
               + u.T @ np.diag(self.uw) @ u

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
        low = np.array([-1.0, -1e-2])
        high = np.array([1.0, 1e-2])
        self.state = self.np_random.uniform(low=low, high=high)
        return self.state
