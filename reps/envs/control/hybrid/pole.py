import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


class PoleWithWall(gym.Env):

    def __init__(self):
        self.state_dim = 2
        self.act_dim = 1

        self.dt = 0.01
        self.sigma = 1e-8

        self.g = np.array([0., 0.])
        self.gw = - np.array([1e2, 1e1])

        self.xmax = np.array([0.25, 1.5])
        self.xmin = np.array([-0.25, -1.5])
        self.observation_space = spaces.Box(low=self.xmin,
                                            high=self.xmax,
                                            dtype=np.float64)

        self.uw = - np.array([1e-1])
        self.umax, self.umin = np.array([4.]), np.array([-4.])
        self.action_space = spaces.Box(low=self.umin,
                                       high=self.umax,
                                       dtype=np.float64)

        self.uniform = True

        self.state = None
        self.np_random = None

        self.seed()

    def dynamics(self, x, u):
        uc = np.clip(u, self.umin, self.umax)

        m, l, g = 1., 1., 10.
        k, d, h = 100., 0.1, 0.01
        wall = d / l

        p, v = x
        if p <= wall:
            A = np.array([[0.,    1.],
                          [g / l, 0.]])
            B = np.array([0., 1./(m * l**2)])
            c = np.array([0., 0.])
        else:
            A = np.array([[0.,             1.],
                          [g / l - k / m,  0.]])
            B = np.array([0., 1. / (m * l ** 2)])
            c = np.array([0., k * d / (m * l)])

        xn = x + self.dt * (A @ x + B * u + c)
        return xn

    def noise(self, x=None, u=None):
        return self.sigma * np.eye(self.state_dim)

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
        obs = self.np_random.multivariate_normal(self.state, sigma)
        return obs, rwrd, False, {}

    def reset(self):
        if self.uniform:
            low = np.array([-0.1, -0.5])
            high = np.array([0.1, 0.5])
            self.state = self.np_random.uniform(low=low, high=high)
        else:
            self.state = np.array([0., 0.8])
        return self.state

    # for plotting
    def fake_step(self, x, u):
        xn = self.dynamics(x, u)
        return xn
