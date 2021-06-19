import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


def normalize(x):
    # wraps angle between [-pi, pi]
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Pendulum(gym.Env):

    def __init__(self):
        self.state_dim = 2
        self.act_dim = 1
        self.obs_dim = 2

        self.dt = 0.01

        self.sigma = 1e-8

        # g = [th, thd]
        self.g = np.array([0., 0.])
        self.gw = - np.array([1e0, 1e-1])

        # x = [th, thd]
        self.xmax = np.array([np.inf, np.inf])
        self.state_space = spaces.Box(low=-self.xmax,
                                      high=self.xmax,
                                      dtype=np.float64)

        # y = [th, thd]
        self.ymax = np.array([np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.ymax,
                                            high=self.ymax,
                                            dtype=np.float64)

        self.uw = - 1e-3 * np.ones((self.act_dim, ))
        self.umax = 2.5 * np.ones((self.act_dim, ))
        self.action_space = spaces.Box(low=-self.umax,
                                       high=self.umax, shape=(1,),
                                       dtype=np.float64)

        self.uniform = True

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
        uc = np.clip(u, -self.ulim, self.ulim)

        g, m, l, k = 9.81, 1., 1., 1e-3

        def f(x, u):
            th, dth = x
            return np.hstack((dth, - 3. * g / (2. * l) * np.sin(th + np.pi) +
                              3. / (m * l ** 2) * (u - k * dth)))
        c1 = f(x, uc)
        c2 = f(x + 0.5 * self.dt * c1, uc)
        c3 = f(x + 0.5 * self.dt * c2, uc)
        c4 = f(x + self.dt * c3, uc)

        xn = x + self.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)

        return xn

    def observe(self, x):
        return np.array([normalize(x[0]), x[1]])

    def noise(self, x=None, u=None):
        _u = np.clip(u, -self.ulim, self.ulim)
        _x = np.clip(x, -self.xlim, self.xlim)
        return self.sigma * np.eye(self.obs_dim)

    def rewrad(self, x, u):
        _x = np.array([normalize(x[0]), x[1]])
        return (_x - self.g).T @ np.diag(self.gw) @ (_x - self.g)\
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
        if self.uniform:
            low = np.array([-np.pi, -8.0])
            high = np.array([np.pi, 8.0])
        else:
            low, high = np.array([np.pi - np.pi / 18., -1.0]), \
                        np.array([np.pi + np.pi / 18., 1.0])

        self.state = self.np_random.uniform(low=low, high=high)
        return self.observe(self.state)

    # for plotting
    def fake_step(self, x, u):
        xn = self.dynamics(x, u)
        return xn


class PendulumWithCartesianObservation(Pendulum):

    def __init__(self):
        super(PendulumWithCartesianObservation, self).__init__()
        self.obs_dim = 3
        self.sigma = 1e-8

        # y = [cos, sin, thd]
        self.ymax = np.array([1., 1., np.inf])
        self.observation_space = spaces.Box(low=-self.ymax,
                                            high=self.ymax,
                                            dtype=np.float64)

    def observe(self, x):
        return np.array([np.cos(x[0]),
                         np.sin(x[0]),
                         x[1]])
