import gym
from gym import spaces
from gym.utils import seeding

import numpy as np


def normalize(x):
    # wraps angle between [-pi, pi]
    return ((x + np.pi) % (2. * np.pi)) - np.pi


class Cartpole(gym.Env):

    def __init__(self):
        self.state_dim = 4
        self.act_dim = 1
        self.obs_dim = 4

        self.dt = 0.01

        self.sigma = 1e-8

        # g = [x, th, dx, dth]
        self.g = np.array([0., 0., 0., 0.])
        self.gw = - np.array([1e0, 2e0, 1e-1, 1e-1])

        # x = [x, th, dx, dth]
        self.xmax = np.array([5., np.inf, np.inf, np.inf])
        self.state_space = spaces.Box(low=-self.xmax,
                                      high=self.xmax,
                                      dtype=np.float64)

        # y = [x, th, dx, dth]
        self.ymax = np.array([5., np.inf, np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.ymax,
                                            high=self.ymax,
                                            dtype=np.float64)

        self.uw = - 1e-3 * np.ones((self.act_dim, ))
        self.umax = 5.0 * np.ones((self.act_dim, ))
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

        # Equations: http://coneural.org/florian/papers/05_cart_pole.pdf
        # x = [x, th, dx, dth]
        g = 9.81
        Mc = 0.37
        Mp = 0.127
        Mt = Mc + Mp
        l = 0.3365
        fr = 0.005

        def f(x, u):
            q, th, dq, dth = x

            sth = np.sin(th)
            cth = np.cos(th)

            # This friction model is not exactly right
            # It neglects the influence of the pole
            num = g * sth + cth * (- (u - fr * dq) - Mp * l * dth**2 * sth) / Mt
            denom = l * ((4. / 3.) - Mp * cth**2 / Mt)
            ddth = num / denom

            ddx = (u + Mp * l * (dth**2 * sth - ddth * cth)) / Mt
            return np.hstack((dq, dth, ddx, ddth))

        c1 = f(x, uc)
        c2 = f(x + 0.5 * self.dt * c1, uc)
        c3 = f(x + 0.5 * self.dt * c2, uc)
        c4 = f(x + self.dt * c3, uc)

        xn = x + self.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)
        xn = np.clip(xn, -self.xlim, self.xlim)

        return xn

    def observe(self, x):
        return np.array([x[0], normalize(x[1]), x[2], x[3]])

    def noise(self, x=None, u=None):
        _u = np.clip(u, -self.ulim, self.ulim)
        _x = np.clip(x, -self.xlim, self.xlim)
        return self.sigma * np.eye(self.obs_dim)

    def rewrad(self, x, u):
        _x = np.array([x[0], normalize(x[1]), x[2], x[3]])
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
            low = np.array([-0.1, -np.pi, -5.0, -10.0])
            high = np.array([0.1, np.pi, 5.0, 10.0])
        else:
            low, high = np.array([0., np.pi - np.pi / 18., 0., -1.0]),\
                        np.array([0., np.pi + np.pi / 18., 0., 1.0])

        self.state = self.np_random.uniform(low=low, high=high)
        return self.observe(self.state)

    # for plotting
    def fake_step(self, x, u):
        xn = self.dynamics(x, u)
        return self.observe(xn)


class CartpoleWithCartesianObservation(Cartpole):

    def __init__(self):
        super(CartpoleWithCartesianObservation, self).__init__()
        self.obs_dim = 5

        # y = [x, cos, sin, xd, thd]
        self.ymax = np.array([5., 1., 1., np.inf, np.inf])
        self.observation_space = spaces.Box(low=-self.ymax,
                                            high=self.ymax,
                                            dtype=np.float64)

    def observe(self, x):
        return np.array([x[0],
                         np.cos(x[1]),
                         np.sin(x[1]),
                         x[2],
                         x[3]])
