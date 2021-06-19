import numpy as np


class CSphere:

    def __init__(self, cntxt_dim, act_dim):
        self.cntxt_dim = cntxt_dim
        self.act_dim = act_dim

        M = np.random.randn(self.act_dim, self.act_dim)
        M = 0.5 * (M + M.T)
        self.Q = M @ M.T

    def context(self, nb_episodes):
        return np.random.uniform(-1.0, 1.0, size=(nb_episodes, self.cntxt_dim))

    def eval(self, x, c):
        diff = x - c
        return - np.einsum('nk,kh,nh->n', diff, self.Q, diff)
