import numpy as np
import numpy.random as npr

from sklearn.preprocessing import PolynomialFeatures


class Himmelblau:
    def __init__(self, dim=2):
        self.dim = dim

    @staticmethod
    def eval(self, x):
        a = x[:, 0] * x[:, 0] + x[:, 1] - 11.0
        b = x[:, 0] + x[:, 1] * x[:, 1] - 7.0
        return -1.0 * (a * a + b * b)


class Sphere:

    def __init__(self, dim):
        self.dim = dim

        M = npr.randn(dim, dim)
        A = M @ M.T
        Q = A[np.nonzero(np.triu(A))]

        q = 0.0 * npr.rand(dim)
        q0 = 0.0 * npr.rand()

        self.param = np.hstack((q0, q, Q))
        self.basis = PolynomialFeatures(degree=2)

    def eval(self, x):
        feat = self.basis.fit_transform(x)
        return - np.dot(feat, self.param)


class Rosenbrock:

    def __init__(self, dim):
        self.dim = dim

    @staticmethod
    def eval(self, x):
        return - np.sum(100.0 * (x[:, 1:] - x[:, :-1] ** 2.0) ** 2.0 +
                        (1 - x[:, :-1]) ** 2.0, axis=-1)


class Styblinski:
    def __init__(self, dim):
        self.dim = dim

    @staticmethod
    def eval(self, x):
        return - 0.5 * np.sum(x**4.0 - 16.0 * x**2 + 5 * x, axis=-1)


class Rastrigin:
    def __init__(self, dim):
        self.dim = dim

    def eval(self, x):
        return - (10.0 * self.dim +
                  np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x), axis=-1))
