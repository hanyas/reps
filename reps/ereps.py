import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize
from scipy import stats

import copy

EXP_MAX = 700.0
EXP_MIN = -700.0


class Policy:

    def __init__(self, act_dim, cov0):
        self.act_dim = act_dim

        self.mu = np.random.randn(self.act_dim)
        self.cov = cov0 * np.eye(self.act_dim)

    def action(self, n):
        u = sc.stats.multivariate_normal(mean=self.mu, cov=self.cov).rvs(n)
        return u.reshape((n, self.act_dim))

    @staticmethod
    def log_likelihood(pi, x):
        mu, cov = pi.mu, pi.cov
        dim =  mu.shape[0]
        diff = mu - x
        loglik = - 0.5 * (np.einsum('nk,kh,nh->n', diff, np.linalg.inv(cov), diff) +
                          np.log(np.linalg.det(cov))
                          + dim * np.log(2.0 * np.pi))
        return loglik

    def kli(self, pi):
        diff = self.mu - pi.mu
        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov)
                    + diff.T @ np.linalg.inv(self.cov) @ diff
                    + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov))
                    - self.act_dim)
        return kl

    def klm(self, pi):
        diff = pi.mu - self.mu
        kl = 0.5 * (np.trace(np.linalg.inv(pi.cov) @ self.cov)
                    + diff.T @ np.linalg.inv(pi.cov) @ diff
                    + np.log(np.linalg.det(pi.cov) / np.linalg.det(self.cov))
                    - self.act_dim)
        return kl

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.cov * 2.0 * np.pi * np.exp(1.0)))

    def wml(self, x, w, eta=np.array([0.0])):
        pol = copy.deepcopy(self)

        pol.mu = (np.sum(w[:, np.newaxis] * x, axis=0) + eta * self.mu) / (np.sum(w, axis=0) + eta)

        diff = x - pol.mu
        tmp = np.einsum('nk,n,nh->nkh', diff, w, diff)
        pol.cov = (np.sum(tmp, axis=0) + eta * self.cov +
                   eta * np.outer(pol.mu - self.mu, pol.mu - self.mu)) / (np.sum(w, axis=0) + eta)
        return pol

    def dual(self, eta, x, w, eps):
        pol = self.wml(x, w, eta)
        return np.sum(w * self.log_likelihood(pol, x)) + eta * (eps - self.klm(pol))

    def wmap(self, x, w, eps=np.array([0.1])):
        res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                   method='SLSQP',
                                   jac=grad(self.dual),
                                   args=(x, w, eps),
                                   bounds=((1e-8, 1e8),))
        eta = res['x']
        pol = self.wml(x, w, eta)
        return pol


class eREPS:

    def __init__(self, func, nb_episodes,
                 kl_bound, **kwargs):

        self.func = func
        self.act_dim = self.func.dim

        self.nb_episodes = nb_episodes
        self.kl_bound = kl_bound

        cov0 = kwargs.get('cov0', 100.0)
        self.ctl = Policy(self.act_dim, cov0)

        self.data = None
        self.w = None
        self.eta = np.array([1.0])

    def sample(self, nb_episodes):
        data = {'x': self.ctl.action(nb_episodes)}
        data['r'] = self.func.eval(data['x'])
        return data

    @staticmethod
    def weights(r, eta, normalize=True):
        adv = r - np.max(r) if normalize else r
        w = np.exp(np.clip(adv / eta, EXP_MIN, EXP_MAX))
        return w, adv

    def dual(self, eta, eps, r):
        w, _ = self.weights(r, eta)
        g = eta * eps + np.max(r) + eta * np.log(np.mean(w, axis=0))
        return g

    def grad(self, eta, eps, r):
        w, adv = self.weights(r, eta)
        dg = eps + np.log(np.mean(w, axis=0)) - \
            np.sum(w * adv, axis=0) / (eta * np.sum(w, axis=0))
        return dg

    @staticmethod
    def sample_kl(w):
        w = np.clip(w, 1e-75, np.inf)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def run(self, nb_iter=100, verbose=False):
        trace = {'rwrd': [],
                'kls': [], 'kli': [], 'klm': [],
                'ent': []}

        for it in range(nb_iter):
            self.data = self.sample(self.nb_episodes)
            rwrd = np.mean(self.data['r'])

            res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                       method='SLSQP', jac=self.grad,
                                       args=(self.kl_bound, self.data['r']),
                                       bounds=((1e-18, 1e18),))

            self.eta = res.x
            self.w, _ = self.weights(self.data['r'], self.eta,
                                     normalize=False)

            # pol = self.ctl.wml(self.data['x'], self.w)
            pol = self.ctl.max_aposteriori(self.data['x'], self.w, eps=self.kl_bound)

            kls = self.sample_kl(self.w)
            kli = self.ctl.kli(pol)
            klm = self.ctl.klm(pol)

            self.ctl = pol
            ent = self.ctl.entropy()

            trace['rwrd'].append(rwrd)
            trace['kls'].append(kls)
            trace['kli'].append(kli)
            trace['klm'].append(klm)
            trace['ent'].append(ent)

            if verbose:
                print('it=', it,
                      f'rwrd={rwrd:{5}.{4}}',
                      f'kls={kls:{5}.{4}}',
                      f'kli={kli:{5}.{4}}',
                      f'klm={klm:{5}.{4}}',
                      f'ent={ent:{5}.{4}}')

            if ent < -3e2:
                break

        return trace
