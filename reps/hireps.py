import autograd.numpy as np
import autograd.numpy.random as npr
import torch
from autograd.numpy.random import multivariate_normal as mvn
from autograd import grad, jacobian, hessian

import scipy as sc
from scipy import optimize
from scipy import stats

import random

import multiprocessing
nb_cores = multiprocessing.cpu_count()

EXP_MAX = 700.0
EXP_MIN = -700.0


def multinomial_rvs(n, p):
    """
    Sample from the multinomial distribution with multiple p vectors.

    * n must be a scalar.
    * p must an n-dimensional numpy array, n >= 1.  The last axis of p
      holds the sequence of probabilities for a multinomial distribution.

    The return value has the same shape as p.
    """
    count = np.full(p.shape[:-1], n)
    out = np.zeros(p.shape, dtype=int)
    ps = p.cumsum(axis=-1)
    # Conditional probabilities
    with np.errstate(divide='ignore', invalid='ignore'):
        condp = p / ps
    condp[np.isnan(condp)] = 0.0
    for i in range(p.shape[-1]-1, 0, -1):
        binsample = np.random.binomial(count, condp[..., i])
        out[..., i] = binsample
        count -= binsample
    out[..., 0] = count
    return out


def one_hot(z, K):
    z = np.atleast_1d(z).astype(int)
    assert np.all(z >= 0) and np.all(z < K)
    shp = z.shape
    N = z.size
    zoh = np.zeros((N, K))
    zoh[np.arange(N), np.arange(K)[np.ravel(z)]] = 1
    zoh = np.reshape(zoh, shp + (K,))
    return zoh


def merge(*dicts):
    d = {}
    for dc in dicts:
        for key in dc:
            try:
                d[key].append(dc[key])
            except KeyError:
                d[key] = [dc[key]]

    for key in d:
        d[key] = np.concatenate(d[key])
    return d


class FourierFeatures:

    def __init__(self, dim, nb_feat, scale, mult, with_offset=True):
        self.dim = dim
        self.with_offset = with_offset
        self.nb_feat = nb_feat - 1 if with_offset else nb_feat

        self.norm = np.sqrt(2) / np.sqrt(self.nb_feat)

        # We sample frequencies from a rescaled normal,
        # which is equivalent to sampling frequencies
        # from a N(0, 1) while standardzing the input.
        self.sqdist = mult * np.array(scale)**2
        self.freq = mvn(mean=np.zeros(self.dim),
                        cov=np.diag(1. / self.sqdist),
                        size=self.nb_feat)
        self.shift = npr.uniform(0., 2. * np.pi, size=self.nb_feat)

    def fit_transform(self, x):
        phi = self.norm * np.cos(np.einsum('kd,...d->...k', self.freq, x) + self.shift)
        if self.with_offset:
            ones = np.ones((x.shape[0],))
            return np.column_stack((phi, ones))
        else:
            return phi


class Vfunction:

    def __init__(self, state_dim, nb_feat, scale, mult):
        self.state_dim = state_dim

        self.nb_feat = nb_feat
        self.scale = scale
        self.mult = mult
        self.basis = FourierFeatures(self.state_dim, self.nb_feat,
                                     self.scale, self.mult)

        self.omega = npr.uniform(size=(self.nb_feat,))

    def features(self, x):
        return self.basis.fit_transform(x)

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class hbREPS:

    def __init__(self, env, dyn, ctl, kl_bound, discount,
                 scale, mult, nb_vfeat, vf_reg,
                 ctl_kwargs={}):

        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.kl_bound = kl_bound
        self.discount = discount

        self.nb_vfeat = nb_vfeat

        self.scale = scale
        self.mult = mult
        self.vf_reg = vf_reg

        self.vfunc = Vfunction(self.state_dim, nb_feat=self.nb_vfeat,
                               scale=self.scale, mult=self.mult)

        self.nb_modes = dyn.nb_states
        self.dyn = dyn
        self.ctl = ctl

        self.ulim = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.eta = np.array([1.0])

    def sample(self, nb_samples, buffer_size=0,
               reset=True, stoch=True, average=True,
               env=None, discount=0.):

        if len(self.rollouts) >= buffer_size:
            rollouts = random.sample(self.rollouts, buffer_size)
        else:
            rollouts = []

        coin = sc.stats.binom(1, 1.0 - discount)

        n = 0
        while True:
            roll = {'xi': np.empty((0, self.state_dim)),
                    'x': np.empty((0, self.state_dim)),
                    'u': np.empty((0, self.act_dim)),
                    'uc': np.empty((0, self.act_dim)),
                    'xn': np.empty((0, self.state_dim)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            x = env.reset()

            roll['xi'] = np.vstack((roll['xi'], x))
            roll['done'] = np.hstack((roll['done'], False))

            done = False
            while not done:
                if reset and coin.rvs():
                    done = True
                    roll['done'][-1] = done
                else:
                    roll['x'] = np.vstack((roll['x'], x))
                    u = self.ctl.action(roll['x'], roll['uc'],
                                        stoch, average)[-1]

                    uc = np.clip(u, -self.ulim, self.ulim)
                    roll['u'] = np.vstack((roll['u'], u))
                    roll['uc'] = np.vstack((roll['uc'], uc))

                    x, r, done, _ = env.step(u)
                    roll['r'] = np.hstack((roll['r'], r))
                    roll['done'] = np.hstack((roll['done'], done))

                    roll['xn'] = np.vstack((roll['xn'], x))

                    n = n + 1
                    if n >= nb_samples:
                        if len(roll['x']) > 0:
                            roll['done'][-1] = True
                            rollouts.append(roll)
                        data = merge(*rollouts)
                        return rollouts, data

            if len(roll['x']) > 0:
                rollouts.append(roll)

    def evaluate(self, nb_rollouts, nb_steps, stoch=False, average=False):
        rollouts = []

        for n in range(nb_rollouts):
            roll = {'x': np.empty((0, self.state_dim)),
                    'u': np.empty((0, self.act_dim)),
                    'uc': np.empty((0, self.act_dim)),
                    'r': np.empty((0,))}

            x = self.env.reset()

            for t in range(nb_steps):
                roll['x'] = np.vstack((roll['x'], x))
                u = self.ctl.action(roll['x'], roll['uc'],
                                    stoch, average)[-1]

                uc = np.clip(u, -self.ulim, self.ulim)
                roll['u'] = np.vstack((roll['u'], u))
                roll['uc'] = np.vstack((roll['uc'], uc))

                x, r, done, _ = self.env.step(u)
                roll['r'] = np.hstack((roll['r'], r))

            rollouts.append(roll)

        data = merge(*rollouts)
        return rollouts, data

    def featurize(self, data):
        ivfeat = self.vfunc.features(data['xi'])
        vfeat = self.vfunc.features(data['x'])
        nvfeat = self.vfunc.features(data['xn'])
        return ivfeat, vfeat, nvfeat

    @staticmethod
    def weights(eta, omega, iphi, phi, nphi, gamma, rwrd, normalize=True):
        ival = np.mean(np.einsum('nd,d->n', iphi, omega), axis=0, keepdims=True)
        val = np.einsum('nd,d->n', phi, omega)
        nval = np.einsum('nd,d->n', nphi, omega)

        adv = rwrd + gamma * nval - val + (1. - gamma) * ival
        delta = adv - np.max(adv) if normalize else adv
        weights = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        return weights, delta, np.max(adv)

    def dual(self, var, epsilon, iphi, phi, nphi, gamma, rwrd):
        eta, omega = var[0], var[1:]
        weights, _, max_adv = self.weights(eta, omega, iphi, phi, nphi, gamma, rwrd)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(weights, axis=0))
        g += self.vf_reg * np.sum(omega ** 2)
        return g

    def dual_eta(self, eta, omega, epsilon, iphi, phi, nphi, gamma, rwrd):
        weights, _, max_adv = self.weights(eta, omega, iphi, phi, nphi, gamma, rwrd)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(weights, axis=0))
        return g

    def dual_omega(self, omega, eta, epsilon, iphi, phi, nphi, gamma, rwrd):
        weights, _, max_adv = self.weights(eta, omega, iphi, phi, nphi, gamma, rwrd)
        g = max_adv + eta * np.log(np.mean(weights, axis=0))
        g = g + self.vf_reg * np.sum(omega ** 2)
        return g

    @staticmethod
    def samples_kl(weights):
        weights = np.clip(weights, 1e-75, np.inf)
        weights = weights / np.mean(weights, axis=0)
        return np.mean(weights * np.log(weights), axis=0)

    def run(self, nb_iter=10, nb_train_samples=5000, buffer_size=0,
            nb_eval_rollouts=25, nb_eval_steps=250, verbose=True,
            ctl_mstep_kwargs={}, iterative=True,
            sim_env=None, nb_sim_samples=1000):

        trace = {'rwrd': [], 'kls': []}

        for it in range(nb_iter):
            _, eval = self.evaluate(nb_eval_rollouts, nb_eval_steps)

            self.rollouts, self.data = self.sample(nb_train_samples, buffer_size,
                                                   env=self.env, discount=self.discount)
            if sim_env is not None:
                self.rollouts, self.data = self.sample(nb_sim_samples, buffer_size=len(self.rollouts),
                                                       env=sim_env, discount=self.discount)

            ivfeat, vfeat, nvfeat = self.featurize(self.data)

            if not iterative:
                from warnings import filterwarnings
                filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
                res = sc.optimize.minimize(self.dual, np.hstack((1e6, self.vfunc.omega)),
                                           method='trust-constr', jac=grad(self.dual),
                                           args=(self.kl_bound, ivfeat, vfeat, nvfeat,
                                                 self.discount, self.data['r']),
                                           bounds=((1e-16, 1e16),) + ((-np.inf, np.inf),) * self.nb_vfeat)

                self.eta, self.vfunc.omega = res.x[0], res.x[1:]
            else:
                self.eta = np.array([1e6])
                # self.vfunc.omega = npr.uniform(size=(self.nb_vfeat,))
                for _ in range(10):
                    res = sc.optimize.minimize(self.dual_omega, self.vfunc.omega,
                                               method='L-BFGS-B', jac=grad(self.dual_omega),
                                               args=(self.eta, self.kl_bound,
                                                     ivfeat, vfeat, nvfeat,
                                                     self.discount, self.data['r']))

                    self.vfunc.omega = res.x

                    res = sc.optimize.minimize(self.dual_eta, self.eta,
                                               method='L-BFGS-B', jac=grad(self.dual_eta),
                                               args=(self.vfunc.omega, self.kl_bound,
                                                     ivfeat, vfeat, nvfeat,
                                                     self.discount, self.data['r']),
                                               bounds=((1e-16, 1e16),))

                    self.eta = res.x

            weights = self.weights(self.eta, self.vfunc.omega,
                                   ivfeat, vfeat, nvfeat,
                                   self.discount, self.data['r'],
                                   normalize=False)[0]

            kls = self.samples_kl(weights)

            # policy update
            ts = [roll['x'].shape[0] for roll in self.rollouts]
            w = np.split(weights[:, np.newaxis], np.cumsum(ts)[:-1])

            x = [roll['x'] for roll in self.rollouts]
            u = [roll['u'] for roll in self.rollouts]
            uc = [roll['uc'] for roll in self.rollouts]

            p = self.dyn.estep(x, uc)[0]
            # p = [one_hot(np.argmax(_p, axis=1), self.nb_modes) for _p in p]
            # p = [multinomial_rvs(1, _p) for _p in p]

            self.ctl.weighted_mstep(p, x, u, w, ctl_mstep_kwargs)

            # rwrd = np.mean(self.data['r'])
            rwrd = np.mean(eval['r'])
            trace['rwrd'].append(rwrd)
            trace['kls'].append(kls)

            if verbose:
                print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kls={kls:{5}.{4}}')

        return trace
