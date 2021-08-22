import autograd.numpy as np
import autograd.numpy.random as npr

from autograd.numpy.random import multivariate_normal as mvn
from autograd import grad, jacobian

import scipy as sc
from scipy.optimize import minimize
from scipy import stats
from scipy.special import comb

from sklearn.preprocessing import PolynomialFeatures

import random


EXP_MAX = 700.0
EXP_MIN = -700.0


def one_hot(K, z):
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


class RFFVfunction:

    def __init__(self, nb_modes, state_dim, nb_feat, scale, mult):
        self.nb_modes = nb_modes
        self.state_dim = state_dim

        self.nb_feat = nb_feat
        self.scale = scale
        self.mult = mult
        self.basis = FourierFeatures(self.state_dim, self.nb_feat,
                                     self.scale, self.mult)

        self.omega = npr.uniform(size=(self.nb_feat * self.nb_modes, ))

    def features(self, x):
        phi = np.zeros((x.shape[0], self.nb_feat, self.nb_modes))
        for k in range(self.nb_modes):
            phi[:, :,  k] = self.basis.fit_transform(x)
        return phi


class PolyVfunction:

    def __init__(self, nb_modes, state_dim, degree):
        self.nb_modes = nb_modes
        self.state_dim = state_dim
        self.degree = degree

        self.nb_feat = int(comb(self.degree + self.state_dim, self.degree))
        self.basis = PolynomialFeatures(self.degree)
        self.omega = npr.uniform(size=(self.nb_feat * self.nb_modes,))

    def features(self, x):
        phi = np.zeros((x.shape[0], self.nb_feat, self.nb_modes))
        for k in range(self.nb_modes):
            phi[:, :,  k] = self.basis.fit_transform(x)
        return phi


# Hybrid REPS with
# a hybrid value function
class hbREPS_hbVf:

    def __init__(self, env, dyn, ctl, kl_bound, discount,
                 vfunc, vf_reg, ctl_kwargs={}):

        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.kl_bound = kl_bound
        self.discount = discount

        self.nb_modes = dyn.nb_states
        self.dyn = dyn
        self.ctl = ctl

        self.vfunc = vfunc
        self.vf_reg = vf_reg
        self.nb_vfeat = len(self.vfunc.omega)

        self.ulim = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.eta = np.array([1.0])

    def sample(self, nb_samples, buffer_size=0,
               reset=True, stoch=True, average=True):

        if len(self.rollouts) >= buffer_size:
            rollouts = random.sample(self.rollouts, buffer_size)
        else:
            rollouts = []

        coin = sc.stats.binom(1, 1.0 - self.discount)

        n = 0
        while True:
            roll = {'xi': np.empty((0, self.state_dim)),
                    'x': np.empty((0, self.state_dim)),
                    'u': np.empty((0, self.act_dim)),
                    'uc': np.empty((0, self.act_dim)),
                    'xn': np.empty((0, self.state_dim)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            x = self.env.reset()

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

                    x, r, done, _ = self.env.step(u)
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

    def inference(self, rollouts):
        for roll in rollouts:
            alpha = self.dyn.filtered_posterior(roll['x'], roll['uc'])
            roll['p'], roll['i'] = alpha, np.atleast_2d(alpha[0, :])

            roll['t'] = np.zeros((len(roll['x']), self.nb_modes, self.nb_modes))
            for i in range(len(roll['x'])):
                roll['t'][i] = self.dyn.transitions.matrix(roll['x'][i], roll['uc'][i])

        data = merge(*rollouts)
        return rollouts, data

    def featurize(self, data):
        ivfeat = np.einsum('nh,ndh->ndh', data['i'], self.vfunc.features(data['xi']))
        vfeat = self.vfunc.features(data['x'])
        nvfeat = np.einsum('nkh,ndh->ndkh', data['t'], self.vfunc.features(data['xn']))
        return ivfeat, vfeat, nvfeat

    @staticmethod
    def weights(eta, omega, ifeat, feat, nfeat, gamma, rwrd, normalize=True):
        d, k = feat.shape[1:]
        theta = np.reshape(omega, (d, k))

        ival = np.mean(np.sum(np.einsum('ndh,dh->nh', ifeat, theta),
                              axis=-1, keepdims=True), axis=0, keepdims=True)
        val = np.einsum('ndk,dk->nk', feat, theta)
        nval = np.sum(np.einsum('ndkh,dh->nkh', nfeat, theta), axis=-1)

        adv = rwrd + gamma * nval - val + (1. - gamma) * ival
        max_adv = np.max(adv)
        delta = adv - max_adv if normalize else adv
        weights = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        return weights, delta, max_adv

    def dual(self, var, epsilon, ifeat, feat, nfeat, gamma, rwrd, blf):
        eta, omega = var[0], var[1:]
        weights, _, max_adv = self.weights(eta, omega, ifeat, feat, nfeat, gamma, rwrd)
        g = eta * epsilon + max_adv\
            + eta * np.log(np.mean(np.sum(blf * weights, axis=1)))
        g = g + self.vf_reg * np.sum(omega ** 2)
        return g

    def dual_omega(self, omega, eta, epsilon, ifeat, feat, nfeat, gamma, rwrd, blf):
        weights, _, max_adv = self.weights(eta, omega, ifeat, feat, nfeat, gamma, rwrd)
        g = max_adv + eta * np.log(np.mean(np.sum(blf * weights, axis=1)))
        g = g + self.vf_reg * np.sum(omega ** 2)
        return g

    def dual_eta(self, eta, omega, epsilon, ifeat, feat, nfeat, gamma, rwrd, blf):
        weights, _, max_adv = self.weights(eta, omega, ifeat, feat, nfeat, gamma, rwrd)
        g = eta * epsilon + max_adv\
            + eta * np.log(np.mean(np.sum(blf * weights, axis=1)))
        g = g + self.vf_reg * np.sum(omega ** 2)
        return g

    @staticmethod
    def samples_kl(weights, blf):
        w = np.clip(weights, 1e-75, np.inf)
        w = np.sum(blf * w, axis=1)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def run(self, nb_iter=10, nb_train_samples=5000, buffer_size=0,
            nb_eval_rollouts=25, nb_eval_steps=250,
            ctl_mstep_kwargs={}, verbose=True):

        trace = {'rwrd': [], 'kls': []}

        for it in range(nb_iter):
            _, eval = self.evaluate(nb_eval_rollouts, nb_eval_steps)

            rollouts, _ = self.sample(nb_train_samples, buffer_size)
            self.rollouts, self.data = self.inference(rollouts)
            ivfeat, vfeat, nvfeat = self.featurize(self.data)

            from warnings import filterwarnings
            filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
            res = minimize(self.dual, np.hstack((1e1, self.vfunc.omega)),
                           method='trust-constr', jac=grad(self.dual),
                           args=(self.kl_bound, ivfeat, vfeat, nvfeat, self.discount,
                                 self.data['r'][:, np.newaxis], self.data['p']),
                           bounds=((1e-16, 1e16),) + ((-np.inf, np.inf),) * self.nb_vfeat)

            self.eta, self.vfunc.omega = res.x[0], res.x[1:]

            weights = self.weights(self.eta, self.vfunc.omega,
                                   ivfeat, vfeat, nvfeat, self.discount,
                                   self.data['r'][:, np.newaxis],
                                   normalize=False)[0]

            kls = self.samples_kl(weights, self.data['p'])

            # policy update
            ts = [roll['x'].shape[0] for roll in self.rollouts]
            w = np.split(weights, np.cumsum(ts)[:-1])

            x = [roll['x'] for roll in self.rollouts]
            u = [roll['u'] for roll in self.rollouts]
            p = [roll['p'] for roll in self.rollouts]

            self.ctl.weighted_mstep(p, x, u, w, ctl_mstep_kwargs)

            # rwrd = np.mean(self.data['r'])
            rwrd = np.mean(eval['r'])
            trace['rwrd'].append(rwrd)
            trace['kls'].append(kls)

            if verbose:
                print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kls={kls:{5}.{4}}')

        return trace
