import autograd.numpy as np
from autograd import grad, jacobian
from autograd.numpy.random import multivariate_normal as mvn

import scipy as sc
from scipy import optimize
from scipy import stats

from sklearn.linear_model import Ridge

import random
from copy import deepcopy


EXP_MAX = 700.0
EXP_MIN = -700.0


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
        self.shift = np.random.uniform(0., 2. * np.pi, size=self.nb_feat)

    def fit_transform(self, x):
        phi = self.norm * np.cos(np.einsum('kd,...d->...k', self.freq, x) + self.shift)
        if self.with_offset:
            ones = np.ones((x.shape[0],))
            return np.column_stack((phi, ones))
        else:
            return phi


class ARDPolicy:

    def __init__(self, state_dim, act_dim,
                 nb_feat, scale, mult,
                 likelihood_precision_prior,
                 parameter_precision_prior):

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.nb_feat = nb_feat
        self.scale = scale
        self.mult = mult
        self.basis = FourierFeatures(self.state_dim, self.nb_feat,
                                     self.scale, self.mult,
                                     with_offset=False)

        self.input_dim = self.nb_feat + 1
        self.output_dim = self.act_dim

        from sds.distributions.gamma import Gamma

        alpha, beta = likelihood_precision_prior['alpha'], likelihood_precision_prior['beta']
        likelihood_precision_prior = Gamma(dim=1,
                                           alphas=alpha * np.ones((1,)),
                                           betas=beta * np.ones((1,)))

        alpha, beta = parameter_precision_prior['alpha'], parameter_precision_prior['beta']
        parameter_precision_prior = Gamma(dim=self.input_dim,
                                          alphas=alpha * np.ones((self.input_dim,)),
                                          betas=beta * np.ones((self.input_dim,)))

        from sds.distributions.composite import MultiOutputLinearGaussianWithAutomaticRelevance
        self.object = MultiOutputLinearGaussianWithAutomaticRelevance(self.input_dim,
                                                                      self.output_dim,
                                                                      likelihood_precision_prior,
                                                                      parameter_precision_prior)

    @property
    def params(self):
        return self.object.params

    @params.setter
    def params(self, values):
        self.object.params = values

    def features(self, x):
        f = self.basis.fit_transform(np.atleast_2d(x))
        return np.squeeze(f) if x.ndim == 1\
               else np.reshape(f, (x.shape[0], -1))

    def mean(self, x):
        f = self.features(x)
        u = self.object.mean(f)
        return np.atleast_1d(u)

    def sample(self, x):
        f = self.features(x)
        u = self.object.rvs(f)
        return np.atleast_1d(u)

    def action(self, x, stoch):
        return self.sample(x) if stoch\
               else self.mean(x)

    def update(self, x, u, w):
        kwargs = {'method': 'direct', 'nb_iter': 50}

        f = self.features(x)
        self.object.em(f, u, w, **kwargs)

        for dist in self.object.dists:
            dist.likelihood_precision_prior = deepcopy(dist.likelihood_precision_posterior)
            dist.parameter_precision_prior = deepcopy(dist.parameter_precision_posterior)


class MLPolicy:

    def __init__(self, state_dim, act_dim,
                 nb_feat, scale, mult, cov0,
                 reg, with_offset=True):

        self.state_dim = state_dim
        self.act_dim = act_dim

        self.nb_feat = nb_feat
        self.scale = scale
        self.mult = mult
        self.basis = FourierFeatures(self.state_dim, self.nb_feat,
                                     self.scale, self.mult, with_offset)

        self.K = np.random.randn(self.act_dim, self.nb_feat)
        self.cov = cov0 * np.eye(act_dim)

        self.reg = reg

    def features(self, x):
        x2d = x.reshape(-1, self.state_dim)
        return np.squeeze(self.basis.fit_transform(x2d))

    def mean(self, x):
        feat = self.features(x)
        return np.einsum('...k,mk->...m', feat, self.K)

    def action(self, x, stoch):
        mean = self.mean(x)
        return mvn(mean, self.cov) if stoch else mean

    def entropy(self):
        cov = self.cov
        return 0.5 * np.log(np.linalg.det(cov * 2.0 * np.pi * np.exp(1.0)))

    def max_likelihood(self, x, u, w):
        model = Ridge(alpha=self.reg, fit_intercept=False)
        model.fit(self.features(x), u, sample_weight=w)
        self.K = model.coef_
        self.cov = np.sum(np.einsum('nk,n,nh->nkh', u - self.mean(x),
                                    w, u - self.mean(x)), axis=0) / np.sum(w)

    def update(self, x, u, w):
        self.max_likelihood(x, u, w)


class Vfunction:

    def __init__(self, state_dim, nb_feat, scale, mult):
        self.state_dim = state_dim

        self.nb_feat = nb_feat
        self.scale = scale
        self.mult = mult
        self.basis = FourierFeatures(self.state_dim, self.nb_feat,
                                     self.scale, self.mult)

        self.omega = np.random.uniform(size=(self.nb_feat,))

    def features(self, x):
        return self.basis.fit_transform(x)

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class REPS:

    def __init__(self, env, kl_bound, discount,
                 scale, mult, nb_pfeat, nb_vfeat, vf_reg):

        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.kl_bound = kl_bound
        self.discount = discount

        self.nb_vfeat = nb_vfeat
        self.nb_pfeat = nb_pfeat

        self.scale = scale
        self.mult = mult
        self.vf_reg = vf_reg

        self.vfunc = Vfunction(self.state_dim, nb_feat=self.nb_vfeat,
                               scale=self.scale, mult=self.mult)

        likelihood_precision_prior = {'alpha': 1., 'beta': 25}
        parameter_precision_prior = {'alpha': 1., 'beta': 1e1}

        self.ctl = ARDPolicy(self.state_dim, self.act_dim,
                             self.nb_pfeat, self.scale, self.mult,
                             likelihood_precision_prior,
                             parameter_precision_prior)

        self.ctl.object.A *= 0.
        self.ctl.object.lmbda = 1./25. * np.eye(self.act_dim)

        self.ulim = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.eta = np.array([1.0])

    def sample(self, nb_samples, buffer_size=0, reset=True, stoch=True):
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
                    u = self.ctl.action(x, stoch)
                    uc = np.clip(u, -self.ulim, self.ulim)

                    roll['x'] = np.vstack((roll['x'], x))
                    roll['u'] = np.vstack((roll['u'], u))
                    roll['uc'] = np.vstack((roll['uc'], uc))

                    x, r, done, _ = self.env.step(u)

                    roll['xn'] = np.vstack((roll['xn'], x))
                    roll['r'] = np.hstack((roll['r'], r))
                    roll['done'] = np.hstack((roll['done'], done))

                    n = n + 1
                    if n >= nb_samples:
                        roll['done'][-1] = True
                        rollouts.append(roll)
                        data = merge(*rollouts)
                        return rollouts, data

            rollouts.append(roll)

    def evaluate(self, nb_rollouts, nb_steps, stoch=False):
        rollouts = []

        for n in range(nb_rollouts):
            roll = {'x': np.empty((0, self.state_dim)),
                    'u': np.empty((0, self.act_dim)),
                    'uc': np.empty((0, self.act_dim)),
                    'r': np.empty((0,))}

            x = self.env.reset()

            for t in range(nb_steps):
                u = self.ctl.action(x, stoch)
                uc = np.clip(u, -self.ulim, self.ulim)

                roll['x'] = np.vstack((roll['x'], x))
                roll['u'] = np.vstack((roll['u'], u))
                roll['uc'] = np.vstack((roll['uc'], uc))

                x, r, done, _ = self.env.step(u)

                roll['r'] = np.hstack((roll['r'], r))

            rollouts.append(roll)

        data = merge(*rollouts)
        return rollouts, data

    def featurize(self, data):
        ivfeat = np.mean(self.vfunc.features(data['xi']),
                         axis=0, keepdims=True)
        vfeat = self.vfunc.features(data['x'])
        nvfeat = self.vfunc.features(data['xn'])
        feat = self.discount * nvfeat - vfeat\
               + (1.0 - self.discount) * ivfeat
        return feat

    @staticmethod
    def huber(delta, x):
        loss = delta**2 * (np.sqrt(1. + (x / delta)**2) - 1.)
        return np.sum(loss)

    @staticmethod
    def weights(eta, omega, phi, rwrd, normalize=True):
        adv = rwrd + np.dot(phi, omega)
        delta = adv - np.max(adv) if normalize else adv
        weights = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        return weights, delta, np.max(adv)

    def dual(self, var, epsilon, phi, rwrd):
        eta, omega = var[0], var[1:]
        weights, _, max_adv = self.weights(eta, omega, phi, rwrd)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(weights, axis=0))
        g = g + self.vf_reg * np.sum(omega ** 2)
        # g = g + self.vf_reg * self.huber(25, omega)
        return g

    def grad(self, var, epsilon, phi, rwrd):
        eta, omega = var[0], var[1:]
        weights, delta, max_adv = self.weights(eta, omega, phi, rwrd)

        deta = epsilon + np.log(np.mean(weights, axis=0)) - \
               np.sum(weights * delta, axis=0) / (eta * np.sum(weights, axis=0))

        domega = np.sum(weights[:, np.newaxis] * phi, axis=0) / np.sum(weights, axis=0)
        domega = domega + self.vf_reg * 2 * omega

        return np.hstack((deta, domega))

    def dual_eta(self, eta, omega, epsilon, phi, rwrd):
        weights, _, max_adv = self.weights(eta, omega, phi, rwrd)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(weights, axis=0))
        return g

    def dual_omega(self, omega, eta, phi, rwrd):
        weights, _, max_adv = self.weights(eta, omega, phi, rwrd)
        g = max_adv + eta * np.log(np.mean(weights, axis=0))
        g = g + self.vf_reg * np.sum(omega ** 2)
        return g

    @staticmethod
    def samples_kl(weights):
        weights = np.clip(weights, 1e-75, np.inf)
        weights = weights / np.mean(weights, axis=0)
        return np.mean(weights * np.log(weights), axis=0)

    def run(self, nb_iter=10, nb_train_samples=5000,
            buffer_size=0, nb_eval_rollouts=25, nb_eval_steps=250,
            verbose=True, iterative=True):

        trace = {'rwrd': [], 'kls': []}

        for it in range(nb_iter):
            _, eval = self.evaluate(nb_eval_rollouts, nb_eval_steps)

            self.rollouts, self.data = self.sample(nb_train_samples, buffer_size)
            vf_feat = self.featurize(self.data)

            if not iterative:
                res = sc.optimize.minimize(self.dual, np.hstack((1e6, self.vfunc.omega)),
                                           method='L-BFGS-B', jac=grad(self.dual),
                                           args=(self.kl_bound, vf_feat, self.data['r']),
                                           bounds=((1e-16, 1e16),) + ((-np.inf, np.inf),) * self.nb_vfeat)

                self.eta, self.vfunc.omega = res.x[0], res.x[1:]
            else:
                self.eta = np.array([1e6])
                # self.vfunc.omega = npr.uniform(size=(self.nb_vfeat,))
                for _ in range(10):
                    res = sc.optimize.minimize(self.dual_omega, self.vfunc.omega,
                                               method='L-BFGS-B', jac=grad(self.dual_omega),
                                               args=(self.eta, vf_feat, self.data['r']))

                    # check = sc.optimize.check_grad(self.dual_omega,
                    #                                self.grad_omega,
                    #                                res.x,
                    #                                self.eta,
                    #                                vf_feat,
                    #                                self.data['r'])
                    # print('Omega Error', check)

                    self.vfunc.omega = res.x

                    res = sc.optimize.minimize(self.dual_eta, self.eta,
                                               method='L-BFGS-B', jac=grad(self.dual_eta),
                                               args=(self.vfunc.omega, self.kl_bound,
                                                     vf_feat, self.data['r']),
                                               bounds=((1e-8, 1e8),))

                    # check = sc.optimize.check_grad(self.dual_eta,
                    #                                self.grad_eta,
                    #                                res.x,
                    #                                self.vfunc.omega,
                    #                                self.kl_bound,
                    #                                vf_feat,
                    #                                self.data['r'])
                    # print('Eta Error', check)

                    self.eta = res.x

            weights, _, _ = self.weights(self.eta, self.vfunc.omega,
                                         vf_feat, self.data['r'],
                                         normalize=False)

            kls = self.samples_kl(weights)

            self.ctl.update(self.data['x'], self.data['u'], weights)

            # rwrd = np.mean(self.data['r'])
            rwrd = np.mean(eval['r'])

            trace['rwrd'].append(rwrd)
            trace['kls'].append(kls)

            if verbose:
                print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kls={kls:{5}.{4}}')

        return trace
