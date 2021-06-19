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
        kwargs = {'method': 'direct', 'nb_iter': 25}

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

        self.omega = 1e-8 * np.random.uniform(size=(self.nb_feat,))

    def features(self, x):
        return self.basis.fit_transform(x)

    def values(self, x):
        feat = self.features(x)
        return np.dot(feat, self.omega)


class Qfunction:

    def __init__(self, state_dim, act_dim, nb_feat, scale, mult):
        self.state_dim = state_dim
        self.act_dim = act_dim

        self.nb_feat = nb_feat
        self.scale = scale
        self.mult = mult
        self.basis = FourierFeatures(self.state_dim + self.act_dim,
                                     self.nb_feat, self.scale, self.mult)

        self.theta = 1e-8 * np.random.uniform(size=(self.nb_feat,))

    def features(self, x, u):
        xu = np.hstack((x, u))
        return self.basis.fit_transform(xu)

    def values(self, x, u):
        feat = self.features(x, u)
        return np.dot(feat, self.theta)


class acREPS:

    def __init__(self, env,
                 kl_bound, discount, lmbda,
                 scale, mult, nb_pfeat,
                 nb_vfeat, vf_reg):

        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]

        self.kl_bound = kl_bound
        self.discount = discount
        self.lmbda = lmbda

        self.nb_vfeat = nb_vfeat
        self.nb_pfeat = nb_pfeat

        self.scale = scale
        self.mult = mult
        self.vf_reg = vf_reg

        self.vfunc = Vfunction(self.state_dim, nb_feat=self.nb_vfeat,
                               scale=self.scale[:self.state_dim],
                               mult=self.mult)

        self.qfunc = Qfunction(self.state_dim, self.act_dim, nb_feat=self.nb_vfeat,
                               scale=self.scale, mult=self.mult)

        likelihood_precision_prior = {'alpha': 1., 'beta': 25.}
        parameter_precision_prior = {'alpha': 1., 'beta': 1e-1}

        self.ctl = ARDPolicy(self.state_dim, self.act_dim,
                             self.nb_pfeat, self.scale[:self.state_dim], self.mult,
                             likelihood_precision_prior,
                             parameter_precision_prior)

        self.ctl.object.A *= 0.
        self.ctl.object.lmbda = 1./25. * np.eye(self.act_dim)

        self.ulim = self.env.action_space.high

        self.data = {}
        self.rollouts = []

        self.eta = np.array([1.0])

    def sample(self, nb_samples, buffer_size=0, stoch=True, render=False):
        if len(self.rollouts) >= buffer_size:
            rollouts = random.sample(self.rollouts, buffer_size)
        else:
            rollouts = []

        n = 0
        while True:
            roll = {'x': np.empty((0, self.state_dim)),
                    'u': np.empty((0, self.act_dim)),
                    'uc': np.empty((0, self.act_dim)),
                    'xn': np.empty((0, self.state_dim)),
                    'r': np.empty((0,)),
                    'done': np.empty((0,), np.int64)}

            x = self.env.reset()

            done = False
            while not done:
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

    def lstd(self, rollouts, gamma, lmbda, alpha=1e-6, beta=1e-6):
        for roll in rollouts:
            # state-action features
            roll['phi'] = self.qfunc.features(roll['x'], roll['u'])
            # actions under current policy
            roll['un'] = self.ctl.action(roll['xn'], stoch=False)
            # next-state-action features
            roll['nphi'] = self.qfunc.features(roll['xn'], roll['un'])
            # find and turn-off features of absorbing states
            absorbing = np.argwhere(roll['done']).flatten()
            roll['nphi'][absorbing, :] *= 0.0

        K = self.qfunc.nb_feat * self.qfunc.act_dim
        A = np.zeros((K, K))
        b = np.zeros((K,))
        I = np.eye(K)

        PHI = np.zeros((0, K))

        for roll in rollouts:
            t = 0
            z = roll['phi'][t, :]

            done = False
            while not done:
                done = roll['done'][t]

                PHI = np.vstack((PHI, roll['phi'][t, :]))
                A += np.outer(z, roll['phi'][t, :] - (1 - done) * gamma * roll['nphi'][t, :])
                b += z * roll['r'][t]

                if not done:
                    z = lmbda * z + roll['phi'][t + 1, :]
                    t = t + 1

        C = np.linalg.solve(PHI.T.dot(PHI) + alpha * I, PHI.T).T
        X = C.dot(A + alpha * I)
        y = C.dot(b)

        theta = np.linalg.solve(X.T.dot(X) + beta * I, X.T.dot(y))

        return theta, rollouts, merge(*rollouts)

    @staticmethod
    def generalized_advantage(data, phi, omega, discount, lmbda):
        values = np.dot(phi, omega)
        adv = np.zeros_like(values)

        for rev_k, v in enumerate(reversed(values)):
            k = len(values) - rev_k - 1
            if data['done'][k]:
                adv[k] = data['r'][k] - values[k]
            else:
                adv[k] = data['r'][k] + discount * values[k + 1] - values[k] +\
                         discount * lmbda * adv[k + 1]

        targets = adv + values
        return targets

    @staticmethod
    def monte_carlo(data, discount):
        rwrds = data['r']
        targets = np.zeros_like(rwrds)

        for ik, v in enumerate(reversed(rwrds)):
            k = len(rwrds) - ik - 1
            if data['done'][k]:
                targets[k] = rwrds[k]
            else:
                targets[k] = rwrds[k] + discount * rwrds[k + 1]
        return targets

    def featurize(self, data):
        vfeat = self.vfunc.features(data['x'])
        mvfeat = np.mean(vfeat, axis=0, keepdims=True)
        return vfeat - mvfeat

    @staticmethod
    def weights(eta, omega, features, targets, normalize=True):
        adv = targets - np.dot(features, omega)
        delta = adv - np.max(adv) if normalize else adv
        weights = np.exp(np.clip(delta / eta, EXP_MIN, EXP_MAX))
        return weights, delta, np.max(adv)

    def dual(self, var, epsilon, phi, targets):
        eta, omega = var[0], var[1:]
        weights, _, max_adv = self.weights(eta, omega, phi, targets)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(weights, axis=0))
        g = g + self.vf_reg * np.sum(omega ** 2)
        return g

    def grad(self, var, epsilon, phi, targets):
        eta, omega = var[0], var[1:]
        weights, delta, max_adv = self.weights(eta, omega, phi, targets)

        deta = epsilon + np.log(np.mean(weights, axis=0)) - \
               np.sum(weights * delta, axis=0) / (eta * np.sum(weights, axis=0))

        domega = - np.sum(weights[:, np.newaxis] * phi, axis=0) / np.sum(weights, axis=0)
        domega = domega + self.vf_reg * 2 * omega

        return np.hstack((deta, domega))

    def dual_eta(self, eta, omega, epsilon, phi, targets):
        weights, _, max_adv = self.weights(eta, omega, phi, targets)
        g = eta * epsilon + max_adv + eta * np.log(np.mean(weights, axis=0))
        return g

    def dual_omega(self, omega, eta, phi, targets):
        weights, _, max_adv = self.weights(eta, omega, phi, targets)
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

            # self.qfunc.theta, self.rollouts, self.data = self.lstd(self.rollouts, gamma=self.discount, lmbda=self.lmbda)
            # targets = self.qfunc.values(self.data['xn'], self.data['un'])

            # targets = self.monte_carlo(self.data, self.discount)

            targets = self.generalized_advantage(self.data, vf_feat, self.vfunc.omega,
                                                 self.discount, self.lmbda)

            if not iterative:
                res = sc.optimize.minimize(self.dual, np.hstack((1e6, self.vfunc.omega)),
                                           method='L-BFGS-B', jac=grad(self.dual),
                                           args=(self.kl_bound, vf_feat, targets),
                                           bounds=((1e-16, 1e16),) + ((-np.inf, np.inf),) * self.nb_vfeat)

                self.eta, self.vfunc.omega = res.x[0], res.x[1:]
            else:
                self.eta, self.vfunc.omega = 1.0, np.random.uniform(size=(self.nb_vfeat,))
                for _ in range(25):
                    res = sc.optimize.minimize(self.dual_eta, self.eta,
                                               method='L-BFGS-B', jac=grad(self.dual_eta),
                                               args=(self.vfunc.omega, self.kl_bound,
                                                     vf_feat, targets),
                                               bounds=((1e-8, 1e8),))

                    # check = sc.optimize.check_grad(self.dual_eta,
                    #                                self.grad_eta, res.x,
                    #                                self.vfunc.omega,
                    #                                self.kl_bound,
                    #                                vf_feat,
                    #                                targets)
                    # print('Eta Error', check)

                    self.eta = res.x

                    res = sc.optimize.minimize(self.dual_omega, self.vfunc.omega,
                                               method='L-BFGS-B', jac=grad(self.dual_omega),
                                               args=(self.eta, vf_feat, targets))

                    # check = sc.optimize.check_grad(self.dual_omega,
                    #                                self.grad_omega, res.x,
                    #                                self.eta,
                    #                                vf_feat,
                    #                                targets)
                    # print('Omega Error', check)

                    self.vfunc.omega = res.x

            weights, _, _ = self.weights(self.eta, self.vfunc.omega,
                                         vf_feat, targets, normalize=False)

            self.ctl.update(self.data['x'], self.data['u'], weights)

            kls = self.samples_kl(weights)

            # rwrd = np.mean(self.data['r'])
            rwrd = np.mean(eval['r'])

            trace['rwrd'].append(rwrd)
            trace['kls'].append(kls)

            if verbose:
                print('it=', it, f'rwrd={rwrd:{5}.{4}}', f'kls={kls:{5}.{4}}')

        return trace
