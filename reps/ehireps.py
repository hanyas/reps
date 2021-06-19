import autograd.numpy as np
from autograd import grad

import scipy as sc
from scipy import optimize

from mimo import models
from mimo.distributions import NormalInverseWishart
from mimo.distributions import BayesianCategoricalWithDirichlet
from mimo.distributions import BayesianGaussian
from mimo.distributions import Dirichlet

import copy
import operator

EXP_MAX = 700.0
EXP_MIN = -700.0


class Policy:

    def __init__(self, act_dim, nb_components):
        self.act_dim = act_dim
        self.nb_components = nb_components

        gating_hypparams = dict(K=self.nb_components, alphas=np.ones((self.nb_components,)))
        gating_prior = Dirichlet(**gating_hypparams)

        components_hypparams = dict(mu=np.zeros((self.act_dim, )),
                                    kappa=0.01,
                                    psi=np.eye(self.act_dim),
                                    nu=self.act_dim + 1)

        components_prior = NormalInverseWishart(**components_hypparams)

        self.mixture = models.Mixture(gating=BayesianCategoricalWithDirichlet(gating_prior),
                                      components=[BayesianGaussian(components_prior)
                                                  for _ in range(self.nb_components)])

    def action(self, n):
        samples, _, resp = self.mixture.generate(n, resp=True)
        return samples, resp

    def update(self, weights):
        allscores = []
        allmodels = []

        for superitr in range(3):
            # Gibbs sampling to wander around the posterior
            for _ in range(100):
                self.mixture.resample_model(importance=[weights])
            # mean field to lock onto a mode
            scores = [self.mixture.meanfield_coordinate_descent_step(importance=[weights]) for _ in range(100)]

            allscores.append(scores)
            allmodels.append(copy.deepcopy(self.mixture))

        models_and_scores = sorted([(m, s[-1]) for m, s in zip(allmodels, allscores)], key=operator.itemgetter(1), reverse=True)

        self.mixture = models_and_scores[0][0]

        # clear stuff
        self.mixture.clear_plot()
        self.mixture.clear_data()
        self.mixture.clear_caches()


class eHiREPS:

    def __init__(self, func, nb_episodes,
                 nb_components, kl_bound):

        self.func = func
        self.act_dim = self.func.act_dim

        self.nb_components = nb_components
        self.nb_episodes = nb_episodes

        self.kl_bound = kl_bound

        self.ctl = Policy(self.act_dim, self.nb_components)

        self.data = None
        self.w = None

        self.eta = np.array([1.0])

    def sample(self, nb_episodes):
        x, p = self.ctl.action(nb_episodes)
        data = {'x': x, 'p': p}
        data['r'] = self.func.eval(data['x'])
        return data

    @staticmethod
    def weights(r, eta, normalize=True):
        adv = r - np.max(r) if normalize else r
        w = np.exp(np.clip(adv / eta, EXP_MIN, EXP_MAX))
        return w, adv

    def dual(self, eta, eps, r, p):
        w, _ = self.weights(r, eta)
        g = eta * eps + np.max(r)\
            + eta * np.log(np.mean(np.sum(p.T * w, axis=0)))
        return g

    @staticmethod
    def samples_kl(p, w):
        w = np.clip(w, 1e-75, np.inf)
        w = np.sum(p.T * w, axis=0)
        w = w / np.mean(w, axis=0)
        return np.mean(w * np.log(w), axis=0)

    def run(self, nb_iter=1, verbose=False):
        trace = {'rwrd': [], 'kls': []}

        for it in range(nb_iter):
            self.data = self.sample(self.nb_episodes)
            rwrd = np.mean(self.data['r'])

            res = sc.optimize.minimize(self.dual, np.array([1.0]),
                                       method='SLSQP', jac=grad(self.dual),
                                       args=(self.kl_bound, self.data['r'], self.data['p']),
                                       bounds=((1e-8, 1e8),))

            self.eta = res.x

            self.w, _ = self.weights(self.data['r'], self.eta,
                                     normalize=False)

            kls = self.samples_kl(self.data['p'], self.w)

            self.ctl.update(self.w)

            trace['rwrd'].append(rwrd)
            trace['kls'].append(kls)

            if verbose:
                print('it=', it,
                      f'rwrd={rwrd:{5}.{4}}',
                      f'kls={kls:{5}.{4}}')

        return trace
