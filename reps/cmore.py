import autograd.numpy as np

import scipy as sc
from scipy import optimize
from scipy import special

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

import copy


class Policy:

    def __init__(self, context_dim, act_dim, degree, cov0):
        self.context_dim = context_dim
        self.act_dim = act_dim

        self.degree = degree
        self.basis = PolynomialFeatures(self.degree, include_bias=False)
        self.nb_feat = int(sc.special.comb(self.degree + self.context_dim, self.degree) - 1)

        self.b = 1e-8 * np.random.randn(self.act_dim, )
        self.K = 1e-8 * np.random.randn(self.act_dim, self.nb_feat)
        self.cov = cov0 * np.eye(act_dim)

    def features(self, c):
        return self.basis.fit_transform(c.reshape(-1, self.context_dim))

    def mean(self, c):
        feat = self.features(c)
        return self.b + np.einsum('...k,mk->...m', feat, self.K)

    def action(self, c, stoch=True):
        mean = self.mean(c)
        if stoch:
            return np.array([np.random.multivariate_normal(mu, self.cov) for mu in mean])
        else:
            return mean

    def kli(self, pi, c):
        diff = self.mean(c) - pi.mean(c)

        kl = 0.5 * (np.trace(np.linalg.inv(self.cov) @ pi.cov) +
                    np.mean(np.einsum('nk,kh,nh->n', diff, np.linalg.inv(self.cov), diff), axis=0) -
                    self.act_dim + np.log(np.linalg.det(self.cov) / np.linalg.det(pi.cov)))
        return kl

    def entropy(self):
        return 0.5 * np.log(np.linalg.det(self.cov * 2.0 * np.pi * np.exp(1.0)))

    def update(self, eta, omega, model):
        pol = copy.deepcopy(self)

        Raa, Rac, ra = model.Raa, model.Rac, model.ra

        b, K = self.b, self.K
        Q = self.cov
        Qi = np.linalg.inv(Q)

        F = np.linalg.inv(eta * Qi - 2.0 * Raa)
        L = eta * (Qi @ K) + 2.0 * Rac
        f = eta * (Qi @ b) + ra

        pol.cov = F * (eta + omega)
        pol.b = F @ f
        pol.K = F @ L

        return pol


class Model:

    def __init__(self, act_dim, nb_cntxt_feat):
        self.nb_cntxt_feat = nb_cntxt_feat
        self.act_dim = act_dim

        self.R = np.zeros((self.act_dim + self.nb_cntxt_feat, self.act_dim + self.nb_cntxt_feat))
        self.r = np.zeros((self.act_dim + self.nb_cntxt_feat, ))

        self.Raa = np.zeros((self.act_dim, self.act_dim))
        self.ra = np.zeros((self.act_dim,))

        self.Rcc = np.zeros((self.nb_cntxt_feat, self.nb_cntxt_feat))
        self.rc = np.zeros((self.nb_cntxt_feat, ))

        self.Rac = np.zeros((self.act_dim, self.nb_cntxt_feat))

        self.r0 = np.zeros((1, ))

    def fit(self, phi, x, r):
        poly = PolynomialFeatures(2)

        input = np.hstack((x, phi))
        feat = poly.fit_transform(input)

        reg = Ridge(alpha=1e-4, fit_intercept=False)
        reg.fit(feat, r)
        par = reg.coef_

        self.r0 = par[0]
        self.r = par[1:self.act_dim + self.nb_cntxt_feat + 1]

        uid = np.triu_indices(self.act_dim + self.nb_cntxt_feat)
        self.R[uid] = par[self.act_dim + self.nb_cntxt_feat + 1:]
        self.R.T[uid] = self.R[uid]

        self.Raa = self.R[:self.act_dim, :self.act_dim]
        self.Rcc = self.R[-self.nb_cntxt_feat:, -self.nb_cntxt_feat:]
        self.Rac = self.R[:self.act_dim, -self.nb_cntxt_feat:]

        self.ra = 2.0 * self.r[:self.act_dim]
        self.rc = 2.0 * self.r[self.act_dim:]

        # check for positive definitness
        w, v = np.linalg.eig(self.Raa)
        w[w >= 0.0] = -1e-12
        self.Raa = v @ np.diag(w) @ v.T
        self.Raa = 0.5 * (self.Raa + self.Raa.T)


class cMORE:

    def __init__(self, func, nb_episodes,
                 kl_bound, ent_rate,
                 cntxt_degree, **kwargs):

        self.func = func
        self.act_dim = self.func.act_dim
        self.cntxt_dim = self.func.cntxt_dim

        self.nb_episodes = nb_episodes

        self.kl_bound = kl_bound
        self.ent_rate = ent_rate

        self.cntxt_degree = cntxt_degree

        self.basis = PolynomialFeatures(self.cntxt_degree, include_bias=False)
        self.nb_cntxt_feat = int(sc.special.comb(self.cntxt_degree + self.cntxt_dim, self.cntxt_degree) - 1)

        cov0 = kwargs.get('cov0', 100.)
        self.ctl = Policy(self.act_dim, self.cntxt_dim,
                          self.cntxt_degree, cov0)

        self.h0 = kwargs.get('h0', 75.0)

        self.model = Model(self.act_dim, self.nb_cntxt_feat)

        self.eta = np.array([1.0])
        self.omega = np.array([1.0])

        self.data = None
        self.phi = None

    def sample(self, nb_episodes):
        data = {'c': self.func.context(nb_episodes)}
        data['x'] = self.ctl.action(data['c'])
        data['r'] = self.func.eval(data['c'], data['x'])
        return data

    def features(self, c):
        return self.basis.fit_transform(c.reshape(-1, self.cntxt_dim))

    def dual(self, var, eps, beta, ctl, model, phi):
        eta = var[0]
        omega = var[1]

        Raa, Rac, ra = model.Raa, model.Rac, model.ra

        b, K, Q = ctl.b, ctl.K, ctl.cov
        Qi = np.linalg.inv(Q)

        F = eta * Qi - 2 * Raa
        Fi = np.linalg.inv(F)

        f = eta * (Qi @ b) + ra
        L = eta * (Qi @ K) + 2 * Rac

        M = 0.5 * (L.T @ (Fi @ L) - eta * K.T @ (Qi @ K))

        _, q_lgdt = np.linalg.slogdet(2.0 * np.pi * Q)
        _, f_lgdt = np.linalg.slogdet(2.0 * np.pi * (eta + omega) * Fi)

        g = eta * eps - omega * beta - 0.5 * eta * b.T @ Qi @ b\
            + 0.5 * f.T @ Fi @ f - 0.5 * eta * q_lgdt + 0.5 * (eta + omega) * f_lgdt\
            + np.mean(phi @ (L.T @ (Fi @ f) - eta * K.T @ (Qi @ b)))\
            + np.mean(np.sum((phi @ M).T * phi.T, axis=0))

        return g

    def grad(self, var, eps, beta, ctl, model, phi):
        eta = var[0]
        omega = var[1]

        Raa, Rac, ra = model.Raa, model.Rac, model.ra

        b, K, Q= ctl.b, ctl.K, ctl.cov
        Qi = np.linalg.inv(Q)

        F = eta * Qi - 2 * Raa
        Fi = np.linalg.inv(F)

        f = eta * (Qi @ b) + ra
        L = eta * (Qi @ K) + 2 * Rac

        M = 0.5 * (L.T @ (Fi @ L) - eta * K.T @ (Qi @ K))

        _, q_lgdt = np.linalg.slogdet(2.0 * np.pi * Q)
        _, f_lgdt = np.linalg.slogdet(2.0 * np.pi * (eta + omega) * Fi)

        dFi_deta = - (Fi.T @ (Qi @ Fi))
        df_deta = Qi @ b

        deta0 = eps - 0.5 * b.T @ df_deta + 0.5 * f.T @ dFi_deta @ f + f.T @ (Fi @ df_deta)\
               - 0.5 * q_lgdt + 0.5 * f_lgdt - 0.5 * (eta + omega) * np.trace(Fi @ Qi)\
               + 0.5 * self.act_dim

        detal = L.T @ (Fi @ df_deta) - L.T @ (Fi.T @ (Qi @ (Fi @ f)))\
               + (Qi @ K).T @ (Fi @ f) - K.T @ (Qi @ b)

        detaq = 0.5 * L.T @ (dFi_deta @ L) + (Qi @ K).T @ (Fi @ L) - 0.5 * K.T @ Qi @ K

        deta = deta0 + np.mean(phi @ detal, axis=0) + np.mean(np.sum((phi @ detaq).T * phi.T, axis=0))

        domega = - beta + 0.5 * (f_lgdt + self.act_dim)

        return np.hstack([deta, domega])

    def run(self, nb_iter=100, verbose=False):
        trace = {'rwrd': [],  'kl': [], 'ent': []}

        for it in range(nb_iter):
            # update entropy bound
            ent_bound = self.ent_rate * (self.ctl.entropy() + self.h0) - self.h0

            # sample current policy
            self.data = self.sample(self.nb_episodes)
            rwrd = np.mean(self.data['r'])

            # get context features
            self.phi = self.features(self.data['c'])

            # fit quadratic model
            self.model.fit(self.phi, self.data['x'], self.data['r'])

            # optimize dual
            init = np.stack((100.0, 1000.0))
            bnds = ((1e-8, 1e8), (1e-8, 1e8))

            res = sc.optimize.minimize(self.dual, init,
                                       method='L-BFGS-B', jac=self.grad,
                                       args=(self.kl_bound, ent_bound,
                                             self.ctl, self.model, self.phi),
                                       bounds=bnds)
            self.eta = res.x[0]
            self.omega = res.x[1]

            # update policy
            pi = self.ctl.update(self.eta, self.omega, self.model)

            # check kl
            kl = self.ctl.kli(pi, self.data['c'])

            self.ctl = pi
            ent = self.ctl.entropy()

            trace['rwrd'].append(rwrd)
            trace['kl'].append(kl)
            trace['ent'].append(ent)

            if verbose:
                print('it=', it, f'rwrd={rwrd:{5}.{4}}',
                      f'kl={kl:{5}.{4}}',
                      f'ent={ent:{5}.{4}}')

            if ent < -3e2:
                break

        return trace
