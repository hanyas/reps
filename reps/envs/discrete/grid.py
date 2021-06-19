import numpy as np
import matplotlib.pyplot as plt


class Grid:
    def __init__(self):
        # states
        self.state_dim = (9, 10)
        self.states = list(np.indices(self.state_dim))

        # actions
        self.act_dim = (5,)
        self.actions = list(np.indices(self.act_dim))

        # reward
        O = -1e5  # Dangerous places to avoid
        D = 35  # Dirt
        W = -100  # Water
        C = -3000  # Cat
        T = 1000  # Toy

        self.rwd = np.array([[0, O, O, 0, 0, O, O, 0, 0, 0],
                             [0, 0, 0, 0, D, O, 0, 0, D, 0],
                             [0, D, 0, 0, 0, O, 0, 0, O, 0],
                             [O, O, O, O, 0, O, 0, O, O, O],
                             [D, 0, 0, D, 0, O, T, D, 0, 0],
                             [0, O, D, D, 0, O, W, 0, 0, 0],
                             [W, O, 0, O, 0, O, D, O, O, 0],
                             [W, 0, 0, O, D, 0, 0, O, D, 0],
                             [0, 0, 0, D, C, O, 0, 0, D, 0]])

        # transition
        p = 0.0
        self.trans = np.array([[1.0-3*p, p, 0.0, p, 0.0],
                               [p, 1.0-3*p, p, 0.0, 0.0],
                               [0.0, p, 1.0-3*p, p, 0.0],
                               [p, 0.0, p, 1.0-3*p, 0.0],
                               [p, p, p, p, 1.0]])

    @staticmethod
    def dynamics(self, s, a):
        sn = s.copy()

        if a == np.array([0]):  # move up
            sn[0] = np.maximum(s[0] - 1, 0)
        elif a == np.array([1]):  # move down
            sn[0] = np.minimum(s[0] + 1, 8)
        elif a == np.array([2]):  # move right
            sn[1] = np.minimum(s[1] + 1, 9)
        elif a == np.array([3]):  # move left
            sn[1] = np.maximum(s[1] - 1, 0)
        elif a == np.array([4]):  # don't move
            pass

        return sn

    def prob(self):
        return self.trans

    def reward(self, s):
        return self.rwd[s]

    # shows grid world without annotations
    def world(self, tlt):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.set_title(tlt)
        ax.set_xticks(np.arange(0.5, 10.5, 1)) # xticks from 0.5 to 9.5
        ax.set_yticks(np.arange(0.5, 9.5, 1)) # yticks from 0.5 to 8.5

        ax.grid(color='b', linestyle='-', linewidth=1)
        ax.imshow(self.rwd, interpolation='nearest', cmap='copper')
        return ax

    # show policy
    @staticmethod
    def policy(self, policy, ax):
        for x in range(policy.shape[0]):
            for y in range(policy.shape[1]):
                if policy[x, y] == 0:
                    ax.annotate(r'$\uparrow$', xy=(y, x), horizontalalignment='center')
                elif policy[x, y] == 1:
                    ax.annotate(r'$\downarrow$', xy=(y, x), horizontalalignment='center')
                elif policy[x, y] == 2:
                    ax.annotate(r'$\rightarrow$', xy=(y, x), horizontalalignment='center')
                elif policy[x, y] == 3:
                    ax.annotate(r'$\leftarrow$', xy=(y, x), horizontalalignment='center')
                elif policy[x, y] == 4:
                    ax.annotate(r'$\perp$', xy=(y, x), horizontalalignment='center')
