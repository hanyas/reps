from reps.cmore import cMORE
from reps.envs import CSphere


cmore = cMORE(func=CSphere(cntxt_dim=1, act_dim=1),
              nb_episodes=1000,
              kl_bound=0.05, ent_rate=0.99,
              cov0=100.0, h0=75.0, cntxt_degree=1)

trace = cmore.run(nb_iter=250, verbose=True)
