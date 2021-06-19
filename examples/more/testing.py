from reps.more import MORE
from reps.envs import Sphere

more = MORE(func=Sphere(dim=2),
            nb_samples=1000,
            kl_bound=0.05,
            ent_rate=0.99,
            cov0=100.0,
            h0=75.0)

more.run(nb_iter=250, verbose=True)
