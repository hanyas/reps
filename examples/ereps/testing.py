from reps.ereps import eREPS
from reps.envs import Sphere


ereps = eREPS(func=Sphere(dim=5),
              nb_episodes=10,
              kl_bound=0.1,
              cov0=10.0)

ereps.run(nb_iter=250, verbose=True)
