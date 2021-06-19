from reps.ehireps import eHiREPS
from reps.envs import Himmelblau


ehireps = eHiREPS(func=Himmelblau(),
                 nb_components=5, nb_episodes=2500,
                 kl_bound=0.1)

trace = ehireps.run(nb_iter=10, verbose=True)
