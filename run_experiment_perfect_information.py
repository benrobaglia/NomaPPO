import numpy as np
import pickle
import os

from envs.env_noma_discrete import CentralizedEnvNomaDiscretePoissonURLLC
from algorithms.noma_ppo_discrete import *

xp_name = 'perfect_sensing_results'

# Create the folder
if xp_name not in os.listdir():
    os.mkdir(xp_name)

ns = [4, 8, 12, 16, 20]
load = 1/9.3
timesteps = 100000

scores_list = []
training_results = []

for n in ns:
    constraints = np.array([5 for _ in range(n)])
    env = CentralizedEnvNomaDiscretePoissonURLLC(n,
                                          load,
                                          constraints,
                                            episode_length=50,
                                           sense_period=1,
                                          verbose=False)

    agent = NomaPPO(env,
                  lr_actor=3e-4,
                  lr_critic=1e-3,
                  hidden_size=128,
                  gamma=0.7,
                  recurrent=False)

    res = agent.learn(timesteps)
    score = agent.test()
    training_results.append(res)
    scores_list.append(score)

pickle.dump(training_results, open(f"{xp_name}/training_results.p", 'wb'))
pickle.dump(scores_list, open(f"{xp_name}/scores_list.p", 'wb'))

