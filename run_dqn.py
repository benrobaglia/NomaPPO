import numpy as np
import pickle
import random

from nomaenv import NomaEnv
from algorithms.dqn import *

output_name = 'results/vanilla_dqn_aperiodic_collision_3gpp.p'

# Fix random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Environment parameters (Config for aperiodic collision model)
episode_length=200
max_simultaneous_devices=3
traffic_model='aperiodic'
channel_model='collision'
distances=None
path_loss=False
shadowing=False
fast_fading=False
verbose=False
lbda = 1/9.3
deadline = 6

# Global parameters of the experiment
k_list = [4, 8, 12, 16, 20]
n_seeds = 1

log = {
    'random_seed':random_seed,
    'episode_length':episode_length,
    'max_simultaneous_devices':max_simultaneous_devices,
    'traffic_model':traffic_model,
    'channel_model':channel_model,
    'distances':distances,
    'path_loss':path_loss,
    'shadowing':shadowing,
    'fast_fading':fast_fading,
    'verbose':verbose,
    'lbda':lbda,
    'deadline':deadline,
    'k_list':k_list,
    'n_seeds':n_seeds,
}

print(log)

scores_vanilla_dqn = []

for k in k_list:
    deadlines = np.array([deadline for _ in range(k)])
    offsets = None
    arrival_probs = None
    period = None
    env = NomaEnv(k,
            deadlines,
            lbda=lbda,
            period=period,
            arrival_probs=arrival_probs,
            offsets=offsets,
            episode_length=episode_length,
            max_simultaneous_devices=max_simultaneous_devices,
            traffic_model=traffic_model,
            channel_model=channel_model,
            distances=distances,
            path_loss=path_loss,
            shadowing=shadowing,
            fast_fading=fast_fading,
            verbose=verbose
            )
    
    dqn_scheduler = DQN(
                        env,
                        replay_start_size=500,
                        replay_buffer_size=1000000,
                        gamma=0.99,
                        update_target_frequency=100,
                        minibatch_size=32,
                        learning_rate=1e-3,
                        update_frequency=1,
                        initial_exploration_rate=1,
                        final_exploration_rate=0.1,
                        adam_epsilon=1e-8,
                        loss='huber',
                        seed=None,
                        )

    _ = dqn_scheduler.train(5000)
    score = dqn_scheduler.test(500)
    print(f"k: {k}, URLLC score: {score}")
    scores_vanilla_dqn.append(score)

results = {
    'scores_nomappo': np.array(scores_vanilla_dqn),
    'log': log
}

pickle.dump(results, open(f'{output_name}', 'wb'))

print("End experiment")