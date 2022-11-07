import numpy as np
import pickle
from nomaenv import NomaEnv
from algorithms.baselines import RandomScheduler, EarliestDeadlineFirstScheduler, SlottedAloha
import random

output_name = 'results/baselines_aperiodic_collision_3gpp.p'

# Fix random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Environment parameters (Config for aperiodic collision model)
episode_length=100
max_simultaneous_devices=3
traffic_model='aperiodic'
channel_model='collision'
distances=None
path_loss=False
shadowing=False
fast_fading=False
verbose=False
lbda = 1/9.3
deadline = 5

# Global parameters of the experiment
k_list = [4, 8, 12, 16, 20]
n_seeds = 5

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

# Random scheduler
scores_random_scheduler = []

for k in k_list:
    score_seed = []
    for seed in range(n_seeds):
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
        
        random_scheduler = RandomScheduler(env, verbose=True)
        score = random_scheduler.run(500)
        print(f"k: {k}, seed: {seed}, URLLC score: {score}")
        score_seed.append(score)
    scores_random_scheduler.append(score_seed)

# Earliest deadline first scheduler
scores_edf_scheduler = []

for k in k_list:
    score_seed = []
    for seed in range(n_seeds):
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
        
        edf_scheduler = EarliestDeadlineFirstScheduler(env, verbose=True)
        score = edf_scheduler.run(500)
        print(f"k: {k}, seed: {seed}, URLLC score: {score}")
        score_seed.append(score)
    scores_edf_scheduler.append(score_seed)

# Slotted ALOHA
scores_aloha = []
transmission_probs_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for k in k_list:
    score_seed = []
    for seed in range(n_seeds):
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
        
        aloha = SlottedAloha(env, transmission_prob=0.5, verbose=False)
        cv = aloha.get_best_transmission_probs(100, transmission_probs_list)
        aloha.transmission_prob = transmission_probs_list[np.argmax(cv)]
        score = aloha.run(500)
        print(f"k: {k}, seed: {seed}, transmission prob: {transmission_probs_list[np.argmax(cv)]}, URLLC score: {score}")
        score_seed.append(score)
    scores_aloha.append(score_seed)

baselines = {
    'random_scheduler': np.array(scores_random_scheduler),
    'edf_scheduler': np.array(scores_edf_scheduler),
    'aloha': np.array(scores_aloha),
    'log':log
}

pickle.dump(baselines, open(f'{output_name}', 'wb'))

print("End experiment")