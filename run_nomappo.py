import numpy as np
import pickle
import random
from algorithms.noma_ppo import *
from nomaenv import NomaEnv

output_name = 'results/nomappo_aperiodic_collision_3gpp_no_prior.p'

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

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
deadline = 5

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

# Run loop
nomappo_scores = []

for k in k_list:
    score_seed = []
    for seed in range(1):
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
        
        nomappo = NomaPPO(env, 
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 hidden_size=256,
                 gamma=0.3,
                 K_epochs=4,
                 eps_clip=0.1,
                 prior_weight=None,
                 beta=0.1,
                 scheduler=False
                 )

        # nomappo.policy_old.load_state_dict(torch.load(f'results/models/policy_{k}_aperiodic_collision.pt'))
        res = nomappo.learn(5000, update_frequency=100, test_length=20, early_stopping=True)
        score, _, _ = nomappo.test(500)
        print(f"k: {k}, seed: {seed}, URLLC score: {score}")
        score_seed.append(score)
    nomappo_scores.append(score_seed)

results = {
    'scores_nomappo': np.array(nomappo_scores),
    'log': log
}

pickle.dump(results, open(f'{output_name}', 'wb'))

print("End experiment")