import numpy as np
import pickle
from nomaenv import NomaEnv
# from algorithms.noma_ppo import NomaPPO
from algorithms.noma_ppo_rnn import NomaPPO
import random

# output_name = 'results/scenario3gpp/nomappo.p'
output_name = 'results/scenario3GPP/nomappo_recurrent.p'

# Fix random seed
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

# Environment parameters (Config for aperiodic collision model)
episode_length=200
max_simultaneous_devices=3
traffic_model = 'aperiodic'
channel_model = 'interference'
path_loss = True
shadowing = True
fast_fading = True
verbose = False
aperiodic_type = 'lambda'

# Load distances and shadowing parameters
baselines = pickle.load(open('results/scenario3GPP/baselines.p', 'rb'))
manual_distances_dic = baselines['manual_distances']
manual_shadowing_dic = baselines['manual_shadowing']

# Global parameters of the experiment
k_list = [6, 12, 18, 24, 30]
n_seeds = 5

log = {
    'random_seed':random_seed,
    'episode_length':episode_length,
    'max_simultaneous_devices':max_simultaneous_devices,
    'traffic_model':traffic_model,
    'channel_model':channel_model,
    'path_loss':path_loss,
    'shadowing':shadowing,
    'fast_fading':fast_fading,
    'verbose':verbose,
    'k_list':k_list,
    'n_seeds':n_seeds,
}

print(log)

scores_noma_ppo = []
jains_noma_ppo = []


for k in k_list:
    print(f"k: {k}")
    
    scores_noma_ppo_seed = []
    jains_noma_ppo_seed = []
    for seed in range(n_seeds):
        # Homogeneous agents
        lbdas = np.array([1/11.2 for _ in range(k)])
        deadlines = np.array([6 for _ in range(k)])
        manual_distances = manual_distances_dic[str(k)][seed]
        manual_shadowing = manual_shadowing_dic[str(k)][seed]

        offsets = None
        arrival_probs = None
        period = None
        env = NomaEnv(k,
                deadlines,
                lbdas=lbdas,
                period=period,
                arrival_probs=arrival_probs,
                offsets=offsets,
                episode_length=episode_length,
                max_simultaneous_devices=max_simultaneous_devices,
                traffic_model=traffic_model,
                channel_model=channel_model,
                radius=None,
                full_obs=False,
                manual_distances=manual_distances,
                manual_shadowing=manual_shadowing,
                path_loss=path_loss,
                shadowing=shadowing,
                fast_fading=fast_fading,
                verbose=verbose,
                aperiodic_type='lambda',
                reward_type=0,
             )

        nomappo = NomaPPO(env, 
                        lr_actor=1e-4,
                        lr_critic=1e-4,
                        hidden_size=64,
                        gamma=0.4,
                        K_epochs=2,
                        eps_clip=0.1,
                        beta=0.1,
                        scheduler=False
                 )

        res = nomappo.learn(50000,
                            update_frequency=50,
                            test_length=100,
                            # model_path=f'results/scenario3gpp/models_nomappo/nomappo_k{k}_seed{seed}.pth',
                            test_frequency=1000,
                            early_stopping=True)
        
        score, jains, avg_reward, avg_collisions = nomappo.test(1000)
        print(f"k: {k}, seed: {seed}, score: {score}, jains: {jains}\n")
        scores_noma_ppo_seed.append(score)
        jains_noma_ppo_seed.append(jains)

    scores_noma_ppo.append(scores_noma_ppo_seed)
    jains_noma_ppo.append(jains_noma_ppo_seed)

    results = {'scores_noma_ppo': np.array(scores_noma_ppo), 'jains_noma_ppo': np.array(jains_noma_ppo)}
    pickle.dump(results, open(f'{output_name}', 'wb'))
    print(f"Data dumped for k {k}")
