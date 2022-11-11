import numpy as np
from nomaenv import NomaEnv
from algorithms.baselines import SlottedAloha

# Simplified version of the game

k = 5
deadlines = np.array([5 for _ in range(k)])
offsets = None
arrival_probs = None
period = None

env = NomaEnv(k,
              deadlines,
              lbda=1,
              period=period,
              arrival_probs=arrival_probs,
              offsets=offsets,
              episode_length=10,
              max_simultaneous_devices=3,
              traffic_model='aperiodic',
              channel_model='collision',
              distances=None,
              path_loss=False,
              shadowing=False,
              fast_fading=False,
              verbose=False
             )

# Run 1 episode

done = False
obs = env.reset()

while not done:
    print(f"Frame: {env.timestep}")
    print(f"Internal state: {obs}")
    print(f"Last time since polled: {env.last_time_since_polled}\t Last feedback {env.last_feedback}\n\n")

    print("Insert your action with 0 and 1 without commas. Ex: 1001")
    done_action = False
    while not done_action:
        action_str = input()
        if len(action_str) == k:
            action = np.array([float(s) for s in action_str])
            done_action = True
        else:
            print(f"The action does not have the correct length, it should be of length {k}. Try again!")

    obs, reward, done = env.step(action)

    print(f"action: {action}")
    print(f"reward: {reward}")

if done:
    print(f"Episode ended!")
    print(f"Successful transmissions: {env.successful_transmissions}")
    print(f"URLLC score: {1 - env.discarded_packets / env.received_packets.sum()}")
