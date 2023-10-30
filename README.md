#  Deep Reinforcement Learning for Uplink Scheduling in NOMA-URLLC Networks

This is the code associated with our paper "Deep Reinforcement Learning for Uplink Scheduling in NOMA-URLLC Networks". 

## Requirements
Requirements can be found in the requirements.txt file.

## Structure
The environment of the project is in nomaenv.py. The algorithms used for the experiments are in the algorithms folder. 

- baselines.py: code for the Random scheduler, Earliest Deadline First scheduler, Slotted ALOHA (GF access) and prior agent (scheduler using the prior only).
- bdq.py: code for the Branching Deep Q-networks algorithm.
- dqn.py: code for the DQN algorithm used as a scheduler.
- idqn.py: code for the independent DQN algorithm used as a multi-agent grant-free protocol.
- irdqn.py: code for the iDQN algorithm with a recurrent neural network in order to tackle partial observability.
- noma_ppo_rnn.py: code for the NOMA-PPO algorithm without the agent state and with a recurrent neural network to handle partial observability.
- noma_ppo.py: code for the NOMA-PPO algorithm (proposed version).

## Launch experiment
The template to run an experiment is in run_noma_ppo.py. Execute it with: 

```bash
python -m run_noma_ppo.py
```

