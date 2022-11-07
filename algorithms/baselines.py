import numpy as np
import random

class RandomScheduler:
    def __init__(self, env, nb_of_polled=None, verbose=False):
        self.env = env
        self.nb_of_polled = nb_of_polled
        self.verbose = verbose
    
    def act(self, state):
        if self.nb_of_polled is None:
            nb_of_polled = np.random.randint(0, self.env.k)
        else:
            nb_of_polled = self.nb_of_polled
        action_idx = random.sample(self.env.action_space, nb_of_polled)
        action = np.zeros(self.env.k)
        action[action_idx] = 1.
        return action
    
    def run(self, n_episodes):
        obs = self.env.reset()
        number_of_discarded = []
        number_of_received = []
        for _ in range(n_episodes):
            done = False
            obs = self.env.reset()

            while not done:
                action = self.act(obs)
                next_obs, reward, done = self.env.step(action)
                obs = next_obs

            number_of_received.append(self.env.received_packets.sum())
            number_of_discarded.append(self.env.discarded_packets)
        
        if self.verbose:
            print(f"Number of received packets: {np.sum(number_of_received)}")

        return 1 - np.sum(number_of_discarded) / np.sum(number_of_received)


class EarliestDeadlineFirstScheduler:
    def __init__(self, env, verbose=False):
        self.env = env
        self.verbose = verbose

    def act(self, state):
        agg_state = self.env.preprocess_state(self.env.current_state)
        n_packets = (agg_state >= 0).sum()
        if n_packets > 0:
            has_a_packet = (agg_state + 1).nonzero()[0]
            sorted_idx = agg_state[has_a_packet].argsort()[:self.env.max_simultaneous_devices]
            action_idx = has_a_packet[sorted_idx]
        else:
            action_idx = random.sample(self.env.action_space, 3)
        action = np.zeros(self.env.k)
        action[action_idx] = 1.
        return action

    def run(self, n_episodes):
        obs = self.env.reset()
        number_of_discarded = []
        number_of_received = []
        for _ in range(n_episodes):
            done = False
            obs = self.env.reset()

            while not done:
                action = self.act(obs)
                next_obs, reward, done = self.env.step(action)
                obs = next_obs

            number_of_received.append(self.env.received_packets.sum())
            number_of_discarded.append(self.env.discarded_packets)
        
        if self.verbose:
            print(f"Number of received packets: {np.sum(number_of_received)}")

        return 1 - np.sum(number_of_discarded) / np.sum(number_of_received)
    

class SlottedAloha:
    def __init__(self, env, transmission_prob, verbose=False):
        self.env = env
        self.transmission_prob = transmission_prob
        self.verbose = verbose
    
    def act(self, state):
        buffers = self.env.current_state.sum(1)
        actions = np.random.binomial(1, p=self.transmission_prob, size=self.env.k)
        actions[buffers == 0] = 0
        return actions

    def get_best_transmission_probs(self, n_episodes, transmission_prob_list):
        cv = []
        for tp in transmission_prob_list:
            self.transmission_prob = tp
            score = self.run(n_episodes)
            cv.append(np.mean(score))
        return cv
    
    def run(self, n_episodes):
        obs = self.env.reset()
        number_of_discarded = []
        number_of_received = []
        for _ in range(n_episodes):
            done = False
            obs = self.env.reset()

            while not done:
                action = self.act(obs)
                next_obs, reward, done = self.env.step(action)
                obs = next_obs

            number_of_received.append(self.env.received_packets.sum())
            number_of_discarded.append(self.env.discarded_packets)
        
        if self.verbose:
            print(f"Number of received packets: {np.sum(number_of_received)}")

        return 1 - np.sum(number_of_discarded) / np.sum(number_of_received)
