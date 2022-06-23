import numpy as np

class CentralizedEnvNomaDiscretePoissonURLLC:

    def __init__(self,
                n,
                lbda_per_slot,
                constraints,
                episode_length=100,
                max_simultaneous_devices=3,
                sense_period=10,
                verbose=False,
                penalization=0.1
    ):  
        self.n = n
        self.state_size = 2
        self.max_simultaneous_devices = max_simultaneous_devices
        self.sense_period = sense_period
        self.lbda_per_slot = lbda_per_slot
        self.constraints = constraints
        self.episode_length = episode_length
        self.verbose = verbose
        self.penalization = penalization

        self.observation_space = self.state_size * n
        self.action_space = list(range(n))
        self.current_state = np.zeros((n, np.max(self.constraints))) # List of vectors of size max_deadline
        
        # Set arrival times
        self.arrival_times = [[] for _ in range(self.n)]

        for i in range(n):
            t1 = np.random.rand()
            self.arrival_times[i].append(np.ceil(self.icdf(t1)))
        
        # Set agent's obs
        # Now: (state, last_time_since_polled)
        self.history = [np.zeros((self.n)) for _ in range(self.episode_length)]
        self.agent_obs = np.zeros((self.n))
        self.sensing_obs = np.zeros((self.n))

        self.timestep = 0
        self.discarded_packets = 0
        self.received_packets = np.copy(self.current_state).sum(1)
        self.successful_transmissions = 0

    def icdf(self, t):
        return - np.log(1-t) / self.lbda_per_slot
    
    def sense_devices(self, state):
        sensing_obs = [np.nonzero(buffer)[0].min() if len(np.nonzero(buffer)[0]) > 0 else -1 for buffer in state]
        return sensing_obs

    def step(self, action):
        # Action is a vector of 0 and 1
        del self.history[0]
        devices_polled = action.nonzero()[0]
        self.timestep += 1

        next_state = np.copy(self.current_state)
        next_sensing_obs = np.copy(self.sensing_obs)

        # Check what device has a packet
        has_a_packet = (self.current_state.sum(1) > 0) * 1.
        
        # Penalizing the reward
        if action.sum() > self.max_simultaneous_devices:
            penalty = action.sum() - self.max_simultaneous_devices
        else:
            penalty = 0

        # Executing the action
        m = has_a_packet[list(devices_polled)].sum() # Number of packets polled
        
        # # Update the agents' observation
        # next_sensing_obs -= 1
        # next_sensing_obs[next_sensing_obs < 0] = -1

        next_obs = action * has_a_packet

        if m == 0:
            reward = 0.
        elif m <= self.max_simultaneous_devices:
            reward = m
            self.successful_transmissions += m
            
            devices_polled_obs = []
            for d in devices_polled:
                col = next_state[d].nonzero()[0]
                if len(col) > 0:
                    devices_polled_obs.append(col.min())
                else:
                    devices_polled_obs.append(0)
                        

            next_state[devices_polled, devices_polled_obs] = 0.

        elif m > self.max_simultaneous_devices:
            reward = -1
            next_obs = -1 * action

        else:
            raise ValueError("m takes an impossible value")           

        # Penalize the reward for each device polled if superior to 3
        reward -=  self.penalization * penalty
        
        # Incrementation
        row, col = np.nonzero(next_state)
        new_col = col - 1
        expired = new_col < 0
        self.discarded_packets += expired.sum()
        new_col = new_col[~expired] # Remove the expired packets
        new_row = row[~expired]
        next_state[row, col] = 0
        next_state[new_row, new_col] = 1

        reward -= expired.sum() * self.penalization
        
        # Receive new packets
        for i in range(self.n):
            if self.arrival_times[i][-1] == self.timestep:
                # A packet is generated and we sample another interarrival time
                next_state[i, self.constraints[i] - 1] = 1.
                self.received_packets[i] += 1.
                t1 = np.random.rand()
                self.arrival_times[i].append(self.arrival_times[i][-1] + np.ceil(self.icdf(t1)))
        
        if self.timestep % self.sense_period == 0:
            next_sensing_obs = self.sense_devices(next_state)

        if self.verbose:
            print(f"Timestep \n{self.timestep}")
            print(f"State \n{self.current_state}")
            print(f"Action \n{action}")
            print(f"Devices polled \n{devices_polled}")
            print(f'Next state \n{next_state}')
            print(f'Agent next obs \n{next_obs}')
            print(f"Next sensing obs \n{next_sensing_obs}")
            print(f"Reward \n{reward}")
            print(f"Penalty\n {penalty}")
            print(f"Received packets \n{self.received_packets}")
            print(f"Discarded packets \n{self.discarded_packets}")
            print("")
        
        if (self.timestep >= self.episode_length) :    
            done = True
        else:
            done = False
            
        self.current_state = np.copy(next_state)
        self.agent_obs = np.copy(next_obs)
        self.history.append(next_obs)
        self.sensing_obs = np.copy(next_sensing_obs)
        
        return np.stack(self.history).astype(np.float32), next_sensing_obs, reward, done
        
    def reset(self):

        self.current_state = np.zeros((self.n, np.max(self.constraints)))        
        # Set arrival times
        self.arrival_times = [[] for _ in range(self.n)]

        for i in range(self.n):
            t1 = np.random.rand()
            self.arrival_times[i].append(np.ceil(self.icdf(t1)))
        
        # Set agent's obs
        # Now: (state, last_time_since_polled)
        self.history = [np.zeros((self.n)) for _ in range(self.episode_length)]
        self.agent_obs = np.zeros((self.n))
        self.sensing_obs = np.zeros((self.n))

        self.timestep = 0
        self.discarded_packets = 0
        self.received_packets = np.copy(self.current_state).sum(1)
        self.successful_transmissions = 0
                
        return np.stack(self.history), self.sensing_obs
